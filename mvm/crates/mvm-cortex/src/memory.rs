use crate::{MemoryConfig, MemoryContext, MemoryEvent};
use mvm_core::canonical::{
    EntityKey, FusedSceneFrame, SceneEntityKind, SceneEvent, SceneEventKind,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Small indexes make retrieval deterministic without introducing a store or
/// a learned dependency. They only reference entries still in `recent_events`.
#[derive(Default)]
struct EventIndex {
    by_entity: BTreeMap<EntityKey, VecDeque<u64>>,
    interaction_frequency_by_object_class: BTreeMap<String, u64>,
}

/// Runtime-only bounded episodic memory. Destruction drops the whole layer;
/// no method writes labels.jsonl or any other durable artifact.
pub(crate) struct EventMemoryLayer {
    recent_events: VecDeque<MemoryEvent>,
    index: EventIndex,
    config: MemoryConfig,
    held_since_ms: BTreeMap<EntityKey, i64>,
}

impl Default for EventMemoryLayer {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}

impl EventMemoryLayer {
    pub(crate) fn new(config: MemoryConfig) -> Self {
        Self {
            recent_events: VecDeque::with_capacity(config.capacity),
            index: EventIndex::default(),
            config,
            held_since_ms: BTreeMap::new(),
        }
    }

    pub(crate) fn update(
        &mut self,
        scene: &FusedSceneFrame,
        events: &[SceneEvent],
        now_unix_ms: i64,
    ) -> MemoryContext {
        for event in events {
            self.ingest(scene, event);
        }
        self.context_for(events, now_unix_ms)
    }

    fn ingest(&mut self, scene: &FusedSceneFrame, event: &SceneEvent) {
        // A withheld, unknown, or negative event cannot become a memory fact.
        let Some(confidence) = event.fact.confidence else {
            return;
        };
        if event.fact.value != Some(true) || !confidence.is_finite() {
            return;
        }

        let object_classes = event
            .entity_keys
            .iter()
            .filter_map(|key| scene.entities.iter().find(|entity| entity.key == *key))
            .filter_map(|entity| match &entity.kind {
                SceneEntityKind::Object(facts) => facts.class_name.value.clone(),
                _ => None,
            })
            .collect::<Vec<_>>();
        let contact_duration_ms = self.contact_duration(event);
        let memory_event = MemoryEvent {
            event_seq: event.event_seq,
            source_frame_id: event.source_frame_id,
            source_timestamp_unix_ms: event.source_timestamp_unix_ms,
            kind: event.kind,
            entities: event.entity_keys.clone(),
            confidence,
            object_classes,
            contact_duration_ms,
        };
        self.insert(memory_event);
    }

    fn contact_duration(&mut self, event: &SceneEvent) -> Option<u64> {
        match event.kind {
            SceneEventKind::ObjectHeld => {
                for entity in &event.entity_keys {
                    self.held_since_ms
                        .entry(entity.clone())
                        .or_insert(event.source_timestamp_unix_ms);
                }
                None
            }
            SceneEventKind::ObjectReleased => event.entity_keys.iter().find_map(|entity| {
                self.held_since_ms.remove(entity).and_then(|started_at| {
                    u64::try_from(event.source_timestamp_unix_ms.saturating_sub(started_at)).ok()
                })
            }),
            _ => None,
        }
    }

    fn insert(&mut self, event: MemoryEvent) {
        if self.config.capacity == 0 {
            return;
        }
        while self.recent_events.len() >= self.config.capacity {
            if let Some(evicted) = self.recent_events.pop_front()
                && evicted.kind == SceneEventKind::ObjectHeld
            {
                for entity in evicted.entities {
                    if self.held_since_ms.get(&entity) == Some(&evicted.source_timestamp_unix_ms) {
                        self.held_since_ms.remove(&entity);
                    }
                }
            }
        }
        self.recent_events.push_back(event);
        self.rebuild_index();
    }

    fn rebuild_index(&mut self) {
        self.index = EventIndex::default();
        for event in &self.recent_events {
            for entity in &event.entities {
                self.index
                    .by_entity
                    .entry(entity.clone())
                    .or_default()
                    .push_back(event.event_seq);
            }
            if is_interaction(event.kind) {
                for class_name in &event.object_classes {
                    *self
                        .index
                        .interaction_frequency_by_object_class
                        .entry(class_name.clone())
                        .or_default() += 1;
                }
            }
        }
    }

    fn context_for(&self, query: &[SceneEvent], now_unix_ms: i64) -> MemoryContext {
        let recent = self
            .recent_events
            .iter()
            .rev()
            .take(self.config.recent_limit)
            .cloned()
            .collect::<Vec<_>>();
        let query_entities: BTreeSet<_> = query
            .iter()
            .flat_map(|event| event.entity_keys.iter().cloned())
            .collect();
        let query_kinds: Vec<_> = query.iter().map(|event| event.kind).collect();
        let minimum_time = now_unix_ms.saturating_sub(self.config.time_window_ms);
        let mut relevant = Vec::new();
        let mut seen = BTreeSet::new();
        let indexed_entity_events: BTreeSet<_> = query_entities
            .iter()
            .flat_map(|entity| {
                self.index
                    .by_entity
                    .get(entity)
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .collect();
        for rank in 0..3 {
            for event in self.recent_events.iter().rev() {
                let matches_rank = match rank {
                    0 => indexed_entity_events.contains(&event.event_seq),
                    1 => query_kinds.contains(&event.kind),
                    _ => event.source_timestamp_unix_ms >= minimum_time,
                };
                if matches_rank && seen.insert(event.event_seq) {
                    relevant.push(event.clone());
                    if relevant.len() == self.config.relevant_limit {
                        break;
                    }
                }
            }
            if relevant.len() == self.config.relevant_limit {
                break;
            }
        }
        MemoryContext {
            recent,
            summary: self.summary(),
            relevant,
        }
    }

    fn summary(&self) -> String {
        let interactions = self
            .index
            .interaction_frequency_by_object_class
            .iter()
            .map(|(class_name, count)| format!("{class_name}:{count}"))
            .collect::<Vec<_>>();
        if interactions.is_empty() {
            format!("{} runtime events retained", self.recent_events.len())
        } else {
            format!(
                "{} runtime events retained; interaction frequency {}",
                self.recent_events.len(),
                interactions.join(", ")
            )
        }
    }
}

fn is_interaction(kind: SceneEventKind) -> bool {
    matches!(
        kind,
        SceneEventKind::ObjectTouched | SceneEventKind::ObjectHeld | SceneEventKind::ObjectReleased
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::canonical::{
        EventTailView, FactEvidenceRef, FactSource, FactStatus, FrameStamp, ObjectFacts,
        SceneEntity, SceneFact, SceneQuality, TrackDomain,
    };

    fn key(id: u32) -> EntityKey {
        EntityKey {
            run_id: "test".into(),
            track_domain: TrackDomain::Object,
            track_id: id,
            incarnation: 0,
        }
    }

    fn scene() -> FusedSceneFrame {
        let object = SceneEntity {
            key: key(9),
            display_id: 9,
            kind: SceneEntityKind::Object(ObjectFacts {
                class_name: SceneFact::claimed(
                    FactStatus::Observed,
                    "cup".into(),
                    1.0,
                    vec![FactEvidenceRef {
                        source: FactSource::Fusion,
                        source_frame_id: 10,
                        entity: Some(key(9)),
                        event_seq: None,
                    }],
                )
                .unwrap(),
                remembered: false,
            }),
            existence: SceneFact::unknown(mvm_core::canonical::FactReason::MissingSource),
            bbox: SceneFact::unknown(mvm_core::canonical::FactReason::MissingSource),
            position_3d: SceneFact::unknown(mvm_core::canonical::FactReason::MissingSource),
            kinematics: SceneFact::unknown(mvm_core::canonical::FactReason::MissingSource),
        };
        FusedSceneFrame {
            stamp: FrameStamp::new("test", 10, 1_000, 1_000).unwrap(),
            sources: Vec::new(),
            entities: vec![object],
            relations: Vec::new(),
            hand_interactions: Vec::new(),
            zones: Vec::new(),
            activities: Vec::new(),
            events: EventTailView::default(),
            anomaly: SceneFact::unknown(mvm_core::canonical::FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    fn event(seq: u64, kind: SceneEventKind, timestamp: i64) -> SceneEvent {
        SceneEvent {
            event_seq: seq,
            source_frame_id: seq,
            source_timestamp_unix_ms: timestamp,
            kind,
            entity_keys: vec![key(9)],
            fact: SceneFact::claimed(
                FactStatus::Temporal,
                true,
                0.9,
                vec![FactEvidenceRef {
                    source: FactSource::Fusion,
                    source_frame_id: seq,
                    entity: Some(key(9)),
                    event_seq: Some(seq),
                }],
            )
            .unwrap(),
            payload: None,
        }
    }

    #[test]
    fn retrieval_precision_at_five_is_one_for_entity_bound_interaction_replay() {
        let scene = scene();
        let mut memory = EventMemoryLayer::new(MemoryConfig {
            capacity: 16,
            ..MemoryConfig::default()
        });
        let mut history = vec![
            event(1, SceneEventKind::EntityAppeared, 1_000),
            event(2, SceneEventKind::ObjectTouched, 2_000),
            event(3, SceneEventKind::ObjectHeld, 3_000),
            event(4, SceneEventKind::ObjectReleased, 5_000),
        ];
        for seq in 5..10 {
            let mut unrelated = event(
                seq,
                SceneEventKind::EntityAppeared,
                i64::try_from(seq).unwrap() * 1_000,
            );
            unrelated.entity_keys = vec![key(10)];
            history.push(unrelated);
        }
        memory.update(&scene, &history, 10_000);
        let query = event(10, SceneEventKind::ObjectTouched, 11_000);
        let context = memory.update(&scene, &[query], 11_000);
        let hits = context
            .relevant
            .iter()
            .filter(|item| item.entities == vec![key(9)])
            .count();
        assert_eq!(context.relevant.len(), 5);
        assert_eq!(hits as f32 / context.relevant.len() as f32, 1.0);
        assert!(
            context
                .relevant
                .iter()
                .any(|item| item.contact_duration_ms == Some(2_000))
        );
        assert!(context.summary.contains("cup:4"));
    }

    #[test]
    fn withheld_events_do_not_enter_runtime_memory() {
        let scene = scene();
        let mut memory = EventMemoryLayer::default();
        let mut withheld = event(1, SceneEventKind::EntityAppeared, 1_000);
        withheld.fact = SceneFact::unknown(mvm_core::canonical::FactReason::MissingSource);
        let context = memory.update(&scene, &[withheld], 1_000);
        assert!(context.recent.is_empty());
        assert_eq!(context.summary, "0 runtime events retained");
    }

    #[test]
    fn eviction_rebuilds_indexes_and_keeps_memory_bounded() {
        let scene = scene();
        let mut memory = EventMemoryLayer::new(MemoryConfig {
            capacity: 2,
            ..MemoryConfig::default()
        });
        memory.update(
            &scene,
            &[
                event(1, SceneEventKind::ObjectTouched, 1_000),
                event(2, SceneEventKind::ObjectHeld, 2_000),
                event(3, SceneEventKind::ObjectReleased, 3_000),
            ],
            3_000,
        );
        assert_eq!(memory.recent_events.len(), 2);
        assert_eq!(
            memory.index.by_entity[&key(9)]
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert!(memory.summary().contains("cup:2"));
    }
}
