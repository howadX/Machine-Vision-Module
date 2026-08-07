//! Canonical immutable scene contract (phase 18.0).
//!
//! This module defines the facts substrate consumed by runtime sinks and the
//! future visual cortex. It deliberately contains no detector orchestration and
//! no raw RGB/depth/mask payloads. Wire timestamps use Unix milliseconds so
//! replay fixtures are portable; runtime monotonic clocks stay in the mesh.

use crate::{Activity, BBox, ObjectState, PersonState, Position3D, WorldState};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Deref;
use std::time::Instant;

/// One published camera-driven scene revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameStamp {
    pub run_id: String,
    pub frame_id: u64,
    pub captured_at_unix_ms: i64,
    pub published_at_unix_ms: i64,
}

impl FrameStamp {
    pub fn new(
        run_id: impl Into<String>,
        frame_id: u64,
        captured_at_unix_ms: i64,
        published_at_unix_ms: i64,
    ) -> Result<Self, CanonicalContractError> {
        let stamp = Self {
            run_id: run_id.into(),
            frame_id,
            captured_at_unix_ms,
            published_at_unix_ms,
        };
        if stamp.run_id.is_empty() {
            return Err(CanonicalContractError::new("run_id must not be empty"));
        }
        if stamp.published_at_unix_ms < stamp.captured_at_unix_ms {
            return Err(CanonicalContractError::new(
                "published timestamp precedes capture timestamp",
            ));
        }
        Ok(stamp)
    }
}

/// Producer of a scene fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactSource {
    Capture,
    Human,
    Scene,
    Depth,
    Tracker,
    Fusion,
    Activity,
    CameraState,
    Motion,
    Semantic,
    Spatial,
    Compatibility,
}

/// Freshness of one expert result relative to the fused camera revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFreshness {
    Fresh,
    Coasted,
    Stale,
    Unavailable,
}

/// Causal stamp for one expert output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceStamp {
    pub source: FactSource,
    pub source_frame_id: u64,
    pub produced_at_unix_ms: i64,
    pub age_frames: u64,
    pub freshness: SourceFreshness,
}

impl SourceStamp {
    /// Construct an available source before classifying it against a fused tick.
    pub fn at_frame(source: FactSource, source_frame_id: u64, produced_at_unix_ms: i64) -> Self {
        Self {
            source,
            source_frame_id,
            produced_at_unix_ms,
            age_frames: 0,
            freshness: SourceFreshness::Fresh,
        }
    }

    /// Classify freshness using frame budgets. Future-source claims are invalid:
    /// a late result may join a later tick, never rewrite an earlier one.
    pub fn relative_to(
        mut self,
        fused_frame_id: u64,
        fresh_through: u64,
        coasted_through: u64,
    ) -> Result<Self, CanonicalContractError> {
        if coasted_through < fresh_through {
            return Err(CanonicalContractError::new(
                "coasted freshness budget is smaller than fresh budget",
            ));
        }
        self.age_frames = fused_frame_id
            .checked_sub(self.source_frame_id)
            .ok_or_else(|| CanonicalContractError::new("source frame is newer than fused frame"))?;
        self.freshness = if self.age_frames <= fresh_through {
            SourceFreshness::Fresh
        } else if self.age_frames <= coasted_through {
            SourceFreshness::Coasted
        } else {
            SourceFreshness::Stale
        };
        Ok(self)
    }

    pub fn unavailable(source: FactSource) -> Self {
        Self {
            source,
            source_frame_id: 0,
            produced_at_unix_ms: 0,
            age_frames: 0,
            freshness: SourceFreshness::Unavailable,
        }
    }
}

/// Honesty status of an evidence-backed world fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactStatus {
    Observed,
    Derived,
    Temporal,
    Remembered,
    Weak,
    Withheld,
    Unknown,
}

impl FactStatus {
    pub fn is_asserted(self) -> bool {
        matches!(
            self,
            Self::Observed | Self::Derived | Self::Temporal | Self::Remembered | Self::Weak
        )
    }

    /// Withheld and unknown are missing claims, never negative supervision.
    pub fn can_be_negative_class(self) -> bool {
        !matches!(self, Self::Withheld | Self::Unknown)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactReason {
    InsufficientCoverage,
    LowConfidence,
    OutOfDistribution,
    StaleSource,
    MissingSource,
    Occluded,
    Uncalibrated,
    LowTexture,
    MotionBlur,
    FrameGap,
    DynamicSceneDominance,
    InsufficientParallax,
    OdometryConflict,
    Remembered,
    Other(String),
}

/// Image-plane camera motion supported by a pair of stamped RGB frames.
/// This is deliberately relative and camera-local: it is not a world pose,
/// odometry replacement, or claim that a zero-flow camera is physically fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelativeCameraMotionKind {
    StationaryConfigured,
    ImageTranslation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelativeCameraMotion {
    pub kind: RelativeCameraMotionKind,
    pub reference_frame: ReferenceFrame,
    pub displacement_px: [f32; 2],
    pub static_background_ratio: f32,
    pub candidate_count: u32,
    pub inlier_count: u32,
}

impl RelativeCameraMotion {
    fn validate(&self) -> Result<(), CanonicalContractError> {
        if self.reference_frame != ReferenceFrame::CameraLocal
            || !self.displacement_px.iter().all(|value| value.is_finite())
            || !self.static_background_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.static_background_ratio)
            || self.inlier_count > self.candidate_count
        {
            return Err(CanonicalContractError::new(
                "relative camera motion has invalid camera-local evidence",
            ));
        }
        Ok(())
    }
}

/// Stable reference from a fact to the expert observation that supports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactEvidenceRef {
    pub source: FactSource,
    pub source_frame_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<EntityKey>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_seq: Option<u64>,
}

/// One typed fact. Missing claims carry neither a value nor fake confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFact<T> {
    pub status: FactStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<FactEvidenceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<FactReason>,
}

impl<T> SceneFact<T> {
    pub fn claimed(
        status: FactStatus,
        value: T,
        confidence: f32,
        evidence: Vec<FactEvidenceRef>,
    ) -> Result<Self, CanonicalContractError> {
        let fact = Self {
            status,
            value: Some(value),
            confidence: Some(confidence),
            evidence,
            reason: None,
        };
        fact.validate()?;
        Ok(fact)
    }

    pub fn withheld(reason: FactReason, evidence: Vec<FactEvidenceRef>) -> Self {
        Self {
            status: FactStatus::Withheld,
            value: None,
            confidence: None,
            evidence,
            reason: Some(reason),
        }
    }

    pub fn unknown(reason: FactReason) -> Self {
        Self {
            status: FactStatus::Unknown,
            value: None,
            confidence: None,
            evidence: Vec::new(),
            reason: Some(reason),
        }
    }

    pub fn validate(&self) -> Result<(), CanonicalContractError> {
        if self.status.is_asserted() {
            if self.value.is_none() {
                return Err(CanonicalContractError::new(
                    "asserted fact must carry a value",
                ));
            }
            let confidence = self.confidence.ok_or_else(|| {
                CanonicalContractError::new("asserted fact must carry confidence")
            })?;
            if !(0.0..=1.0).contains(&confidence) || !confidence.is_finite() {
                return Err(CanonicalContractError::new(
                    "fact confidence must be finite and within [0,1]",
                ));
            }
            if self.evidence.is_empty() {
                return Err(CanonicalContractError::new(
                    "asserted fact must cite evidence",
                ));
            }
        } else if self.value.is_some() || self.confidence.is_some() {
            return Err(CanonicalContractError::new(
                "withheld/unknown fact must not carry value or confidence",
            ));
        }
        Ok(())
    }
}

/// Explicit coordinate authority for spatial facts. `CameraLocal` needs no
/// body transform; `AgentLocal` is valid only with current intrinsics and a
/// camera-to-body transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceFrame {
    CameraLocal,
    AgentLocal,
}

macro_rules! run_scoped_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name {
            pub run_id: String,
            pub local_id: u64,
        }

        impl $name {
            pub fn new(
                run_id: impl Into<String>,
                local_id: u64,
            ) -> Result<Self, CanonicalContractError> {
                let id = Self {
                    run_id: run_id.into(),
                    local_id,
                };
                if id.run_id.is_empty() {
                    return Err(CanonicalContractError::new(concat!(
                        stringify!($name),
                        " run_id must not be empty"
                    )));
                }
                Ok(id)
            }
        }
    };
}

run_scoped_id!(PersistentEntityHypothesisId);
run_scoped_id!(RegionId);
run_scoped_id!(ZoneId);
run_scoped_id!(SurfaceId);
run_scoped_id!(SpatialShardId);
run_scoped_id!(EventId);
run_scoped_id!(ClaimId);
run_scoped_id!(GraphEdgeId);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FrameId {
    pub run_id: String,
    pub frame_id: u64,
}

impl FrameId {
    pub fn new(run_id: impl Into<String>, frame_id: u64) -> Result<Self, CanonicalContractError> {
        let id = Self {
            run_id: run_id.into(),
            frame_id,
        };
        if id.run_id.is_empty() {
            return Err(CanonicalContractError::new(
                "FrameId run_id must not be empty",
            ));
        }
        Ok(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObservationId {
    pub run_id: String,
    pub source: FactSource,
    pub source_frame_id: u64,
    pub ordinal: u32,
}

impl ObservationId {
    pub fn new(
        run_id: impl Into<String>,
        source: FactSource,
        source_frame_id: u64,
        ordinal: u32,
    ) -> Result<Self, CanonicalContractError> {
        let id = Self {
            run_id: run_id.into(),
            source,
            source_frame_id,
            ordinal,
        };
        if id.run_id.is_empty() {
            return Err(CanonicalContractError::new(
                "ObservationId run_id must not be empty",
            ));
        }
        Ok(id)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub image_width: u32,
    pub image_height: u32,
    pub version: String,
}

impl CameraIntrinsics {
    fn validate(&self) -> Result<(), CanonicalContractError> {
        if !self.fx.is_finite()
            || !self.fy.is_finite()
            || self.fx <= 0.0
            || self.fy <= 0.0
            || !self.cx.is_finite()
            || !self.cy.is_finite()
            || self.image_width == 0
            || self.image_height == 0
            || self.version.is_empty()
        {
            return Err(CanonicalContractError::new(
                "camera intrinsics are incomplete or invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraToBodyExtrinsics {
    pub translation_m: [f32; 3],
    pub rotation_rpy_degrees: [f32; 3],
    pub version: String,
}

impl CameraToBodyExtrinsics {
    fn validate(&self) -> Result<(), CanonicalContractError> {
        if !self.translation_m.iter().all(|value| value.is_finite())
            || !self
                .rotation_rpy_degrees
                .iter()
                .all(|value| value.is_finite())
            || self.version.is_empty()
        {
            return Err(CanonicalContractError::new(
                "camera-to-body extrinsics are incomplete or invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialCalibration {
    pub intrinsics: SceneFact<CameraIntrinsics>,
    pub camera_to_body: SceneFact<CameraToBodyExtrinsics>,
}

impl Default for SpatialCalibration {
    fn default() -> Self {
        Self {
            intrinsics: SceneFact::unknown(FactReason::Uncalibrated),
            camera_to_body: SceneFact::unknown(FactReason::Uncalibrated),
        }
    }
}

impl SpatialCalibration {
    fn validate(&self, sources: &[SourceStamp]) -> Result<(), CanonicalContractError> {
        self.intrinsics.validate()?;
        self.camera_to_body.validate()?;
        validate_observed_evidence(&self.intrinsics, sources)?;
        validate_observed_evidence(&self.camera_to_body, sources)?;
        if let Some(value) = &self.intrinsics.value {
            value.validate()?;
        }
        if let Some(value) = &self.camera_to_body.value {
            value.validate()?;
        }
        Ok(())
    }

    fn supports_agent_local(&self) -> bool {
        self.intrinsics.status.is_asserted()
            && self.camera_to_body.status.is_asserted()
            && self.intrinsics.value.is_some()
            && self.camera_to_body.value.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    Floor,
    Wall,
    Ceiling,
    Support,
    Opening,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceGeometry {
    pub boundary_normalized: Vec<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plane: Option<[f32; 4]>,
}

impl SurfaceGeometry {
    fn validate(&self) -> Result<(), CanonicalContractError> {
        if self.boundary_normalized.iter().any(|point| {
            point
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        }) || self
            .plane
            .is_some_and(|plane| !plane.iter().all(|value| value.is_finite()))
        {
            return Err(CanonicalContractError::new(
                "surface geometry contains invalid coordinates",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceFact {
    pub id: SurfaceId,
    pub reference_frame: ReferenceFrame,
    pub kind: SceneFact<SurfaceKind>,
    pub geometry: SceneFact<SurfaceGeometry>,
}

impl SurfaceFact {
    fn validate(
        &self,
        run_id: &str,
        calibration: &SpatialCalibration,
        sources: &[SourceStamp],
    ) -> Result<(), CanonicalContractError> {
        validate_run_id(&self.id.run_id, run_id, "surface")?;
        self.kind.validate()?;
        self.geometry.validate()?;
        validate_observed_evidence(&self.kind, sources)?;
        validate_observed_evidence(&self.geometry, sources)?;
        if let Some(geometry) = &self.geometry.value {
            geometry.validate()?;
        }
        require_reference_authority(
            self.reference_frame,
            self.kind.status.is_asserted() || self.geometry.status.is_asserted(),
            calibration,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisibilityState {
    Visible,
    Occluded,
    OutOfView,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisibilityFact {
    pub target: GraphNodeId,
    pub reference_frame: ReferenceFrame,
    pub state: SceneFact<VisibilityState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub occluders: Vec<GraphNodeId>,
}

impl VisibilityFact {
    fn validate(
        &self,
        run_id: &str,
        calibration: &SpatialCalibration,
        sources: &[SourceStamp],
    ) -> Result<(), CanonicalContractError> {
        validate_run_id(self.target.run_id(), run_id, "visibility target")?;
        if self
            .occluders
            .iter()
            .any(|occluder| occluder.run_id() != run_id)
        {
            return Err(CanonicalContractError::new(
                "visibility occluder run_id differs from frame run_id",
            ));
        }
        self.state.validate()?;
        validate_observed_evidence(&self.state, sources)?;
        if self.state.value == Some(VisibilityState::Occluded) && self.occluders.is_empty() {
            return Err(CanonicalContractError::new(
                "occluded visibility requires an evidence-linked occluder",
            ));
        }
        require_reference_authority(
            self.reference_frame,
            self.state.status.is_asserted(),
            calibration,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccupancyState {
    Occupied,
    Free,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OccupancyCell {
    pub shard: SpatialShardId,
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellOccupancyFact {
    pub cell: OccupancyCell,
    pub reference_frame: ReferenceFrame,
    pub occupancy: SceneFact<OccupancyState>,
    pub visibility: SceneFact<VisibilityState>,
    pub clearance_m: SceneFact<f32>,
}

impl CellOccupancyFact {
    pub fn validate(
        &self,
        calibration: &SpatialCalibration,
        sources: &[SourceStamp],
    ) -> Result<(), CanonicalContractError> {
        self.occupancy.validate()?;
        self.visibility.validate()?;
        self.clearance_m.validate()?;
        validate_observed_evidence(&self.occupancy, sources)?;
        validate_observed_evidence(&self.visibility, sources)?;
        validate_observed_evidence(&self.clearance_m, sources)?;
        if let Some(clearance) = self.clearance_m.value
            && (!clearance.is_finite() || clearance < 0.0)
        {
            return Err(CanonicalContractError::new(
                "occupancy clearance must be finite and non-negative",
            ));
        }
        if !self.occupancy.status.is_asserted() && self.clearance_m.status.is_asserted() {
            return Err(CanonicalContractError::new(
                "missing occupancy cannot carry synthetic clearance",
            ));
        }
        if self.occupancy.value == Some(OccupancyState::Free) {
            if !self.visibility.status.is_asserted()
                || self.visibility.value != Some(VisibilityState::Visible)
            {
                return Err(CanonicalContractError::new(
                    "free occupancy requires positive visible-space evidence",
                ));
            }
            if self.occupancy.evidence.is_empty() || self.visibility.evidence.is_empty() {
                return Err(CanonicalContractError::new(
                    "free occupancy requires geometry and visibility evidence",
                ));
            }
        }
        require_reference_authority(
            self.reference_frame,
            self.occupancy.status.is_asserted()
                || self.visibility.status.is_asserted()
                || self.clearance_m.status.is_asserted(),
            calibration,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialRelationKind {
    On,
    Inside,
    Supports,
    Behind,
    Occludes,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialRelationFact {
    pub subject: GraphNodeId,
    pub object: GraphNodeId,
    pub kind: SpatialRelationKind,
    pub reference_frame: ReferenceFrame,
    pub fact: SceneFact<bool>,
}

impl SpatialRelationFact {
    pub fn validate(
        &self,
        run_id: &str,
        calibration: &SpatialCalibration,
        sources: &[SourceStamp],
    ) -> Result<(), CanonicalContractError> {
        validate_run_id(self.subject.run_id(), run_id, "spatial relation subject")?;
        validate_run_id(self.object.run_id(), run_id, "spatial relation object")?;
        self.fact.validate()?;
        validate_observed_evidence(&self.fact, sources)?;
        require_reference_authority(
            self.reference_frame,
            self.fact.status.is_asserted(),
            calibration,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SpatialQuality {
    pub surface_count: u32,
    pub visibility_count: u32,
    pub occupancy_cell_count: u32,
    pub relation_count: u32,
    pub unknown_facts: u32,
    pub withheld_facts: u32,
    pub stale_sources: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialScene {
    #[serde(default)]
    pub calibration: SpatialCalibration,
    /// Phase 19.1 relative camera-state evidence. Missing/withheld values never
    /// imply a fixed camera or zero world motion.
    #[serde(default = "unknown_camera_motion")]
    pub camera_motion: SceneFact<RelativeCameraMotion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub surfaces: Vec<SurfaceFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visibility: Vec<VisibilityFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub occupancy: Vec<CellOccupancyFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<SpatialRelationFact>,
    #[serde(default)]
    pub quality: SpatialQuality,
}

impl Default for SpatialScene {
    fn default() -> Self {
        Self {
            calibration: SpatialCalibration::default(),
            camera_motion: unknown_camera_motion(),
            surfaces: Vec::new(),
            visibility: Vec::new(),
            occupancy: Vec::new(),
            relations: Vec::new(),
            quality: SpatialQuality::default(),
        }
    }
}

impl SpatialScene {
    pub fn is_empty(&self) -> bool {
        self.surfaces.is_empty()
            && self.visibility.is_empty()
            && self.occupancy.is_empty()
            && self.relations.is_empty()
            && self.camera_motion.status == FactStatus::Unknown
            && self.camera_motion.reason == Some(FactReason::MissingSource)
            && !self.calibration.intrinsics.status.is_asserted()
            && !self.calibration.camera_to_body.status.is_asserted()
    }

    pub fn canonicalize(&mut self) {
        self.surfaces.sort_by(|left, right| left.id.cmp(&right.id));
        for fact in &mut self.visibility {
            fact.occluders.sort();
        }
        self.visibility.sort_by(|left, right| {
            (&left.target, left.reference_frame).cmp(&(&right.target, right.reference_frame))
        });
        self.occupancy
            .sort_by(|left, right| left.cell.cmp(&right.cell));
        self.relations.sort_by(|left, right| {
            (&left.subject, &left.object, left.kind, left.reference_frame).cmp(&(
                &right.subject,
                &right.object,
                right.kind,
                right.reference_frame,
            ))
        });
        self.refresh_quality();
    }

    pub fn validate(
        &self,
        run_id: &str,
        sources: &[SourceStamp],
    ) -> Result<(), CanonicalContractError> {
        self.calibration.validate(sources)?;
        self.camera_motion.validate()?;
        validate_camera_motion_evidence(&self.camera_motion, sources)?;
        if let Some(motion) = &self.camera_motion.value {
            motion.validate()?;
        }
        for surface in &self.surfaces {
            surface.validate(run_id, &self.calibration, sources)?;
        }
        for visibility in &self.visibility {
            visibility.validate(run_id, &self.calibration, sources)?;
        }
        for occupancy in &self.occupancy {
            validate_run_id(&occupancy.cell.shard.run_id, run_id, "occupancy shard")?;
            occupancy.validate(&self.calibration, sources)?;
        }
        for relation in &self.relations {
            relation.validate(run_id, &self.calibration, sources)?;
        }
        Ok(())
    }

    fn refresh_quality(&mut self) {
        self.quality.surface_count = saturating_u32(self.surfaces.len());
        self.quality.visibility_count = saturating_u32(self.visibility.len());
        self.quality.occupancy_cell_count = saturating_u32(self.occupancy.len());
        self.quality.relation_count = saturating_u32(self.relations.len());
        let statuses = std::iter::once(self.camera_motion.status).chain(
            self.surfaces
                .iter()
                .flat_map(|fact| [fact.kind.status, fact.geometry.status])
                .chain(self.visibility.iter().map(|fact| fact.state.status))
                .chain(self.occupancy.iter().flat_map(|fact| {
                    [
                        fact.occupancy.status,
                        fact.visibility.status,
                        fact.clearance_m.status,
                    ]
                }))
                .chain(self.relations.iter().map(|fact| fact.fact.status)),
        );
        let (unknown, withheld) = statuses.fold((0_u32, 0_u32), |counts, status| match status {
            FactStatus::Unknown => (counts.0.saturating_add(1), counts.1),
            FactStatus::Withheld => (counts.0, counts.1.saturating_add(1)),
            _ => counts,
        });
        self.quality.unknown_facts = unknown;
        self.quality.withheld_facts = withheld;
    }
}

fn unknown_camera_motion() -> SceneFact<RelativeCameraMotion> {
    SceneFact::unknown(FactReason::MissingSource)
}

fn validate_camera_motion_evidence(
    fact: &SceneFact<RelativeCameraMotion>,
    sources: &[SourceStamp],
) -> Result<(), CanonicalContractError> {
    if !fact.status.is_asserted() {
        return Ok(());
    }
    if fact.evidence.is_empty()
        || !fact
            .evidence
            .iter()
            .any(|evidence| evidence.source == FactSource::Motion)
    {
        return Err(CanonicalContractError::new(
            "asserted camera motion requires motion evidence",
        ));
    }
    for evidence in &fact.evidence {
        let source = sources.iter().find(|source| {
            source.source == evidence.source && source.source_frame_id == evidence.source_frame_id
        });
        if !source.is_some_and(|source| source.freshness == SourceFreshness::Fresh) {
            return Err(CanonicalContractError::new(
                "asserted camera motion cites a missing or non-fresh source",
            ));
        }
    }
    if fact.value.as_ref().is_some_and(|motion| {
        motion.kind == RelativeCameraMotionKind::StationaryConfigured
            && !fact
                .evidence
                .iter()
                .any(|evidence| evidence.source == FactSource::CameraState)
    }) {
        return Err(CanonicalContractError::new(
            "stationary camera motion requires explicit camera-state evidence",
        ));
    }
    Ok(())
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn validate_run_id(
    candidate: &str,
    expected: &str,
    label: &str,
) -> Result<(), CanonicalContractError> {
    if candidate != expected {
        return Err(CanonicalContractError::new(format!(
            "{label} run_id differs from frame run_id"
        )));
    }
    Ok(())
}

fn require_reference_authority(
    reference_frame: ReferenceFrame,
    asserted: bool,
    calibration: &SpatialCalibration,
) -> Result<(), CanonicalContractError> {
    if asserted
        && reference_frame == ReferenceFrame::AgentLocal
        && !calibration.supports_agent_local()
    {
        return Err(CanonicalContractError::new(
            "agent-local spatial claim requires intrinsics and camera-to-body extrinsics",
        ));
    }
    Ok(())
}

fn validate_observed_evidence<T>(
    fact: &SceneFact<T>,
    sources: &[SourceStamp],
) -> Result<(), CanonicalContractError> {
    if fact.status != FactStatus::Observed {
        return Ok(());
    }
    for evidence in &fact.evidence {
        let source = sources.iter().find(|source| {
            source.source == evidence.source && source.source_frame_id == evidence.source_frame_id
        });
        if !source.is_some_and(|source| source.freshness == SourceFreshness::Fresh) {
            return Err(CanonicalContractError::new(
                "observed spatial fact cites a missing or non-fresh source",
            ));
        }
    }
    Ok(())
}

/// Tracker namespace. Local ids are only unique within one domain and incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackDomain {
    Person,
    Object,
    Animal,
}

/// Run-scoped identity of one concrete tracker incarnation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityKey {
    pub run_id: String,
    pub track_domain: TrackDomain,
    pub track_id: u32,
    pub incarnation: u32,
}

/// Stable node identity for bounded graph projections. Existing entity
/// identity is reused directly; spatial node IDs are run-scoped newtypes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum GraphNodeId {
    Entity(EntityKey),
    PersistentEntityHypothesis(PersistentEntityHypothesisId),
    Region(RegionId),
    Zone(ZoneId),
    Surface(SurfaceId),
    SpatialShard(SpatialShardId),
}

impl GraphNodeId {
    pub fn run_id(&self) -> &str {
        match self {
            Self::Entity(id) => &id.run_id,
            Self::PersistentEntityHypothesis(id) => &id.run_id,
            Self::Region(id) => &id.run_id,
            Self::Zone(id) => &id.run_id,
            Self::Surface(id) => &id.run_id,
            Self::SpatialShard(id) => &id.run_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: GraphNodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeKind {
    Contains,
    MemberOf,
    SpatialRelation(SpatialRelationKind),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: GraphEdgeId,
    pub from: GraphNodeId,
    pub to: GraphNodeId,
    pub kind: GraphEdgeKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum GraphOperation {
    NodeAdded {
        node: GraphNode,
    },
    NodeUpdated {
        node: GraphNode,
    },
    NodeRetired {
        node: GraphNodeId,
    },
    EdgeAdded {
        edge: GraphEdge,
    },
    EdgeChanged {
        edge: GraphEdge,
    },
    EdgeExpired {
        edge: GraphEdgeId,
    },
    ObservationLinked {
        node: GraphNodeId,
        observation: ObservationId,
        evidence: Vec<FactEvidenceRef>,
    },
    ContradictionRecorded {
        node: GraphNodeId,
        observation: ObservationId,
    },
    ReconfirmationRecorded {
        node: GraphNodeId,
        observation: ObservationId,
    },
}

impl GraphOperation {
    fn canonical_key(&self) -> (u8, String) {
        let rank = match self {
            Self::NodeAdded { .. } => 0,
            Self::NodeUpdated { .. } => 1,
            Self::EdgeAdded { .. } => 2,
            Self::EdgeChanged { .. } => 3,
            Self::ObservationLinked { .. } => 4,
            Self::ContradictionRecorded { .. } => 5,
            Self::ReconfirmationRecorded { .. } => 6,
            Self::EdgeExpired { .. } => 7,
            Self::NodeRetired { .. } => 8,
        };
        (rank, format!("{self:?}"))
    }

    fn validate_run(&self, run_id: &str) -> Result<(), CanonicalContractError> {
        let valid = match self {
            Self::NodeAdded { node } | Self::NodeUpdated { node } => node.id.run_id() == run_id,
            Self::NodeRetired { node } => node.run_id() == run_id,
            Self::EdgeAdded { edge } | Self::EdgeChanged { edge } => {
                edge.id.run_id == run_id
                    && edge.from.run_id() == run_id
                    && edge.to.run_id() == run_id
            }
            Self::EdgeExpired { edge } => edge.run_id == run_id,
            Self::ObservationLinked {
                node,
                observation,
                evidence,
            } => {
                node.run_id() == run_id
                    && observation.run_id == run_id
                    && !evidence.is_empty()
                    && evidence.iter().all(|item| {
                        item.source == observation.source
                            && item.source_frame_id == observation.source_frame_id
                    })
            }
            Self::ContradictionRecorded { node, observation }
            | Self::ReconfirmationRecorded { node, observation } => {
                node.run_id() == run_id && observation.run_id == run_id
            }
        };
        if !valid {
            return Err(CanonicalContractError::new(
                "graph operation has invalid run scope or evidence link",
            ));
        }
        Ok(())
    }
}

/// Immutable change envelope between two revisions of a bounded graph
/// projection. Source observations are link-only and never edited by deltas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphDelta {
    pub run_id: String,
    pub from_revision: u64,
    pub to_revision: u64,
    pub source_frame: FrameId,
    pub operations: Vec<GraphOperation>,
}

impl GraphDelta {
    pub const MAX_OPERATIONS: usize = 256;

    pub fn canonicalize(&mut self) {
        self.operations.sort_by_key(GraphOperation::canonical_key);
    }

    pub fn validate(&self) -> Result<(), CanonicalContractError> {
        if self.run_id.is_empty() || self.source_frame.run_id != self.run_id {
            return Err(CanonicalContractError::new(
                "graph delta and source frame must share a non-empty run_id",
            ));
        }
        if self.to_revision <= self.from_revision {
            return Err(CanonicalContractError::new(
                "graph delta revision must advance",
            ));
        }
        if self.operations.len() > Self::MAX_OPERATIONS {
            return Err(CanonicalContractError::new(
                "graph delta exceeds bounded operation capacity",
            ));
        }
        for operation in &self.operations {
            operation.validate_run(&self.run_id)?;
        }
        Ok(())
    }

    /// Compose contiguous deltas for a slower sink. Batching changes delivery
    /// cadence, not replay semantics; a true revision gap is rejected.
    pub fn compose(deltas: &[Self]) -> Result<Self, CanonicalContractError> {
        let first = deltas
            .first()
            .ok_or_else(|| CanonicalContractError::new("cannot compose an empty delta list"))?;
        let last = deltas.last().expect("checked non-empty");
        let mut previous = first.from_revision;
        let mut operations = Vec::new();
        for delta in deltas {
            delta.validate()?;
            if delta.run_id != first.run_id || delta.from_revision != previous {
                return Err(CanonicalContractError::new(
                    "graph delta composition contains a run or revision gap",
                ));
            }
            previous = delta.to_revision;
            operations.extend(delta.operations.iter().cloned());
        }
        let mut composed = Self {
            run_id: first.run_id.clone(),
            from_revision: first.from_revision,
            to_revision: last.to_revision,
            source_frame: last.source_frame.clone(),
            operations,
        };
        composed.canonicalize();
        composed.validate()?;
        Ok(composed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphObservationLink {
    pub node: GraphNodeId,
    pub observation: ObservationId,
    pub evidence: Vec<FactEvidenceRef>,
}

/// Rebuildable bounded graph projection. Retirement and expiry are tombstones;
/// linked source observations remain immutable and queryable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphProjection {
    run_id: String,
    revision: u64,
    nodes: Vec<GraphNode>,
    retired_nodes: BTreeSet<GraphNodeId>,
    edges: Vec<GraphEdge>,
    expired_edges: BTreeSet<GraphEdgeId>,
    observations: Vec<GraphObservationLink>,
    contradictions: BTreeSet<(GraphNodeId, ObservationId)>,
    reconfirmations: BTreeSet<(GraphNodeId, ObservationId)>,
}

impl GraphProjection {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            revision: 0,
            nodes: Vec::new(),
            retired_nodes: BTreeSet::new(),
            edges: Vec::new(),
            expired_edges: BTreeSet::new(),
            observations: Vec::new(),
            contradictions: BTreeSet::new(),
            reconfirmations: BTreeSet::new(),
        }
    }

    pub fn apply(&mut self, delta: &GraphDelta) -> Result<(), CanonicalContractError> {
        delta.validate()?;
        if delta.run_id != self.run_id || delta.from_revision != self.revision {
            return Err(CanonicalContractError::new(
                "graph projection received a foreign run or revision gap",
            ));
        }
        for operation in &delta.operations {
            match operation {
                GraphOperation::NodeAdded { node } | GraphOperation::NodeUpdated { node } => {
                    if let Some(existing) = self.nodes.iter_mut().find(|item| item.id == node.id) {
                        *existing = node.clone();
                    } else {
                        self.nodes.push(node.clone());
                    }
                }
                GraphOperation::NodeRetired { node } => {
                    self.retired_nodes.insert(node.clone());
                }
                GraphOperation::EdgeAdded { edge } | GraphOperation::EdgeChanged { edge } => {
                    if let Some(existing) = self.edges.iter_mut().find(|item| item.id == edge.id) {
                        *existing = edge.clone();
                    } else {
                        self.edges.push(edge.clone());
                    }
                }
                GraphOperation::EdgeExpired { edge } => {
                    self.expired_edges.insert(edge.clone());
                }
                GraphOperation::ObservationLinked {
                    node,
                    observation,
                    evidence,
                } => {
                    let link = GraphObservationLink {
                        node: node.clone(),
                        observation: observation.clone(),
                        evidence: evidence.clone(),
                    };
                    if let Some(existing) = self
                        .observations
                        .iter()
                        .find(|item| item.observation == *observation)
                    {
                        if existing != &link {
                            return Err(CanonicalContractError::new(
                                "graph delta attempts to rewrite source evidence",
                            ));
                        }
                    } else {
                        self.observations.push(link);
                    }
                }
                GraphOperation::ContradictionRecorded { node, observation } => {
                    self.contradictions
                        .insert((node.clone(), observation.clone()));
                }
                GraphOperation::ReconfirmationRecorded { node, observation } => {
                    self.reconfirmations
                        .insert((node.clone(), observation.clone()));
                }
            }
        }
        self.nodes.sort_by(|left, right| left.id.cmp(&right.id));
        self.edges.sort_by(|left, right| left.id.cmp(&right.id));
        self.observations
            .sort_by(|left, right| left.observation.cmp(&right.observation));
        self.revision = delta.to_revision;
        Ok(())
    }

    pub fn observations(&self) -> impl Iterator<Item = &GraphObservationLink> {
        self.observations.iter()
    }

    pub fn retired_nodes(&self) -> impl Iterator<Item = &GraphNodeId> {
        self.retired_nodes.iter()
    }
}

#[derive(Debug, Clone, Copy)]
struct IncarnationSlot {
    incarnation: u32,
    active: bool,
}

/// Assigns a fresh canonical identity whenever a tracker reuses a local id.
#[derive(Debug, Clone)]
pub struct IncarnationRegistry {
    run_id: String,
    slots: BTreeMap<(TrackDomain, u32), IncarnationSlot>,
}

impl IncarnationRegistry {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            slots: BTreeMap::new(),
        }
    }

    pub fn activate(&mut self, domain: TrackDomain, track_id: u32) -> EntityKey {
        let slot = self
            .slots
            .entry((domain, track_id))
            .and_modify(|slot| {
                if !slot.active {
                    slot.incarnation = slot.incarnation.saturating_add(1);
                    slot.active = true;
                }
            })
            .or_insert(IncarnationSlot {
                incarnation: 0,
                active: true,
            });
        EntityKey {
            run_id: self.run_id.clone(),
            track_domain: domain,
            track_id,
            incarnation: slot.incarnation,
        }
    }

    pub fn retire(&mut self, key: &EntityKey) -> bool {
        if key.run_id != self.run_id {
            return false;
        }
        match self.slots.get_mut(&(key.track_domain, key.track_id)) {
            Some(slot) if slot.active && slot.incarnation == key.incarnation => {
                slot.active = false;
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Existence {
    Present,
    Coasted,
    Occluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Kinematics {
    pub bbox_velocity: [f32; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity_3d_mps: Option<[f32; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GalleryIdentity {
    pub gallery_id: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonFacts {
    pub gallery_identity: SceneFact<GalleryIdentity>,
    pub posture: SceneFact<String>,
    pub emotion: SceneFact<String>,
    /// Compact sensory observations, boxed to keep object/animal scene entities
    /// cheap while retaining an additive canonical contract.
    #[serde(default)]
    pub sensory: Box<PersonSensoryFacts>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonSensoryFacts {
    /// Compact observation regime. This is evidence about visible body coverage,
    /// not an inferred posture.
    #[serde(default = "unknown_fact")]
    pub body_coverage: SceneFact<BodyCoverage>,
    /// Visual head pose only. It is never a gaze target, attention, or intent.
    #[serde(default = "unknown_fact")]
    pub head_orientation: SceneFact<HeadOrientationFact>,
    /// Bounded hand summary; raw hand landmarks remain outside canonical facts.
    #[serde(default = "unknown_fact")]
    pub hands: SceneFact<Vec<HandFact>>,
}

impl Default for PersonSensoryFacts {
    fn default() -> Self {
        Self {
            body_coverage: unknown_fact(),
            head_orientation: unknown_fact(),
            hands: unknown_fact(),
        }
    }
}

fn unknown_fact<T>() -> SceneFact<T> {
    SceneFact::unknown(FactReason::MissingSource)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyCoverage {
    Full,
    Upper,
    Head,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadFacing {
    Camera,
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadOrientationFact {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub yaw_deg: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facing: Option<HeadFacing>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandGesture {
    Open,
    Fist,
    Point,
    Pinch,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandFact {
    pub side: HandSide,
    pub presence: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gesture: Option<HandGesture>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectFacts {
    pub class_name: SceneFact<String>,
    pub remembered: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimalFacts {
    pub class_name: SceneFact<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "facts", rename_all = "snake_case")]
pub enum SceneEntityKind {
    Person(PersonFacts),
    Object(ObjectFacts),
    Animal(AnimalFacts),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntity {
    pub key: EntityKey,
    pub display_id: u32,
    pub kind: SceneEntityKind,
    pub existence: SceneFact<Existence>,
    pub bbox: SceneFact<BBox>,
    pub position_3d: SceneFact<Position3D>,
    pub kinematics: SceneFact<Kinematics>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneRelationKind {
    Near,
    Overlapping,
    PartOf,
    Touching,
    HeldBy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneRelation {
    pub a: EntityKey,
    pub b: EntityKey,
    pub kind: SceneRelationKind,
    pub distance_m: SceneFact<f32>,
    pub overlap: SceneFact<f32>,
}

/// Hand-to-object candidate derived by Fusion from present hand evidence. A
/// candidate is deliberately not equivalent to a physical hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandInteractionKind {
    Near,
    TouchingCandidate,
    HoldingCandidate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandInteractionFact {
    pub person: EntityKey,
    pub hand_side: HandSide,
    pub target_entity: EntityKey,
    pub kind: HandInteractionKind,
    pub fact: SceneFact<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneZone {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub members: Vec<EntityKey>,
    pub age_frames: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneActivity {
    pub label: String,
    pub participants: Vec<EntityKey>,
    pub fact: SceneFact<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneEventKind {
    EntityAppeared,
    EntityDisappeared,
    PersonEntered,
    PersonExited,
    PersonSatDown,
    PersonStoodUp,
    ObjectTouched,
    ObjectHeld,
    ObjectReleased,
    ObjectMoved,
    ActivityStarted,
    ActivityEnded,
    AnomalySpike,
    ZoneOccupied,
    ZoneFreed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SceneEventPayload {
    ActivityEpisode {
        label: String,
        started_at_s: f64,
        ended_at_s: f64,
        duration_s: f64,
        peak_confidence: f32,
        stable_participants: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEvent {
    pub event_seq: u64,
    pub source_frame_id: u64,
    pub source_timestamp_unix_ms: i64,
    pub kind: SceneEventKind,
    pub entity_keys: Vec<EntityKey>,
    pub fact: SceneFact<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<SceneEventPayload>,
}

/// Bounded immutable event suffix published with a scene frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EventTailView {
    pub first_seq: u64,
    pub next_seq: u64,
    pub events: Vec<SceneEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventGap {
    pub requested_after: u64,
    pub available_from: u64,
}

#[derive(Debug)]
pub struct EventDelta<'a> {
    pub events: Vec<&'a SceneEvent>,
    pub next_cursor: Option<u64>,
    pub gap: Option<EventGap>,
}

impl EventTailView {
    /// Read non-destructively after the consumer's last seen sequence.
    ///
    /// A cursor older than the bounded tail produces an explicit gap while still
    /// returning every event that remains available. Re-reading the same cursor
    /// is idempotent and never mutates the tail.
    pub fn delta_after(&self, cursor: Option<u64>) -> EventDelta<'_> {
        let gap = cursor.and_then(|requested_after| {
            if requested_after.saturating_add(1) < self.first_seq {
                Some(EventGap {
                    requested_after,
                    available_from: self.first_seq,
                })
            } else {
                None
            }
        });
        let events: Vec<&SceneEvent> = self
            .events
            .iter()
            .filter(|event| cursor.is_none_or(|seq| event.event_seq > seq))
            .collect();
        let next_cursor = events.last().map(|event| event.event_seq).or(cursor);
        EventDelta {
            events,
            next_cursor,
            gap,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SceneQuality {
    pub withheld_facts: u32,
    pub unknown_facts: u32,
    pub stale_sources: u32,
    pub event_gap: bool,
}

/// The single immutable authority for all facts of a published scene revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedSceneFrame {
    pub stamp: FrameStamp,
    pub sources: Vec<SourceStamp>,
    pub entities: Vec<SceneEntity>,
    pub relations: Vec<SceneRelation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hand_interactions: Vec<HandInteractionFact>,
    pub zones: Vec<SceneZone>,
    pub activities: Vec<SceneActivity>,
    /// Additive Phase 19 spatial substrate. Empty in 19.0 production runtime;
    /// old frames deserialize to the same explicit unknown spatial state.
    #[serde(default, skip_serializing_if = "SpatialScene::is_empty")]
    pub spatial: SpatialScene,
    pub events: EventTailView,
    pub anomaly: SceneFact<f32>,
    pub quality: SceneQuality,
}

impl FusedSceneFrame {
    /// Stable ordering is part of the replay/wire contract.
    pub fn canonicalize(&mut self) {
        self.sources.sort_by_key(|source| source.source);
        self.entities.sort_by(|a, b| a.key.cmp(&b.key));
        self.relations
            .sort_by(|a, b| (&a.a, &a.b, a.kind).cmp(&(&b.a, &b.b, b.kind)));
        self.hand_interactions.sort_by(|a, b| {
            (&a.person, a.hand_side, &a.target_entity, a.kind).cmp(&(
                &b.person,
                b.hand_side,
                &b.target_entity,
                b.kind,
            ))
        });
        for zone in &mut self.zones {
            zone.members.sort();
        }
        self.zones.sort_by_key(|zone| zone.id);
        for activity in &mut self.activities {
            activity.participants.sort();
        }
        self.activities
            .sort_by(|a, b| (&a.label, &a.participants).cmp(&(&b.label, &b.participants)));
        self.spatial.canonicalize();
        self.spatial.quality.stale_sources = saturating_u32(
            self.sources
                .iter()
                .filter(|source| {
                    matches!(
                        source.source,
                        FactSource::Depth
                            | FactSource::CameraState
                            | FactSource::Motion
                            | FactSource::Semantic
                            | FactSource::Spatial
                    ) && source.freshness == SourceFreshness::Stale
                })
                .count(),
        );
        for event in &mut self.events.events {
            event.entity_keys.sort();
        }
        self.events.events.sort_by_key(|event| event.event_seq);
    }

    pub fn validate(&self) -> Result<(), CanonicalContractError> {
        for source in &self.sources {
            if source.freshness != SourceFreshness::Unavailable
                && source.source_frame_id > self.stamp.frame_id
            {
                return Err(CanonicalContractError::new(
                    "source frame is newer than fused frame",
                ));
            }
        }
        self.anomaly.validate()?;
        for entity in &self.entities {
            if entity.key.run_id != self.stamp.run_id {
                return Err(CanonicalContractError::new(
                    "entity key run_id differs from frame run_id",
                ));
            }
            entity.existence.validate()?;
            entity.bbox.validate()?;
            entity.position_3d.validate()?;
            entity.kinematics.validate()?;
            match &entity.kind {
                SceneEntityKind::Person(facts) => {
                    facts.gallery_identity.validate()?;
                    facts.posture.validate()?;
                    facts.emotion.validate()?;
                    facts.sensory.body_coverage.validate()?;
                    facts.sensory.head_orientation.validate()?;
                    facts.sensory.hands.validate()?;
                }
                SceneEntityKind::Object(facts) => facts.class_name.validate()?,
                SceneEntityKind::Animal(facts) => facts.class_name.validate()?,
            }
        }
        for relation in &self.relations {
            if relation.a.run_id != self.stamp.run_id || relation.b.run_id != self.stamp.run_id {
                return Err(CanonicalContractError::new(
                    "relation endpoint run_id differs from frame run_id",
                ));
            }
            relation.distance_m.validate()?;
            relation.overlap.validate()?;
        }
        for interaction in &self.hand_interactions {
            if interaction.person.run_id != self.stamp.run_id
                || interaction.target_entity.run_id != self.stamp.run_id
            {
                return Err(CanonicalContractError::new(
                    "hand interaction endpoint run_id differs from frame run_id",
                ));
            }
            interaction.fact.validate()?;
        }
        for zone in &self.zones {
            if zone
                .members
                .iter()
                .any(|member| member.run_id != self.stamp.run_id)
            {
                return Err(CanonicalContractError::new(
                    "zone member run_id differs from frame run_id",
                ));
            }
        }
        for activity in &self.activities {
            activity.fact.validate()?;
            if activity
                .participants
                .iter()
                .any(|participant| participant.run_id != self.stamp.run_id)
            {
                return Err(CanonicalContractError::new(
                    "activity participant run_id differs from frame run_id",
                ));
            }
        }
        self.spatial.validate(&self.stamp.run_id, &self.sources)?;
        if let Some(first) = self.events.events.first()
            && first.event_seq != self.events.first_seq
        {
            return Err(CanonicalContractError::new(
                "event tail first_seq differs from first event",
            ));
        }
        if let Some(last) = self.events.events.last() {
            if last.event_seq.saturating_add(1) != self.events.next_seq {
                return Err(CanonicalContractError::new(
                    "event tail next_seq does not follow last event",
                ));
            }
        } else if self.events.first_seq != self.events.next_seq {
            return Err(CanonicalContractError::new(
                "empty event tail must have first_seq equal to next_seq",
            ));
        }
        let mut previous = None;
        for event in &self.events.events {
            event.fact.validate()?;
            match (&event.kind, &event.payload) {
                (
                    SceneEventKind::ActivityEnded,
                    Some(SceneEventPayload::ActivityEpisode { .. }),
                )
                | (_, None) => {}
                _ => {
                    return Err(CanonicalContractError::new(
                        "scene event payload does not match event kind",
                    ));
                }
            }
            if event.source_frame_id > self.stamp.frame_id {
                return Err(CanonicalContractError::new(
                    "event source frame is newer than fused frame",
                ));
            }
            if previous.is_some_and(|seq| event.event_seq <= seq) {
                return Err(CanonicalContractError::new(
                    "event sequence is not strictly increasing",
                ));
            }
            if event
                .entity_keys
                .iter()
                .any(|entity| entity.run_id != self.stamp.run_id)
            {
                return Err(CanonicalContractError::new(
                    "event entity run_id differs from frame run_id",
                ));
            }
            previous = Some(event.event_seq);
        }
        Ok(())
    }

    /// Build the legacy state as an owned, read-only view. Phase 18.1 will move
    /// existing sinks to this projection and remove all direct agent write-sites.
    pub fn compatibility_world_state(&self, now: Instant) -> WorldStateProjection {
        let mut people = Vec::new();
        let mut objects = Vec::new();
        let mut animals = Vec::new();

        for entity in &self.entities {
            let Some(bbox) = entity.bbox.value else {
                continue;
            };
            match &entity.kind {
                SceneEntityKind::Person(facts) => {
                    people.push(PersonState {
                        track_id: entity.display_id,
                        bbox,
                        face_bbox: None,
                        identity: facts.gallery_identity.value.as_ref().map(|identity| {
                            (
                                identity.name.clone(),
                                facts.gallery_identity.confidence.unwrap_or(0.0),
                            )
                        }),
                        emotion: fact_label(&facts.emotion),
                        posture: fact_label(&facts.posture),
                        position_3d: entity.position_3d.value,
                        emotion_distribution: None,
                        wrists: Vec::new(),
                        hands: Vec::new(),
                        body_coverage: None,
                        torso_attitude: None,
                        motion_state: None,
                        velocity: entity.kinematics.value.map(|kinematics| {
                            let [x, y] = kinematics.bbox_velocity;
                            x.hypot(y)
                        }),
                        left_gesture: None,
                        right_gesture: None,
                        recon_error: None,
                        plausibility: None,
                        torso_flexion: None,
                        head_yaw: None,
                        head_facing: None,
                        last_seen: now,
                    });
                }
                SceneEntityKind::Object(facts) => {
                    objects.push(ObjectState {
                        id: entity.display_id,
                        class: facts
                            .class_name
                            .value
                            .clone()
                            .unwrap_or_else(|| "unknown".into()),
                        bbox,
                        confidence: facts.class_name.confidence.unwrap_or(0.0),
                        age: 0,
                        position_3d: entity.position_3d.value,
                        mask: None,
                        remembered: facts.remembered,
                        last_seen: now,
                    });
                }
                SceneEntityKind::Animal(_) => animals.push(bbox),
            }
        }

        let activities = self
            .activities
            .iter()
            .filter(|activity| activity.fact.status.is_asserted())
            .map(|activity| Activity {
                label: activity.label.clone(),
                confidence: activity.fact.confidence.unwrap_or(0.0),
                participants: activity
                    .participants
                    .iter()
                    .map(|key| key.track_id)
                    .collect(),
            })
            .collect();

        WorldStateProjection(WorldState {
            frame_id: self.stamp.frame_id,
            timestamp: now,
            people,
            objects,
            animals,
            audio_events: Vec::new(),
            activities,
            events: Vec::new(),
            activity_instances: Vec::new(),
            anomaly_score: self.anomaly.value.unwrap_or(0.0),
            current_session_id: None,
            last_people_seen: None,
        })
    }
}

fn fact_label(fact: &SceneFact<String>) -> Option<(String, f32)> {
    fact.value
        .as_ref()
        .map(|value| (value.clone(), fact.confidence.unwrap_or(0.0)))
}

/// Compatibility wrapper intentionally exposes no mutable dereference.
#[derive(Debug, Clone)]
pub struct WorldStateProjection(WorldState);

impl Deref for WorldStateProjection {
    type Target = WorldState;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalContractError {
    message: String,
}

impl CanonicalContractError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CanonicalContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for CanonicalContractError {}
