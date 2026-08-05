//! Human/LLM read-only rendering of the latest Cortex artifact (phase 18.9).

use anyhow::{Context, Result, bail};
use mvm_cortex::{AttentionPolicy, CortexFrame, PolicyOutcomeFrame};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

use crate::agents::{AttentionPolicyBus, CortexFrameBus, PolicyOutcomeBus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CortexView {
    Full,
    Attention,
    Forecast,
    SelfModel,
    Uncertainty,
}

impl CortexView {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "full" => Some(Self::Full),
            "attention" => Some(Self::Attention),
            "forecast" => Some(Self::Forecast),
            "self" => Some(Self::SelfModel),
            "uncertainty" => Some(Self::Uncertainty),
            _ => None,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Attention => "attention",
            Self::Forecast => "forecast",
            Self::SelfModel => "self",
            Self::Uncertainty => "uncertainty",
        }
    }
}

pub(crate) struct CortexArgs {
    pub(crate) input: Option<PathBuf>,
    pub(crate) view: CortexView,
    pub(crate) top_k: usize,
    pub(crate) horizon_seconds: f64,
    pub(crate) raw_json: bool,
}

pub(crate) fn parse_args(raw: &[String]) -> Result<CortexArgs> {
    let mut args = CortexArgs {
        input: None,
        view: CortexView::Full,
        top_k: 5,
        horizon_seconds: 2.0,
        raw_json: false,
    };
    let mut i = 2;
    while i < raw.len() {
        match raw[i].as_str() {
            "--input" if i + 1 < raw.len() => {
                args.input = Some(PathBuf::from(&raw[i + 1]));
                i += 1;
            }
            "--view" if i + 1 < raw.len() => {
                args.view = CortexView::parse(&raw[i + 1])
                    .ok_or_else(|| anyhow::anyhow!("unknown Cortex view: {}", raw[i + 1]))?;
                i += 1;
            }
            "--top-k" if i + 1 < raw.len() => {
                args.top_k = raw[i + 1].parse().context("invalid --top-k")?;
                i += 1;
            }
            "--horizon-seconds" if i + 1 < raw.len() => {
                args.horizon_seconds = raw[i + 1].parse().context("invalid --horizon-seconds")?;
                if !args.horizon_seconds.is_finite() || args.horizon_seconds < 0.0 {
                    bail!("--horizon-seconds must be finite and non-negative");
                }
                i += 1;
            }
            "--json" => args.raw_json = true,
            _ => {}
        }
        i += 1;
    }
    args.top_k = args.top_k.clamp(1, 20);
    Ok(args)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CortexArtifact {
    pub(crate) schema: String,
    pub(crate) cortex: CortexFrame,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) attention_policy: Option<AttentionPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) policy_outcomes: Option<PolicyOutcomeFrame>,
}

pub(crate) struct CortexReportWriter {
    frames: CortexFrameBus,
    policies: AttentionPolicyBus,
    outcomes: PolicyOutcomeBus,
    path: PathBuf,
    last_scene_frame: Option<u64>,
    last_outcome_frame: Option<u64>,
}

impl CortexReportWriter {
    pub(crate) fn new(
        frames: CortexFrameBus,
        policies: AttentionPolicyBus,
        outcomes: PolicyOutcomeBus,
        path: PathBuf,
    ) -> Self {
        Self {
            frames,
            policies,
            outcomes,
            path,
            last_scene_frame: None,
            last_outcome_frame: None,
        }
    }

    fn write(
        &self,
        frame: CortexFrame,
        policy: Option<AttentionPolicy>,
        outcomes: Option<PolicyOutcomeFrame>,
    ) -> Result<()> {
        let artifact = CortexArtifact {
            schema: "18.9.0".into(),
            cortex: frame,
            attention_policy: policy,
            policy_outcomes: outcomes,
        };
        let bytes = serde_json::to_vec_pretty(&artifact)?;
        let tmp = self
            .path
            .with_extension(format!("json.tmp.{}", std::process::id()));
        std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("publish {}", self.path.display()))?;
        Ok(())
    }
}

impl Agent for CortexReportWriter {
    fn name(&self) -> String {
        "cortex-report".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0;
        let mut errors = 0;
        while ctx.running() {
            if let Some(frame) = self.frames.latest() {
                let outcome_frame = self.outcomes.latest();
                let outcome_id = outcome_frame
                    .as_ref()
                    .map(|value| value.source_scene_frame_id);
                if self.last_scene_frame != Some(frame.scene_frame_id)
                    || self.last_outcome_frame != outcome_id
                {
                    self.last_scene_frame = Some(frame.scene_frame_id);
                    self.last_outcome_frame = outcome_id;
                    if let Err(error) = self.write(frame, self.policies.latest(), outcome_frame) {
                        eprintln!("  cortex-report: {error}");
                        errors += 1;
                    } else {
                        ticks += 1;
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        AgentHealth {
            name: self.name(),
            ticks,
            errors,
        }
    }
}

pub(crate) fn render_file(
    path: &Path,
    view: CortexView,
    top_k: usize,
    horizon_seconds: f64,
    raw_json: bool,
) -> Result<String> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read Cortex artifact {}", path.display()))?;
    let value: Value = serde_json::from_str(&text).context("parse Cortex artifact")?;
    if raw_json {
        return Ok(serde_json::to_string_pretty(&value)?);
    }
    render_value(&value, view, top_k, horizon_seconds)
}

fn render_value(
    value: &Value,
    view: CortexView,
    top_k: usize,
    horizon_seconds: f64,
) -> Result<String> {
    let cortex = value
        .get("cortex")
        .ok_or_else(|| anyhow::anyhow!("Cortex artifact has no cortex frame"))?;
    let frame_id = cortex
        .get("scene_frame_id")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let timestamp = cortex
        .get("timestamp_unix_ms")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let mut out = format!(
        "# Cortex {} · scene frame {} · view={}\n",
        timestamp,
        frame_id,
        view.as_str()
    );
    match view {
        CortexView::Full => {
            render_attention(&mut out, cortex, top_k);
            render_forecast(&mut out, cortex, horizon_seconds);
            render_self(&mut out, cortex);
            render_uncertainty(&mut out, cortex);
            render_generic_section(&mut out, "Affordances", cortex.get("affordances"));
            render_generic_section(&mut out, "Memory", cortex.get("memory"));
            render_generic_section(&mut out, "Policy outcomes", value.get("policy_outcomes"));
        }
        CortexView::Attention => {
            render_attention(&mut out, cortex, top_k);
            render_uncertainty(&mut out, cortex);
        }
        CortexView::Forecast => render_forecast(&mut out, cortex, horizon_seconds),
        CortexView::SelfModel => render_self(&mut out, cortex),
        CortexView::Uncertainty => render_uncertainty(&mut out, cortex),
    }
    Ok(out)
}

fn render_attention(out: &mut String, cortex: &Value, top_k: usize) {
    out.push_str("\n## Attention\n");
    let focus = cortex.pointer("/attention/top_k").and_then(Value::as_array);
    match focus {
        Some(scores) if !scores.is_empty() => {
            let mut ranked = scores.iter().collect::<Vec<_>>();
            ranked.sort_by(|left, right| {
                right
                    .get("score")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
                    .partial_cmp(&left.get("score").and_then(Value::as_f64).unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for score in ranked.into_iter().take(top_k.max(1)) {
                let target = score
                    .get("target")
                    .map(compact_value)
                    .unwrap_or_else(|| "unknown".into());
                let value = score.get("score").and_then(Value::as_f64).unwrap_or(0.0);
                out.push_str(&format!(
                    "- {target}: {value:.3}{}\n",
                    evidence_suffix(score)
                ));
            }
        }
        _ => out.push_str("- none (no evidence-backed focus)\n"),
    }
}

fn render_forecast(out: &mut String, cortex: &Value, horizon_seconds: f64) {
    out.push_str("\n## Forecast\n");
    match cortex.get("forecast") {
        Some(forecast) if !forecast.is_null() => {
            let actual_ms = forecast
                .get("horizon_ms")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let actual = actual_ms as f64 / 1000.0;
            out.push_str(&format!(
                "- bounded horizon: {actual:.3}s (requested {horizon_seconds:.3}s)\n"
            ));
            let entities = forecast
                .get("entities")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            out.push_str(&format!("- entity forecasts: {entities}\n"));
        }
        _ => out.push_str("- withheld: no forecast evidence\n"),
    }
    render_uncertainty(out, cortex);
}

fn render_self(out: &mut String, cortex: &Value) {
    out.push_str("\n## Visual self-model\n");
    if let Some(model) = cortex.get("self_model") {
        out.push_str(&format!(
            "- spatial: {}\n",
            compact_value(model.get("spatial").unwrap_or(&Value::Null))
        ));
        out.push_str(&format!(
            "- camera-relative pose: {}\n",
            claim_value(model.get("camera_relative_pose"))
        ));
        out.push_str(&format!(
            "- world pose: {}\n",
            claim_value(model.get("world_pose"))
        ));
        out.push_str(&format!(
            "- calibration: {}\n",
            compact_value(model.get("calibration").unwrap_or(&Value::Null))
        ));
    } else {
        out.push_str("- unknown: self-model unavailable\n");
    }
}

fn render_uncertainty(out: &mut String, cortex: &Value) {
    out.push_str("\n## Uncertainty\n");
    match cortex
        .pointer("/uncertainty/claims")
        .and_then(Value::as_array)
    {
        Some(claims) if !claims.is_empty() => {
            for claim in claims {
                let status = claim
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let reasons = claim
                    .pointer("/reasons")
                    .map(compact_value)
                    .unwrap_or_else(|| "[]".into());
                out.push_str(&format!("- {status}: reasons={reasons}\n"));
            }
        }
        _ => out.push_str("- none recorded\n"),
    }
}

fn render_generic_section(out: &mut String, title: &str, value: Option<&Value>) {
    out.push_str(&format!("\n## {title}\n"));
    out.push_str(&format!(
        "{}\n",
        value.map(compact_value).unwrap_or_else(|| "unknown".into())
    ));
}

fn claim_value(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return "unknown".into();
    };
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    format!(
        "{status}{}",
        value
            .get("reasons")
            .map(|v| format!(" reasons={}", compact_value(v)))
            .unwrap_or_default()
    )
}

fn evidence_suffix(value: &Value) -> String {
    value
        .get("evidence")
        .map(|e| format!(" evidence={}", compact_value(e)))
        .unwrap_or_default()
}

fn compact_value(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_attention_applies_top_k_without_dropping_honesty_fields() {
        let artifact = serde_json::json!({
            "schema": "18.9.0",
            "cortex": {
                "timestamp_unix_ms": 1,
                "scene_frame_id": 7,
                "attention": {"top_k": [
                    {"target": "scene", "score": 0.2, "evidence": []},
                    {"target": {"entity": {"run_id":"r","track_domain":"person","track_id":1,"incarnation":0}}, "score": 0.9,
                     "breakdown": {"uncertainty": 0.8}, "evidence": [{"scene_frame_id":7}]}
                ]},
                "uncertainty": {"claims": [{"status":"withheld","reasons":["stale_source"]}]}
            }
        });
        let output = render_value(&artifact, CortexView::Attention, 1, 2.0).unwrap();
        assert!(output.contains("0.900"));
        assert!(!output.contains("0.200"));
        assert!(
            output.contains("stale_source"),
            "honesty reason must remain visible"
        );
    }

    #[test]
    fn forecast_view_states_bounded_horizon_and_unknown_claims() {
        let artifact = serde_json::json!({
            "schema": "18.9.0",
            "cortex": {
                "timestamp_unix_ms": 1,
                "scene_frame_id": 7,
                "forecast": {"horizon_seconds": 1.0, "entities": []},
                "uncertainty": {"claims": [{"status":"unknown","reasons":["uncalibrated"]}]}
            }
        });
        let output = render_value(&artifact, CortexView::Forecast, 5, 2.0).unwrap();
        assert!(output.contains("bounded"));
        assert!(output.contains("unknown"));
        assert!(output.contains("uncalibrated"));
    }
}
