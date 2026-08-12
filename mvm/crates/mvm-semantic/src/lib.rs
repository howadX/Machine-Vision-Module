//! Measurement-only SceneParse150 semantic expert.
//! The output is deliberately not a canonical surface, occupancy, or support fact.

use anyhow::{Context, bail};
use image::DynamicImage;
use ort::{session::Session, value::Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticMeasurementLabel {
    Floor,
    Wall,
    Ceiling,
    OpeningCandidate,
    OtherNonSurface,
    Withheld,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticMeasurement {
    pub label: SemanticMeasurementLabel,
    pub confidence: f32,
    pub entropy: f32,
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticGrid {
    pub width: usize,
    pub height: usize,
    pub labels: Vec<SemanticMeasurementLabel>,
    pub confidence: Vec<f32>,
    pub entropy: Vec<f32>,
    pub valid: Vec<bool>,
}
impl SemanticGrid {
    pub fn new(
        width: usize,
        height: usize,
        labels: Vec<SemanticMeasurementLabel>,
        confidence: Vec<f32>,
        entropy: Vec<f32>,
        valid: Vec<bool>,
    ) -> anyhow::Result<Self> {
        let n = width
            .checked_mul(height)
            .context("semantic grid dimensions overflow")?;
        if labels.len() != n || confidence.len() != n || entropy.len() != n || valid.len() != n {
            bail!("semantic grid vectors must have width*height entries")
        }
        if confidence
            .iter()
            .chain(entropy.iter())
            .any(|v| !v.is_finite())
        {
            bail!("semantic grid confidence and entropy must be finite")
        }
        Ok(Self {
            width,
            height,
            labels,
            confidence,
            entropy,
            valid,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SceneParse150Reducer {
    pub confidence_threshold: f32,
    pub entropy_threshold: f32,
    pub opening_threshold: f32,
}
impl Default for SceneParse150Reducer {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.55,
            entropy_threshold: 2.5,
            opening_threshold: 0.45,
        }
    }
}
impl SceneParse150Reducer {
    pub const WALL: usize = 0;
    pub const FLOOR: usize = 3;
    pub const CEILING: usize = 5;
    pub const OPENING: &'static [usize] = &[8, 14, 58];
    fn mass(p: &[f32], ids: &[usize]) -> f32 {
        ids.iter().map(|&i| p.get(i).copied().unwrap_or(0.0)).sum()
    }
    fn withheld() -> SemanticMeasurement {
        SemanticMeasurement {
            label: SemanticMeasurementLabel::Withheld,
            confidence: 0.0,
            entropy: 0.0,
            valid: false,
        }
    }
    pub fn reduce_probabilities(&self, p: &[f32]) -> anyhow::Result<SemanticMeasurement> {
        if p.len() != 150 {
            bail!("expected 150 probabilities")
        }
        if p.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Ok(Self::withheld());
        }
        let sum: f32 = p.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            bail!("probability mass is invalid")
        }
        let q: Vec<f32> = p.iter().map(|v| v / sum).collect();
        let confidence = [Self::FLOOR, Self::WALL, Self::CEILING]
            .iter()
            .map(|&i| q[i])
            .chain(std::iter::once(Self::mass(&q, Self::OPENING)))
            .chain(std::iter::once(
                1.0 - q[Self::FLOOR]
                    - q[Self::WALL]
                    - q[Self::CEILING]
                    - Self::mass(&q, Self::OPENING),
            ))
            .fold(0.0, f32::max);
        let entropy = -q
            .iter()
            .filter(|v| **v > 0.0)
            .map(|v| v * v.ln())
            .sum::<f32>();
        let opening_mass = Self::mass(&q, Self::OPENING);
        let other_mass = 1.0 - q[Self::FLOOR] - q[Self::WALL] - q[Self::CEILING] - opening_mass;
        let (candidate_conf, label) = [
            (q[Self::FLOOR], SemanticMeasurementLabel::Floor),
            (q[Self::WALL], SemanticMeasurementLabel::Wall),
            (q[Self::CEILING], SemanticMeasurementLabel::Ceiling),
            (opening_mass, SemanticMeasurementLabel::OpeningCandidate),
            (other_mass, SemanticMeasurementLabel::OtherNonSurface),
        ]
        .into_iter()
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
        let label = if entropy > self.entropy_threshold || confidence < self.confidence_threshold {
            SemanticMeasurementLabel::Withheld
        } else if candidate_conf < self.opening_threshold
            && matches!(label, SemanticMeasurementLabel::OpeningCandidate)
        {
            SemanticMeasurementLabel::Withheld
        } else if candidate_conf >= self.confidence_threshold {
            label
        } else {
            SemanticMeasurementLabel::OtherNonSurface
        };
        Ok(SemanticMeasurement {
            label,
            confidence: candidate_conf,
            entropy,
            valid: label != SemanticMeasurementLabel::Withheld,
        })
    }
    pub fn reduce_logits(&self, logits: &[f32]) -> anyhow::Result<SemanticMeasurement> {
        if logits.len() != 150 {
            bail!("expected 150 logits")
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Ok(Self::withheld());
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (*v - max).exp()).collect();
        self.reduce_probabilities(&exps)
    }
}

fn preprocess_rgb_256(image: &DynamicImage) -> anyhow::Result<ndarray::Array4<f32>> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let s = 256usize;
    let img = image
        .resize_exact(256, 256, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let mut flat = vec![0.0; 3 * s * s];
    for y in 0..s {
        for x in 0..s {
            let p = img.get_pixel(x as u32, y as u32);
            let i = y * s + x;
            for c in 0..3 {
                flat[c * s * s + i] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    ndarray::Array4::from_shape_vec((1, 3, s, s), flat).context("invalid semantic input shape")
}

pub struct SemanticEstimator {
    session: Session,
    reducer: SceneParse150Reducer,
}
impl SemanticEstimator {
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("failed to load semantic model {}", path.display()))?;
        Ok(Self {
            session,
            reducer: Default::default(),
        })
    }
    pub fn estimate(&mut self, image: &DynamicImage) -> anyhow::Result<SemanticGrid> {
        let input = Tensor::from_array(preprocess_rgb_256(image)?)
            .context("failed to create semantic tensor")?;
        let outputs = self.session.run(ort::inputs!["pixel_values" => input])?;
        let (shape, raw) = outputs[0].try_extract_tensor::<f32>()?;
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let (h, w) = match dims.as_slice() {
            [1, 150, h, w] => (*h, *w),
            _ => bail!("semantic output must have shape [1,150,H,W], got {dims:?}"),
        };
        if raw.iter().any(|v| !v.is_finite()) {
            bail!("semantic output contains non-finite values")
        }
        let mut labels = Vec::with_capacity(h * w);
        let mut confidence = Vec::with_capacity(h * w);
        let mut entropy = Vec::with_capacity(h * w);
        let mut valid = Vec::with_capacity(h * w);
        for cell in 0..h * w {
            let m = self.reducer.reduce_logits(
                (0..150)
                    .map(|c| raw[c * h * w + cell])
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?;
            labels.push(m.label);
            confidence.push(m.confidence);
            entropy.push(m.entropy);
            valid.push(m.valid);
        }
        SemanticGrid::new(w, h, labels, confidence, entropy, valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn confident_unmapped_class_is_other_non_surface() {
        let r = SceneParse150Reducer::default();
        let mut p = vec![0.0; 150];
        p[10] = 0.9;
        let m = r.reduce_probabilities(&p).unwrap();
        assert_eq!(m.label, SemanticMeasurementLabel::OtherNonSurface);
        assert!(m.valid);
    }
    #[test]
    fn probability_mass_groups_map_to_measurement_labels() {
        let reducer = SceneParse150Reducer::default();
        let cases = [
            (
                &[SceneParse150Reducer::FLOOR][..],
                SemanticMeasurementLabel::Floor,
            ),
            (
                &[SceneParse150Reducer::WALL][..],
                SemanticMeasurementLabel::Wall,
            ),
            (
                &[SceneParse150Reducer::CEILING][..],
                SemanticMeasurementLabel::Ceiling,
            ),
            (
                SceneParse150Reducer::OPENING,
                SemanticMeasurementLabel::OpeningCandidate,
            ),
            (&[10][..], SemanticMeasurementLabel::OtherNonSurface),
        ];

        for (classes, expected) in cases {
            let mut probabilities = vec![0.0; 150];
            for class in classes {
                probabilities[*class] = 1.0 / classes.len() as f32;
            }
            let measurement = reducer.reduce_probabilities(&probabilities).unwrap();
            assert_eq!(measurement.label, expected);
            assert!(measurement.valid);
        }
    }

    #[test]
    fn low_confidence_is_withheld() {
        let r = SceneParse150Reducer::default();
        let p = vec![1.0 / 150.0; 150];
        assert_eq!(
            r.reduce_probabilities(&p).unwrap().label,
            SemanticMeasurementLabel::Withheld
        );
        assert!(!r.reduce_probabilities(&p).unwrap().valid);
    }
    #[test]
    fn non_finite_logits_are_withheld_with_finite_metadata() {
        let r = SceneParse150Reducer::default();
        let mut logits = vec![0.0; 150];
        logits[3] = f32::NAN;
        let measurement = r.reduce_logits(&logits).unwrap();
        assert_eq!(measurement.label, SemanticMeasurementLabel::Withheld);
        assert!(!measurement.valid);
        assert!(measurement.confidence.is_finite());
        assert!(measurement.entropy.is_finite());
    }
    #[test]
    fn opening_uses_grouped_probability_mass() {
        let r = SceneParse150Reducer::default();
        let mut p = vec![0.0; 150];
        p[8] = 0.25;
        p[14] = 0.25;
        p[58] = 0.1;
        assert_eq!(
            r.reduce_probabilities(&p).unwrap().label,
            SemanticMeasurementLabel::OpeningCandidate
        );
    }
    #[test]
    fn grid_rejects_wrong_lengths_and_non_finite() {
        let l = vec![SemanticMeasurementLabel::Withheld];
        assert!(SemanticGrid::new(2, 1, l.clone(), vec![0.1], vec![0.1], vec![true]).is_err());
        assert!(SemanticGrid::new(1, 1, l, vec![f32::NAN], vec![0.1], vec![true]).is_err());
    }

    #[test]
    fn preprocessing_is_static_imagenet_rgb_nchw() {
        let image =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0])));
        let input = preprocess_rgb_256(&image).unwrap();
        assert_eq!(input.shape(), &[1, 3, 256, 256]);
        assert!((input[[0, 0, 0, 0]] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((input[[0, 1, 0, 0]] - (0.0 - 0.456) / 0.224).abs() < 1e-5);
    }
}
