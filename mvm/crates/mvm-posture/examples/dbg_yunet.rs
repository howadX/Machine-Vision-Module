//! Diagnostic: YuNet face detection rate on saved frames at several thresholds.
use mvm_core::{Config, Module};
use mvm_detect::YunetDetector;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or("/tmp/oracle_frames".into());
    let config = Config {
        models_path: std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")),
        ..Default::default()
    };
    let mut det = YunetDetector::init(&config)?;
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .map(|x| x == "png" || x == "jpg")
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    for thr in [0.5f32, 0.3, 0.15, 0.05] {
        det.set_score_threshold(thr);
        let (mut hit, mut total_faces) = (0u32, 0u32);
        for p in &files {
            let img = image::open(p)?;
            let d = det.detect(&img)?;
            if !d.is_empty() {
                hit += 1;
                total_faces += d.len() as u32;
            }
        }
        println!(
            "thr={thr:.2}: frames_with_face={hit}/{}  total_faces={total_faces}",
            files.len()
        );
    }
    Ok(())
}
