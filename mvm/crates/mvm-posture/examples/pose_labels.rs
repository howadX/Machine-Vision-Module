//! 7.5.5 oracle: native NativePose labels on the committed p_V references (to diff vs Python golden).
use mvm_core::Config;
use mvm_posture::NativePose;

fn main() -> anyhow::Result<()> {
    let config = Config {
        models_path: std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")),
        ..Default::default()
    };
    let mut np = NativePose::init(&config)?;
    let dir = config.models_path.join("references");
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("p_V") && n.ends_with(".jpg")).unwrap_or(false))
        .collect();
    files.sort();
    for p in &files {
        let img = image::open(p)?;
        let poses = np.poses(&img)?;
        let name = p.file_name().unwrap().to_str().unwrap();
        match poses.first() {
            Some(r) => println!("{name}: {} ({:.3})", r.label, r.confidence),
            None => println!("{name}: none"),
        }
    }
    Ok(())
}
