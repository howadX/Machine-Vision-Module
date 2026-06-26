//! Diagnostic: native pose on p_V_03 → print normalized 33×3 + class, to diff vs golden.
use mvm_core::Config;
use mvm_posture::NativePose;

fn main() -> anyhow::Result<()> {
    let config = Config {
        models_path: std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")),
        ..Default::default()
    };
    let mut np = NativePose::init(&config)?;
    let img = image::open(config.models_path.join("references/p_V_03.jpg"))?;
    let (w, h) = (img.width() as f32, img.height() as f32);
    let people = np.landmarks_per_person(&img)?;
    let poses = np.poses(&img)?;
    let lm = &people[0];
    for k in 0..33 {
        println!("{k} {:.4} {:.4} {:.4}", lm[k][0] / w, lm[k][1] / h, lm[k][2] / w);
    }
    eprintln!("CLASS {} @ {:.3} recon {:.4} plaus {:.2}",
        poses[0].label, poses[0].confidence, poses[0].recon_error, poses[0].plausibility);
    Ok(())
}
