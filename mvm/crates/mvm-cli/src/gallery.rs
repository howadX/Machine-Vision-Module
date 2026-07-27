//! Watched identity gallery (phase 14.2) — `<gallery_dir>/<Name>/*.{jpg,png}`.
//!
//! Each subfolder is a person and its name IS the display name (Unicode, so
//! "Вадим" works verbatim); every photo inside is a reference shot. Drop a
//! photo in a folder → that person is known. This replaces the old
//! `models/references/p_V_*.jpg` + hardcoded `[("V", "Вадим")]` name-map: no
//! map, no baked-in paths, no filename convention.
//!
//! Photos are stored as-is (whole frames); the face is detected and cropped at
//! load time, so `mvm gallery add` just copies the file in.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Image extensions accepted as gallery photos (lowercase compare).
const PHOTO_EXTS: &[&str] = &["jpg", "jpeg", "png"];

fn is_photo(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| PHOTO_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// `(name, sorted photos)` for every non-empty person folder under `dir`,
/// sorted by name so reserved-id assignment is deterministic. Missing dir → [].
pub fn people(dir: &Path) -> Vec<(String, Vec<PathBuf>)> {
    let mut out: Vec<(String, Vec<PathBuf>)> = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let photos = photos_in(&path);
        if !photos.is_empty() {
            out.push((name, photos));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Sorted photo paths directly inside `person_dir`.
fn photos_in(person_dir: &Path) -> Vec<PathBuf> {
    let mut photos: Vec<PathBuf> = std::fs::read_dir(person_dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && is_photo(p))
                .collect()
        })
        .unwrap_or_default();
    photos.sort();
    photos
}

/// Populate `identity` from the gallery: for each person folder, detect + crop
/// each photo's face and add it as a reference. An empty/missing gallery is not
/// an error — everyone simply stays anonymous. Returns photos ingested.
pub fn load_into(
    identity: &mut mvm_identity::IdentityModule,
    detector: &mut mvm_detect::YunetDetector,
    dir: &Path,
) -> Result<usize> {
    // Reference photos can be small / off-angle — loosen the face threshold
    // while building (the caller restores the live threshold afterwards).
    detector.set_score_threshold(0.3);
    let mut total = 0usize;
    for (name, photos) in people(dir) {
        for photo in &photos {
            match ingest_photo(identity, detector, &name, photo) {
                Ok(true) => total += 1,
                Ok(false) => eprintln!("  gallery: no face in {}, skipping", photo.display()),
                Err(e) => eprintln!("  gallery: {} — {e:#}", photo.display()),
            }
        }
    }
    eprintln!(
        "  gallery: {} people ({total} reference photos)",
        identity.gallery_size()
    );
    Ok(total)
}

/// Detect the largest face in `photo`, crop it, and add it as a reference for
/// `name`. Returns false if no face was found (photo skipped, not an error).
fn ingest_photo(
    identity: &mut mvm_identity::IdentityModule,
    detector: &mut mvm_detect::YunetDetector,
    name: &str,
    photo: &Path,
) -> Result<bool> {
    let img = image::open(photo).with_context(|| format!("open {}", photo.display()))?;
    let dets = match detector.detect(&img) {
        Ok(d) => d,
        Err(_) => return Ok(false),
    };
    let Some(best) = dets.first() else {
        return Ok(false);
    };
    let (w, h) = (img.width(), img.height());
    let x1 = (best.bbox.x1 * w as f32) as u32;
    let y1 = (best.bbox.y1 * h as f32) as u32;
    let x2 = (best.bbox.x2 * w as f32) as u32;
    let y2 = (best.bbox.y2 * h as f32) as u32;
    if x2 <= x1 || y2 <= y1 {
        return Ok(false);
    }
    let face = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
    identity.add_reference(name, &face)?;
    Ok(true)
}

/// Copy `photos` into `<dir>/<name>/`, numbering them after any already there.
/// Returns the number copied. Non-image files are rejected.
pub fn add(dir: &Path, name: &str, photos: &[PathBuf]) -> Result<usize> {
    if name.trim().is_empty() {
        anyhow::bail!("gallery: person name must not be empty");
    }
    if photos.is_empty() {
        anyhow::bail!("gallery: no photos given (usage: mvm gallery add <name> <photo>…)");
    }
    let person_dir = dir.join(name);
    std::fs::create_dir_all(&person_dir)
        .with_context(|| format!("create {}", person_dir.display()))?;
    let mut next = photos_in(&person_dir).len();
    let mut added = 0usize;
    for src in photos {
        if !is_photo(src) {
            anyhow::bail!("gallery: {} is not a .jpg/.jpeg/.png", src.display());
        }
        if !src.is_file() {
            anyhow::bail!("gallery: {} not found", src.display());
        }
        let ext = src
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg")
            .to_ascii_lowercase();
        next += 1;
        let dest = person_dir.join(format!("{next:02}.{ext}"));
        std::fs::copy(src, &dest)
            .with_context(|| format!("copy {} → {}", src.display(), dest.display()))?;
        added += 1;
    }
    Ok(added)
}

/// `(name, photo count)` per person, sorted by name.
pub fn list(dir: &Path) -> Vec<(String, usize)> {
    people(dir).into_iter().map(|(n, p)| (n, p.len())).collect()
}

// ── `mvm gallery <add|list>` subcommand ─────────────────────────────────────

pub fn run_gallery(dir: &Path, raw: &[String]) -> Result<()> {
    match raw.get(2).map(String::as_str) {
        Some("add") => {
            let name = raw
                .get(3)
                .context("usage: mvm gallery add <name> <photo>…")?;
            let photos: Vec<PathBuf> = raw[4.min(raw.len())..].iter().map(PathBuf::from).collect();
            let n = add(dir, name, &photos)?;
            eprintln!(
                "gallery: added {n} photo(s) for '{name}' → {}",
                dir.join(name).display()
            );
            Ok(())
        }
        Some("list") => {
            let people = list(dir);
            if people.is_empty() {
                println!("gallery: empty ({})", dir.display());
            } else {
                println!("gallery: {} ({} people)", dir.display(), people.len());
                for (name, count) in people {
                    println!("  {name} — {count} photo(s)");
                }
            }
            Ok(())
        }
        other => {
            anyhow::bail!(
                "unknown gallery command {:?} — use `mvm gallery add <name> <photo>…` or `mvm gallery list`",
                other.unwrap_or("")
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let uniq = N.fetch_add(1, Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("mvm_gallery_{}_{uniq}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn people_uses_folder_name_and_sorts() {
        let dir = tmp();
        std::fs::create_dir_all(dir.join("Вадим")).unwrap();
        std::fs::create_dir_all(dir.join("Alice")).unwrap();
        std::fs::write(dir.join("Вадим/02.jpg"), b"x").unwrap();
        std::fs::write(dir.join("Вадим/01.jpg"), b"x").unwrap();
        std::fs::write(dir.join("Alice/a.png"), b"x").unwrap();
        let people = people(&dir);
        assert_eq!(people.len(), 2);
        assert_eq!(people[0].0, "Alice", "sorted by name");
        assert_eq!(people[1].0, "Вадим");
        // Photos within a person are sorted.
        assert!(people[1].1[0].ends_with("01.jpg"));
        assert!(people[1].1[1].ends_with("02.jpg"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_folder_is_ignored_non_images_skipped() {
        let dir = tmp();
        std::fs::create_dir_all(dir.join("Empty")).unwrap();
        std::fs::create_dir_all(dir.join("Bob")).unwrap();
        std::fs::write(dir.join("Bob/notes.txt"), b"x").unwrap(); // not a photo
        std::fs::write(dir.join("Bob/face.jpg"), b"x").unwrap();
        let people = people(&dir);
        assert_eq!(people.len(), 1, "Empty folder has no photos → dropped");
        assert_eq!(people[0].0, "Bob");
        assert_eq!(people[0].1.len(), 1, "notes.txt is not counted");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn add_copies_and_numbers_after_existing() {
        let dir = tmp();
        let src = dir.join("src.jpg");
        std::fs::write(&src, b"fake").unwrap();
        assert_eq!(add(&dir, "Carol", &[src.clone()]).unwrap(), 1);
        assert_eq!(add(&dir, "Carol", &[src.clone()]).unwrap(), 1);
        let photos = photos_in(&dir.join("Carol"));
        assert_eq!(photos.len(), 2);
        assert!(photos[0].ends_with("01.jpg"));
        assert!(photos[1].ends_with("02.jpg"), "numbered after existing");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn add_rejects_non_image_and_missing() {
        let dir = tmp();
        let txt = dir.join("x.txt");
        std::fs::write(&txt, b"x").unwrap();
        assert!(add(&dir, "D", &[txt]).is_err(), "non-image rejected");
        assert!(
            add(&dir, "D", &[dir.join("nope.jpg")]).is_err(),
            "missing rejected"
        );
        assert!(
            add(&dir, "", &[]).is_err(),
            "empty name + no photos rejected"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn list_reports_counts() {
        let dir = tmp();
        let src = dir.join("s.png");
        std::fs::write(&src, b"x").unwrap();
        add(&dir, "Eve", &[src.clone(), src.clone()]).unwrap();
        let listed = list(&dir);
        assert_eq!(listed, vec![("Eve".to_string(), 2)]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_gallery_dir_is_empty_not_error() {
        assert!(people(Path::new("/definitely/no/such/gallery")).is_empty());
        assert!(list(Path::new("/definitely/no/such/gallery")).is_empty());
    }
}
