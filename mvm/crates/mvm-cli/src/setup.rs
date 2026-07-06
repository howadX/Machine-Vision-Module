//! `mvm setup` (phase 14.1) — fetch the ONNX models into the models dir.
//!
//! Manifest-driven (`manifest::MODELS`): for each model, skip if it's already
//! present and its SHA-256 matches, otherwise download it (HTTP range-resume
//! for interrupted transfers) and verify the checksum before committing. A
//! `.part` file holds the in-flight bytes; only a checksum-verified file is
//! renamed into place, so a killed `setup` never leaves a corrupt model that
//! `run` would load.
//!
//! Product layer only (ureq + sha2 live in `mvm-cli`); the vision core is
//! untouched.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::config_file::ProductConfig;
use crate::manifest::{ModelSpec, MODELS};

pub struct SetupArgs {
    /// Re-download even models already present with a matching checksum.
    force: bool,
}

pub fn parse_setup_args(raw: &[String]) -> SetupArgs {
    let mut a = SetupArgs { force: false };
    for tok in &raw[2.min(raw.len())..] {
        if tok == "--force" {
            a.force = true;
        }
    }
    a
}

/// One model's outcome, for the summary line.
enum Outcome {
    Present,
    Fetched,
    SkippedNoUrl,
    Failed(String),
}

/// `mvm setup` — fetch every model that has a public URL. Owner-hosted weights
/// (`url: None`) are reported, not fetched. Fails the process only if a
/// *required* model couldn't be obtained.
pub fn run_setup(product: &ProductConfig, args: SetupArgs) -> Result<()> {
    let dir = &product.models_dir;
    crate::paths::ensure_dir_private(dir)?;
    eprintln!("mvm setup — models → {}", dir.display());

    let mut required_failures = 0;
    for spec in MODELS {
        let outcome = ensure_model(spec, dir, args.force);
        report(spec, &outcome);
        if spec.required && !matches!(outcome, Outcome::Present | Outcome::Fetched) {
            required_failures += 1;
        }
    }

    if required_failures > 0 {
        bail!("{required_failures} required model(s) could not be set up (see above)");
    }
    eprintln!("mvm setup — done");
    Ok(())
}

/// Fetch just the named models (used by the first-run auto-trigger — required
/// only). Returns Ok if every requested model ends up present+valid.
// Wired into `run`'s first-launch auto-trigger in 14.1c (next commit).
#[allow(dead_code)]
pub fn fetch_specific(dir: &Path, files: &[&str]) -> Result<()> {
    crate::paths::ensure_dir_private(dir)?;
    let mut failed = Vec::new();
    for spec in MODELS.iter().filter(|m| files.contains(&m.file)) {
        let outcome = ensure_model(spec, dir, false);
        report(spec, &outcome);
        if !matches!(outcome, Outcome::Present | Outcome::Fetched) {
            failed.push(spec.file);
        }
    }
    if !failed.is_empty() {
        bail!("could not fetch: {}", failed.join(", "));
    }
    Ok(())
}

/// Make `spec` present-and-valid in `dir`, downloading if needed.
fn ensure_model(spec: &ModelSpec, dir: &Path, force: bool) -> Outcome {
    let target = dir.join(spec.file);
    if !force && target.exists() {
        match sha256_file(&target) {
            Ok(h) if h == spec.sha256 => return Outcome::Present,
            // Present but wrong/corrupt → re-fetch (fall through).
            _ => {}
        }
    }
    let url = match spec.url {
        Some(u) => u,
        None => return Outcome::SkippedNoUrl,
    };
    match download_verify(spec, url, dir) {
        Ok(()) => Outcome::Fetched,
        Err(e) => Outcome::Failed(format!("{e:#}")),
    }
}

/// Download `url` to `<dir>/<file>.part` (resuming an existing part via HTTP
/// Range), verify SHA-256, then atomically rename to the final path. One
/// automatic retry-from-scratch if a resumed transfer fails to verify.
fn download_verify(spec: &ModelSpec, url: &str, dir: &Path) -> Result<()> {
    let part = dir.join(format!("{}.part", spec.file));
    eprintln!("  ↓ {} ({:.1} MB)", spec.file, spec.size as f64 / 1e6);

    match fetch_once(spec, url, &part) {
        Ok(()) => {}
        Err(_) => {
            // A resumed transfer can fail if the partial bytes are stale; wipe
            // and try once more from zero before giving up.
            let _ = std::fs::remove_file(&part);
            fetch_once(spec, url, &part)?;
        }
    }

    let got = sha256_file(&part).context("hash downloaded file")?;
    if got != spec.sha256 {
        let _ = std::fs::remove_file(&part);
        bail!("sha256 mismatch (got {got}, want {})", spec.sha256);
    }
    std::fs::rename(&part, dir.join(spec.file)).context("commit downloaded model")?;
    Ok(())
}

/// One transfer attempt into `part`, resuming from its current length if the
/// server honours a Range request (206); otherwise starting fresh.
fn fetch_once(spec: &ModelSpec, url: &str, part: &Path) -> Result<()> {
    let mut offset = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    // A leftover part at/over the expected size is unusable for a range resume.
    if offset >= spec.size {
        let _ = std::fs::remove_file(part);
        offset = 0;
    }

    let mut resp = if offset > 0 {
        ureq::get(url).header("Range", &format!("bytes={offset}-")).call()
    } else {
        ureq::get(url).call()
    }
    .with_context(|| format!("GET {url}"))?;

    let status = resp.status().as_u16();
    let (mut file, mut written) = if status == 206 && offset > 0 {
        (
            std::fs::OpenOptions::new().append(true).open(part).context("reopen .part")?,
            offset,
        )
    } else if (200..300).contains(&status) {
        // Server ignored the range (or none requested) → full body from zero.
        (std::fs::File::create(part).context("create .part")?, 0u64)
    } else {
        bail!("HTTP {status} for {url}");
    };

    let mut reader = resp.body_mut().as_reader();
    let mut buf = [0u8; 65536];
    let mut last_pct = u64::MAX;
    loop {
        let n = reader.read(&mut buf).context("read response body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write .part")?;
        written += n as u64;
        let pct = (written * 100 / spec.size.max(1)).min(100);
        if pct != last_pct && pct % 10 == 0 && pct < 100 {
            eprint!("\r    {pct}%   ");
            let _ = std::io::stderr().flush();
            last_pct = pct;
        }
    }
    file.flush().context("flush .part")?;
    eprintln!("\r    100%  ");
    Ok(())
}

fn report(spec: &ModelSpec, outcome: &Outcome) {
    match outcome {
        Outcome::Present => eprintln!("  ✓ {} (present)", spec.file),
        Outcome::Fetched => eprintln!("  ✓ {} ({})", spec.file, spec.note),
        Outcome::SkippedNoUrl => eprintln!(
            "  ⚠ {} — no public URL; provide manually or await a project release ({})",
            spec.file, spec.note
        ),
        Outcome::Failed(e) => eprintln!("  ✗ {} — {e}", spec.file),
    }
}

/// Streaming SHA-256 of a file as lowercase hex.
fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Path helper for tests / callers that only need one model's location.
#[allow(dead_code)]
pub fn model_path(dir: &Path, file: &str) -> PathBuf {
    dir.join(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        // SHA-256("abc") — the canonical NIST test vector.
        let dir = std::env::temp_dir().join(format!("mvm_setup_sha_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("abc.txt");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hex_lower_is_zero_padded_lowercase() {
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
    }

    #[test]
    fn present_valid_model_is_skipped_not_refetched() {
        // A file whose bytes hash to the recorded sha would be "present"; here
        // we use a fake spec whose sha matches a tiny file, and assert no URL
        // fetch is attempted (url None + Present outcome, no failure).
        let dir = std::env::temp_dir().join(format!("mvm_setup_present_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("abc.txt"), b"abc").unwrap();
        let spec = ModelSpec {
            file: "abc.txt",
            url: None, // would be SkippedNoUrl if it tried to fetch
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            size: 3,
            required: false,
            note: "test",
        };
        assert!(matches!(ensure_model(&spec, &dir, false), Outcome::Present));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn owner_hosted_without_url_is_skipped_when_absent() {
        let dir = std::env::temp_dir().join(format!("mvm_setup_nourl_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let spec = ModelSpec {
            file: "yolo_missing.onnx",
            url: None,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            size: 10,
            required: false,
            note: "project-hosted",
        };
        assert!(matches!(ensure_model(&spec, &dir, false), Outcome::SkippedNoUrl));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
