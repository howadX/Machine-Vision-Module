//! Product config layer (phase 14.0.2) — `~/.config/mvm/config.toml`.
//!
//! Priority: **CLI flags > env (`MVM_*`) > file > defaults**. A missing file is
//! not an error (everything has a default); a *malformed* file is — silent
//! fallback would hide typos. This is the CLI-layer config, not to be confused
//! with the runtime `mvm_core::Config` — this layer only *feeds* that one
//! (`camera_path`, `models_path`). CLI-flag precedence is applied at the call
//! sites (the flags simply overwrite the resolved values).

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Raw deserialized `config.toml`. Every field optional — absent means
/// "use the next layer down".
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    #[serde(default)]
    pub camera: CameraSection,
    #[serde(default)]
    pub output: OutputSection,
    #[serde(default)]
    pub models: ModelsSection,
    #[serde(default)]
    pub gallery: GallerySection,
    #[serde(default)]
    pub cadence: CadenceSection,
    /// Dataset Factory `[dataset]` section (phase 16.1.1). The struct lives in
    /// `mvm-dataset`; here it is just deserialized and its `dir` tilde-expanded.
    #[serde(default)]
    pub dataset: mvm_dataset::DatasetConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraSection {
    /// "auto" (scan /dev/video*) or an explicit device path.
    pub device: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Explicit mount state used by Phase 19.1. `false` never means moving; it
    /// means that zero optical flow lacks authority to assert a fixed camera.
    pub fixed: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSection {
    /// state + scene/events logs + journal, all in one place.
    pub data_dir: Option<String>,
    /// Optional override for current_state.md alone (external consumers).
    pub state_path: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsSection {
    pub dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GallerySection {
    pub dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CadenceSection {
    pub snapshot_ms: Option<u64>,
}

/// Fully resolved product config — what the rest of the CLI consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductConfig {
    /// "auto" or a device path like "/dev/video2".
    pub camera_device: String,
    pub camera_width: u32,
    pub camera_height: u32,
    pub camera_fixed: bool,
    pub data_dir: PathBuf,
    /// Defaults to `<data_dir>/current_state.md`.
    pub state_path: PathBuf,
    pub models_dir: PathBuf,
    pub gallery_dir: PathBuf,
    pub snapshot_ms: u64,
    /// Dataset Factory config (phase 16.1). `dataset.dir` is already tilde-expanded.
    pub dataset: mvm_dataset::DatasetConfig,
}

type EnvFn<'a> = &'a dyn Fn(&str) -> Option<String>;

fn std_env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Load `config.toml` from the XDG config dir (if present), apply env
/// overrides, and resolve everything to concrete values. The one entry point
/// for `run`/`snapshot`/`serve`.
pub fn load() -> anyhow::Result<ProductConfig> {
    let path = crate::paths::config_dir().join("config.toml");
    let file = read_file_config(&path)?;
    Ok(resolve(file, &std_env, &|p| p.exists()))
}

/// Parse `config.toml` at `path`. Missing file → `None` (defaults); malformed
/// file → error with the path in context.
pub fn read_file_config(path: &Path) -> anyhow::Result<Option<FileConfig>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow::anyhow!("read {}: {e}", path.display())),
    };
    let cfg: FileConfig =
        toml::from_str(&text).map_err(|e| anyhow::anyhow!("parse {}: {e}", path.display()))?;
    Ok(Some(cfg))
}

/// Merge env (`MVM_*`) over file over defaults. Pure — env and filesystem
/// probing are injected so tests don't touch process globals.
pub(crate) fn resolve(
    file: Option<FileConfig>,
    env: EnvFn,
    exists: &dyn Fn(&Path) -> bool,
) -> ProductConfig {
    let home = env("HOME").unwrap_or_default();
    let file = file.unwrap_or_default();

    let pick = |env_key: &str, file_val: Option<&String>| -> Option<String> {
        env(env_key)
            .filter(|v| !v.is_empty())
            .or_else(|| file_val.cloned())
    };

    let camera_device =
        pick("MVM_CAMERA_DEVICE", file.camera.device.as_ref()).unwrap_or_else(|| "auto".into());
    let camera_width = pick("MVM_CAMERA_WIDTH", None)
        .and_then(|v| v.parse().ok())
        .or(file.camera.width)
        .unwrap_or(1280);
    let camera_height = pick("MVM_CAMERA_HEIGHT", None)
        .and_then(|v| v.parse().ok())
        .or(file.camera.height)
        .unwrap_or(720);
    let camera_fixed = pick("MVM_CAMERA_FIXED", None)
        .and_then(|value| value.parse().ok())
        .or(file.camera.fixed)
        .unwrap_or(false);

    let data_dir = pick("MVM_DATA_DIR", file.output.data_dir.as_ref())
        .map(|v| expand_tilde(&v, &home))
        .unwrap_or_else(|| crate::paths::data_dir_from(env));

    let state_path = pick("MVM_STATE_PATH", file.output.state_path.as_ref())
        .map(|v| expand_tilde(&v, &home))
        .unwrap_or_else(|| data_dir.join("current_state.md"));

    let models_explicit =
        pick("MVM_MODELS_DIR", file.models.dir.as_ref()).map(|v| expand_tilde(&v, &home));
    let models_dir = resolve_models_dir(
        models_explicit,
        crate::paths::data_dir_from(env).join("models"),
        PathBuf::from("./models"),
        exists,
    );

    let gallery_dir = pick("MVM_GALLERY_DIR", file.gallery.dir.as_ref())
        .map(|v| expand_tilde(&v, &home))
        .unwrap_or_else(|| crate::paths::config_dir_from(env).join("gallery"));

    let snapshot_ms = pick("MVM_SNAPSHOT_MS", None)
        .and_then(|v| v.parse().ok())
        .or(file.cadence.snapshot_ms)
        .unwrap_or(1000);

    // Dataset Factory (phase 16.1): the section comes wholesale from the file
    // (serde defaults fill the rest); only its storage dir needs tilde expansion.
    let mut dataset = file.dataset;
    dataset.dir = expand_tilde(&dataset.dir, &home)
        .to_string_lossy()
        .into_owned();

    ProductConfig {
        camera_device,
        camera_width,
        camera_height,
        camera_fixed,
        data_dir,
        state_path,
        models_dir,
        gallery_dir,
        snapshot_ms,
        dataset,
    }
}

/// Models dir resolution: explicit (env/file) always wins; otherwise the XDG
/// location if it exists (`mvm setup`, 14.1); otherwise `./models` if it exists
/// (dev checkout, cwd = mvm/); otherwise XDG anyway — so the "models missing →
/// run `mvm setup`" error points at the canonical place, not at a random cwd.
fn resolve_models_dir(
    explicit: Option<PathBuf>,
    xdg: PathBuf,
    cwd_fallback: PathBuf,
    exists: &dyn Fn(&Path) -> bool,
) -> PathBuf {
    if let Some(p) = explicit {
        return p;
    }
    if exists(&xdg) {
        return xdg;
    }
    if exists(&cwd_fallback) {
        return cwd_fallback;
    }
    xdg
}

/// Expand a leading `~/` (or bare `~`) with `home`. Anything else unchanged.
fn expand_tilde(path: &str, home: &str) -> PathBuf {
    if path == "~" {
        return PathBuf::from(home);
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k: &str| {
            pairs
                .iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v.to_string())
        }
    }

    const NOTHING_EXISTS: &dyn Fn(&Path) -> bool = &|_| false;

    #[test]
    fn full_toml_parses() {
        let cfg: FileConfig = toml::from_str(
            r#"
            [camera]
            device = "/dev/video2"
            width = 640
            height = 480
            fixed = true

            [output]
            data_dir = "~/mydata"
            state_path = "~/state.md"

            [models]
            dir = "~/models"

            [gallery]
            dir = "~/faces"

            [cadence]
            snapshot_ms = 500
            "#,
        )
        .unwrap();
        assert_eq!(cfg.camera.device.as_deref(), Some("/dev/video2"));
        assert_eq!(cfg.camera.width, Some(640));
        assert_eq!(cfg.camera.fixed, Some(true));
        assert_eq!(cfg.output.data_dir.as_deref(), Some("~/mydata"));
        assert_eq!(cfg.models.dir.as_deref(), Some("~/models"));
        assert_eq!(cfg.gallery.dir.as_deref(), Some("~/faces"));
        assert_eq!(cfg.cadence.snapshot_ms, Some(500));
    }

    #[test]
    fn empty_toml_is_all_defaults() {
        let cfg: FileConfig = toml::from_str("").unwrap();
        assert!(cfg.camera.device.is_none());
        assert!(cfg.output.data_dir.is_none());
    }

    #[test]
    fn unknown_key_is_an_error() {
        // Typos must not be silently ignored (deny_unknown_fields).
        assert!(toml::from_str::<FileConfig>("[camera]\ndevices = \"x\"\n").is_err());
    }

    #[test]
    fn defaults_without_file_or_env() {
        let env = env_of(&[("HOME", "/home/u")]);
        let c = resolve(None, &env, NOTHING_EXISTS);
        assert_eq!(c.camera_device, "auto");
        assert_eq!((c.camera_width, c.camera_height), (1280, 720));
        assert!(!c.camera_fixed);
        assert_eq!(c.data_dir, PathBuf::from("/home/u/.local/share/mvm"));
        assert_eq!(
            c.state_path,
            PathBuf::from("/home/u/.local/share/mvm/current_state.md")
        );
        assert_eq!(
            c.models_dir,
            PathBuf::from("/home/u/.local/share/mvm/models")
        );
        assert_eq!(c.gallery_dir, PathBuf::from("/home/u/.config/mvm/gallery"));
        assert_eq!(c.snapshot_ms, 1000);
    }

    #[test]
    fn file_overrides_defaults_with_tilde_expansion() {
        let file: FileConfig = toml::from_str(
            "[camera]\ndevice = \"/dev/video2\"\n[output]\ndata_dir = \"~/vision\"\n[cadence]\nsnapshot_ms = 250\n",
        )
        .unwrap();
        let env = env_of(&[("HOME", "/home/u")]);
        let c = resolve(Some(file), &env, NOTHING_EXISTS);
        assert_eq!(c.camera_device, "/dev/video2");
        assert_eq!(c.data_dir, PathBuf::from("/home/u/vision"));
        // state follows the overridden data_dir unless state_path is set.
        assert_eq!(
            c.state_path,
            PathBuf::from("/home/u/vision/current_state.md")
        );
        assert_eq!(c.snapshot_ms, 250);
    }

    #[test]
    fn env_overrides_file() {
        let file: FileConfig = toml::from_str(
            "[camera]\ndevice = \"/dev/video2\"\n[models]\ndir = \"/file/models\"\n",
        )
        .unwrap();
        let env = env_of(&[
            ("HOME", "/home/u"),
            ("MVM_CAMERA_DEVICE", "/dev/video9"),
            ("MVM_MODELS_DIR", "/env/models"),
            ("MVM_CAMERA_FIXED", "true"),
        ]);
        let c = resolve(Some(file), &env, NOTHING_EXISTS);
        assert_eq!(c.camera_device, "/dev/video9");
        assert_eq!(c.models_dir, PathBuf::from("/env/models"));
        assert!(c.camera_fixed);
    }

    #[test]
    fn explicit_state_path_beats_data_dir_join() {
        let file: FileConfig =
            toml::from_str("[output]\nstate_path = \"~/elsewhere/state.md\"\n").unwrap();
        let env = env_of(&[("HOME", "/home/u")]);
        let c = resolve(Some(file), &env, NOTHING_EXISTS);
        assert_eq!(c.state_path, PathBuf::from("/home/u/elsewhere/state.md"));
        assert_eq!(c.data_dir, PathBuf::from("/home/u/.local/share/mvm"));
    }

    #[test]
    fn models_dir_prefers_xdg_then_cwd_then_xdg_for_errors() {
        let xdg = PathBuf::from("/x/mvm/models");
        let cwd = PathBuf::from("./models");
        // XDG populated (post `mvm setup`) → XDG.
        let picked = resolve_models_dir(None, xdg.clone(), cwd.clone(), &|p| p == xdg);
        assert_eq!(picked, xdg);
        // XDG absent, dev checkout has ./models → ./models.
        let picked = resolve_models_dir(None, xdg.clone(), cwd.clone(), &|p| p == cwd);
        assert_eq!(picked, cwd);
        // Neither exists → XDG (so the error points at the canonical dir).
        let picked = resolve_models_dir(None, xdg.clone(), cwd.clone(), NOTHING_EXISTS);
        assert_eq!(picked, xdg);
        // Explicit always wins, even if it doesn't exist yet.
        let explicit = PathBuf::from("/custom");
        let picked = resolve_models_dir(Some(explicit.clone()), xdg.clone(), cwd, &|p| p == xdg);
        assert_eq!(picked, explicit);
    }

    #[test]
    fn missing_file_is_none_malformed_is_error() {
        let dir = std::env::temp_dir().join(format!("mvm_cfg_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(
            read_file_config(&dir.join("config.toml"))
                .unwrap()
                .is_none()
        );
        let bad = dir.join("config.toml");
        std::fs::write(&bad, "not [valid toml").unwrap();
        assert!(read_file_config(&bad).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tilde_expansion_variants() {
        assert_eq!(
            expand_tilde("~/a/b", "/home/u"),
            PathBuf::from("/home/u/a/b")
        );
        assert_eq!(expand_tilde("~", "/home/u"), PathBuf::from("/home/u"));
        assert_eq!(
            expand_tilde("/abs/path", "/home/u"),
            PathBuf::from("/abs/path")
        );
        assert_eq!(
            expand_tilde("rel/path", "/home/u"),
            PathBuf::from("rel/path")
        );
    }
}
