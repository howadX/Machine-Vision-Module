//! XDG path seam — single source of truth for product paths (phase 14.0.1).
//!
//! Hand-rolled XDG Base Directory resolution (+0 deps). All path decisions for
//! the product shell live here; the vision core never touches these. This is
//! also the cross-platform seam: macOS (`~/Library/…`) / Windows (`%APPDATA%`)
//! branches land in this module later without changing call sites.
//!
//! Per the XDG spec, a relative `$XDG_*` value is invalid and must be ignored
//! (fall back to the default).

// TODO(14.0.2/14.0.4): remove once the config/state layers consume this module.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Environment lookup used by the resolvers. Tests pass a closure instead of
/// mutating process-global env (parallel tests would race otherwise).
type EnvFn<'a> = &'a dyn Fn(&str) -> Option<String>;

fn std_env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// `~/.local/share/mvm` (or `$XDG_DATA_HOME/mvm`) — state + scene/events logs
/// + journal, all in one place.
pub fn data_dir() -> PathBuf {
    data_dir_from(&std_env)
}

/// `~/.config/mvm` (or `$XDG_CONFIG_HOME/mvm`) — config.toml, later gallery.
pub fn config_dir() -> PathBuf {
    config_dir_from(&std_env)
}

/// `<data_dir>/models` — fetched ONNX models (`mvm setup`, 14.1). Deliberately
/// under data, not `$XDG_CACHE_HOME`: the OS may purge cache, and without
/// models the product is dead.
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

pub(crate) fn data_dir_from(env: EnvFn) -> PathBuf {
    xdg_base(env, "XDG_DATA_HOME", &[".local", "share"])
}

pub(crate) fn config_dir_from(env: EnvFn) -> PathBuf {
    xdg_base(env, "XDG_CONFIG_HOME", &[".config"])
}

/// Resolve `$key/mvm`, falling back to `$HOME/<fallback…>/mvm`. Empty or
/// relative `$key` values are ignored per the XDG spec.
fn xdg_base(env: EnvFn, key: &str, fallback: &[&str]) -> PathBuf {
    if let Some(v) = env(key) {
        let p = PathBuf::from(&v);
        if !v.is_empty() && p.is_absolute() {
            return p.join("mvm");
        }
    }
    let home = env("HOME").unwrap_or_default();
    let mut p = PathBuf::from(home);
    for seg in fallback {
        p.push(seg);
    }
    p.join("mvm")
}

/// Create `dir` (and parents) with private permissions (0700 on unix): the
/// journal records identities and daily routine — private data.
pub fn ensure_dir_private(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k: &str| pairs.iter().find(|(key, _)| *key == k).map(|(_, v)| v.to_string())
    }

    #[test]
    fn data_dir_honors_xdg_data_home() {
        let env = env_of(&[("XDG_DATA_HOME", "/custom/data"), ("HOME", "/home/u")]);
        assert_eq!(data_dir_from(&env), PathBuf::from("/custom/data/mvm"));
    }

    #[test]
    fn data_dir_falls_back_to_home_local_share() {
        let env = env_of(&[("HOME", "/home/u")]);
        assert_eq!(data_dir_from(&env), PathBuf::from("/home/u/.local/share/mvm"));
    }

    #[test]
    fn empty_xdg_value_is_ignored() {
        let env = env_of(&[("XDG_DATA_HOME", ""), ("HOME", "/home/u")]);
        assert_eq!(data_dir_from(&env), PathBuf::from("/home/u/.local/share/mvm"));
    }

    #[test]
    fn relative_xdg_value_is_ignored_per_spec() {
        let env = env_of(&[("XDG_DATA_HOME", "rel/path"), ("HOME", "/home/u")]);
        assert_eq!(data_dir_from(&env), PathBuf::from("/home/u/.local/share/mvm"));
    }

    #[test]
    fn config_dir_honors_xdg_and_falls_back() {
        let env = env_of(&[("XDG_CONFIG_HOME", "/custom/cfg"), ("HOME", "/home/u")]);
        assert_eq!(config_dir_from(&env), PathBuf::from("/custom/cfg/mvm"));
        let env = env_of(&[("HOME", "/home/u")]);
        assert_eq!(config_dir_from(&env), PathBuf::from("/home/u/.config/mvm"));
    }

    #[test]
    fn models_dir_is_under_data_dir() {
        // models_dir() delegates to data_dir(); verify composition via the
        // internal resolver to stay env-independent.
        let env = env_of(&[("XDG_DATA_HOME", "/d")]);
        assert_eq!(data_dir_from(&env).join("models"), PathBuf::from("/d/mvm/models"));
    }

    #[test]
    #[cfg(unix)]
    fn ensure_dir_private_creates_0700() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir()
            .join(format!("mvm_paths_test_{}", std::process::id()))
            .join("nested");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        ensure_dir_private(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700, "data dir must be private (0700)");
        std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
    }
}
