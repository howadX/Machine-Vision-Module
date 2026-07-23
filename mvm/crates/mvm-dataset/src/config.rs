//! `DatasetConfig` — конфиг датасет-слоя (§16.1.1), секция `[dataset]` в `config.toml`.
//!
//! Живёт в `mvm-dataset` (не в ядре): богатый конфиг записи датасета — дело
//! dataset-слоя. В `mvm-core::Config` уходит лишь скаляр `session_timeout_secs`,
//! который нужен session-grouper'у; `mvm-cli` копирует его туда при загрузке.
//!
//! Все поля имеют `#[serde(default)]` → частичная секция `[dataset]` валидна,
//! отсутствующие ключи берут значение по умолчанию. CLI-флаги переопределяют
//! поля поверх файла — см. [`DatasetConfig::apply_cli_overrides`].

use serde::{Deserialize, Serialize};

/// Формат сохранения RGB-кадра.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FrameFormat {
    /// JPEG (по умолчанию) — компактно, с потерями.
    #[default]
    Jpg,
    /// PNG — lossless, тяжелее.
    Png,
}

impl FrameFormat {
    /// Расширение файла (`jpg`/`png`).
    pub fn ext(self) -> &'static str {
        match self {
            FrameFormat::Jpg => "jpg",
            FrameFormat::Png => "png",
        }
    }
}

/// Режим приватности сохранения лиц/identity (§16.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Privacy {
    /// `safe` (по умолчанию): лица галерейных сохраняются, лица анонимов размыты,
    /// identity/эмбеддинги — только для галерейных.
    #[default]
    Safe,
    /// `full`: всё сохраняется как есть (для владельца).
    Full,
    /// `anonymous`: все лица маскированы, identity/эмбеддинги не сохраняются.
    Anonymous,
}

/// Целевой ML-формат экспорта по умолчанию.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    /// PyTorch `IterableDataset` (по умолчанию).
    #[default]
    Pytorch,
    /// TFRecord + `tf.data`.
    Tfrecord,
    /// WebDataset (shards).
    Webdataset,
}

/// Подсекция `[dataset.export]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportConfig {
    /// Формат экспорта по умолчанию для `mvm dataset export`.
    pub default_format: ExportFormat,
}

impl Default for ExportConfig {
    fn default() -> Self {
        ExportConfig { default_format: ExportFormat::Pytorch }
    }
}

/// Секция `[dataset]` конфига (§16.1.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DatasetConfig {
    /// Включить запись (CLI: `--record`). По умолчанию выключено.
    pub enabled: bool,
    /// Корень хранилища датасетов.
    pub dir: String,
    /// Сохранять каждый N-й кадр (1 fps при 30 fps камере).
    pub frame_cadence: u32,
    /// Формат RGB-кадра.
    pub frame_format: FrameFormat,
    /// Качество JPEG (0–100), игнорируется для PNG.
    pub frame_quality: u8,
    /// Сохранять RGB-кадры. `false` → **text-only режим**: сессия несёт только
    /// `labels.jsonl` (Canonical Observation) + `session.toml`, без пикселей —
    /// подробный семантический лог сцены с той же session-дисциплиной (grouper,
    /// `check`, `prune`), что и полный датасет.
    pub save_frames: bool,
    /// Сохранять карты глубины (16-bit PNG).
    pub save_depth: bool,
    /// Сохранять instance-маски (если активен `--masks`).
    pub save_masks: bool,
    /// Сохранять SFace 128d эмбеддинги.
    pub save_raw_embeddings: bool,
    /// Сохранять BlazePose 33×3 + Hands 21×2 лендмарки.
    pub save_landmarks: bool,
    /// Режим приватности.
    pub privacy: Privacy,
    /// Разрыв >N секунд без людей → новая сессия. Дублируется в `mvm-core::Config`
    /// (grouper читает оттуда); `mvm-cli` копирует это значение при загрузке.
    pub session_timeout_secs: u64,
    /// Настройки экспорта.
    pub export: ExportConfig,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        DatasetConfig {
            enabled: false,
            dir: "~/.local/share/mvm/datasets".to_string(),
            frame_cadence: 30,
            frame_format: FrameFormat::Jpg,
            frame_quality: 85,
            save_frames: true,
            save_depth: true,
            save_masks: true,
            save_raw_embeddings: true,
            save_landmarks: true,
            privacy: Privacy::Safe,
            session_timeout_secs: 30,
            export: ExportConfig::default(),
        }
    }
}

impl DatasetConfig {
    /// Применить CLI-переопределения поверх файла (§16.1.1: «Флаги CLI переопределяют»).
    /// `None`-аргументы оставляют значение из файла нетронутым.
    pub fn apply_cli_overrides(
        &mut self,
        enabled: Option<bool>,
        dir: Option<String>,
        privacy: Option<Privacy>,
    ) {
        if let Some(e) = enabled {
            self.enabled = e;
        }
        if let Some(d) = dir {
            self.dir = d;
        }
        if let Some(p) = privacy {
            self.privacy = p;
        }
    }

    /// JPEG-качество, зажатое в валидный диапазон [1, 100].
    pub fn frame_quality_clamped(&self) -> u8 {
        self.frame_quality.clamp(1, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_plan_defaults() {
        let c = DatasetConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.frame_cadence, 30);
        assert_eq!(c.frame_format, FrameFormat::Jpg);
        assert_eq!(c.frame_quality, 85);
        assert!(c.save_frames, "frames saved by default (backward-compatible)");
        assert!(c.save_depth && c.save_masks && c.save_raw_embeddings && c.save_landmarks);
        assert_eq!(c.privacy, Privacy::Safe);
        assert_eq!(c.session_timeout_secs, 30);
        assert_eq!(c.export.default_format, ExportFormat::Pytorch);
    }

    #[test]
    fn parses_full_dataset_section_from_toml() {
        let toml_src = r#"
enabled = true
dir = "/data/mvm"
frame_cadence = 15
frame_format = "png"
frame_quality = 92
save_depth = false
save_masks = false
save_raw_embeddings = false
save_landmarks = true
privacy = "full"
session_timeout_secs = 45

[export]
default_format = "webdataset"
"#;
        let c: DatasetConfig = toml::from_str(toml_src).expect("parse");
        assert!(c.enabled);
        assert_eq!(c.dir, "/data/mvm");
        assert_eq!(c.frame_cadence, 15);
        assert_eq!(c.frame_format, FrameFormat::Png);
        assert_eq!(c.frame_quality, 92);
        assert!(!c.save_depth && !c.save_masks && !c.save_raw_embeddings);
        assert!(c.save_landmarks);
        assert_eq!(c.privacy, Privacy::Full);
        assert_eq!(c.session_timeout_secs, 45);
        assert_eq!(c.export.default_format, ExportFormat::Webdataset);
    }

    #[test]
    fn partial_toml_fills_missing_from_defaults() {
        // Only two keys set — everything else must fall back to Default.
        let c: DatasetConfig =
            toml::from_str("enabled = true\nprivacy = \"anonymous\"\n").expect("parse");
        assert!(c.enabled);
        assert_eq!(c.privacy, Privacy::Anonymous);
        // Untouched keys keep plan defaults.
        assert_eq!(c.frame_cadence, 30);
        assert_eq!(c.frame_quality, 85);
        assert_eq!(c.export.default_format, ExportFormat::Pytorch);
    }

    #[test]
    fn cli_overrides_win_over_file() {
        let mut c: DatasetConfig = toml::from_str("enabled = false\ndir = \"/from/file\"\n").unwrap();
        c.apply_cli_overrides(Some(true), Some("/from/cli".to_string()), Some(Privacy::Anonymous));
        assert!(c.enabled);
        assert_eq!(c.dir, "/from/cli");
        assert_eq!(c.privacy, Privacy::Anonymous);
        // None leaves the file value intact.
        c.apply_cli_overrides(None, None, None);
        assert!(c.enabled);
        assert_eq!(c.dir, "/from/cli");
    }

    #[test]
    fn frame_quality_is_clamped() {
        let mut c = DatasetConfig::default();
        c.frame_quality = 200;
        assert_eq!(c.frame_quality_clamped(), 100);
        c.frame_quality = 0;
        assert_eq!(c.frame_quality_clamped(), 1);
    }

    #[test]
    fn frame_format_extensions() {
        assert_eq!(FrameFormat::Jpg.ext(), "jpg");
        assert_eq!(FrameFormat::Png.ext(), "png");
    }
}
