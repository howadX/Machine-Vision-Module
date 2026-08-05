//! Privacy-слой: какие области кадра размыть перед сохранением (§16.5–16.6).
//!
//! Здесь — только **чистое вычисление регионов** (нормализованные прямоугольники);
//! само размытие пикселей применяет вызывающая сторона (mesh-агент, крейт `image`).
//! Так `mvm-dataset` остаётся без image-зависимости и тестируем без картинок.
//!
//! Приоритет региона (16.5):
//! 1. **Точный YuNet-face-bbox** (слегка расширенный), когда лицо сопоставлено телу
//!    в этом кадре — блюрит ровно лицо.
//! 2. **Fallback: верхняя полоса bbox тела**, когда у анонима лицо НЕ сопоставлено
//!    (профиль/недетект), но тело видно — privacy-first: не оставить лицо чётким.

use crate::config::Privacy;
use mvm_core::BBox;

/// Доля высоты bbox тела сверху — fallback-оценка лица, когда точного bbox нет.
/// Щедрая (privacy-first): при недетекте лицо может быть ниже верхушки bbox
/// (наклон вперёд, поднятая рука раздувает bbox вверх), поэтому берём верхнюю
/// половину, а не узкую полоску — лучше пере-размыть, чем оставить лицо чётким.
const FACE_BAND_FRAC: f32 = 0.5;
/// Симметричный отступ вокруг точного face-bbox (доля его размера) — гарантирует
/// покрытие ушей/подбородка/края причёски, которые YuNet обрезает.
const FACE_PAD_FRAC: f32 = 0.18;

/// Одна персона для privacy-решения: bbox тела, сопоставленное лицо (если есть в
/// этом кадре) и является ли персона анонимом (нет галерейной identity).
#[derive(Debug, Clone, Copy)]
pub struct PrivacyPerson {
    pub body: BBox,
    pub face: Option<BBox>,
    pub is_anon: bool,
}

/// Регионы для размытия (нормализованные [0,1] прямоугольники) при данном режиме.
///
/// По режимам (§16.6.1): `full` — ничего; `safe` — лица **анонимов**; `anonymous`
/// — лица **всех**. Для каждой размываемой персоны берётся точный padded face-bbox,
/// иначе (лицо не сопоставлено) — верхняя полоса тела.
pub fn blur_regions(people: &[PrivacyPerson], mode: Privacy) -> Vec<BBox> {
    people
        .iter()
        .filter(|p| match mode {
            Privacy::Full => false,
            Privacy::Safe => p.is_anon,
            Privacy::Anonymous => true,
        })
        .map(|p| match p.face {
            Some(f) => pad(&f, FACE_PAD_FRAC),
            None => face_band(&p.body),
        })
        .collect()
}

/// Расширить bbox на `frac` его ширины/высоты с каждой стороны, зажать в [0,1].
fn pad(b: &BBox, frac: f32) -> BBox {
    let dx = (b.x2 - b.x1).max(0.0) * frac;
    let dy = (b.y2 - b.y1).max(0.0) * frac;
    BBox {
        x1: (b.x1 - dx).max(0.0),
        y1: (b.y1 - dy).max(0.0),
        x2: (b.x2 + dx).min(1.0),
        y2: (b.y2 + dy).min(1.0),
    }
}

/// Верхняя полоса bbox тела как приближение области лица (fallback).
fn face_band(b: &BBox) -> BBox {
    let h = (b.y2 - b.y1).max(0.0);
    BBox {
        x1: b.x1,
        y1: b.y1,
        x2: b.x2,
        y2: b.y1 + h * FACE_BAND_FRAC,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> BBox {
        BBox {
            x1: 0.3,
            y1: 0.1,
            x2: 0.6,
            y2: 0.9,
        }
    }
    fn face() -> BBox {
        BBox {
            x1: 0.40,
            y1: 0.15,
            x2: 0.52,
            y2: 0.30,
        }
    }
    fn anon_with_face() -> PrivacyPerson {
        PrivacyPerson {
            body: body(),
            face: Some(face()),
            is_anon: true,
        }
    }
    fn gallery_with_face() -> PrivacyPerson {
        PrivacyPerson {
            body: body(),
            face: Some(face()),
            is_anon: false,
        }
    }
    fn anon_no_face() -> PrivacyPerson {
        PrivacyPerson {
            body: body(),
            face: None,
            is_anon: true,
        }
    }

    #[test]
    fn full_mode_blurs_nothing() {
        assert!(blur_regions(&[anon_with_face(), gallery_with_face()], Privacy::Full).is_empty());
    }

    #[test]
    fn safe_mode_blurs_only_anonymous() {
        let r = blur_regions(&[anon_with_face(), gallery_with_face()], Privacy::Safe);
        assert_eq!(r.len(), 1, "only the anonymous face");
    }

    #[test]
    fn anonymous_mode_blurs_everyone() {
        assert_eq!(
            blur_regions(&[anon_with_face(), gallery_with_face()], Privacy::Anonymous).len(),
            2
        );
    }

    #[test]
    fn precise_face_used_when_available_and_padded() {
        let r = blur_regions(&[anon_with_face()], Privacy::Safe);
        let region = r[0];
        // Padded around the real face, not the body-top band → tighter than the body
        // width, and it starts below the body top (face is inside the body).
        assert!(
            region.x1 > body().x1 && region.x2 < body().x2,
            "narrower than body"
        );
        assert!(
            region.y1 < face().y1 + 1e-6 && region.y2 > face().y2 - 1e-6,
            "covers the face"
        );
        // Padding stays within the frame.
        assert!(region.x1 >= 0.0 && region.y1 >= 0.0 && region.x2 <= 1.0 && region.y2 <= 1.0);
    }

    #[test]
    fn falls_back_to_body_band_when_no_face() {
        // Anon person, face not matched this frame → cover the head region (top band)
        // so a visible-but-undetected face never leaks (privacy-first).
        let r = blur_regions(&[anon_no_face()], Privacy::Safe);
        assert_eq!(r.len(), 1);
        let region = r[0];
        assert!((region.x1 - body().x1).abs() < 1e-6, "full body width");
        assert!(region.y2 < body().y2, "top band only");
    }

    #[test]
    fn pad_clamps_to_frame() {
        // A face at the very edge — padding must not exceed [0,1].
        let edge = PrivacyPerson {
            body: BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            face: Some(BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 0.2,
                y2: 0.2,
            }),
            is_anon: true,
        };
        let r = blur_regions(&[edge], Privacy::Anonymous)[0];
        assert!(r.x1 >= 0.0 && r.y1 >= 0.0 && r.x2 <= 1.0 && r.y2 <= 1.0);
    }
}
