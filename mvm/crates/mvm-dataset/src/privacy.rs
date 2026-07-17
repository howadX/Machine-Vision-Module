//! Privacy-слой: какие области кадра размыть перед сохранением (§16.6).
//!
//! Здесь — только **чистое вычисление регионов** (нормализованные прямоугольники);
//! само размытие пикселей применяет вызывающая сторона (mesh-агент, крейт `image`).
//! Так `mvm-dataset` остаётся без image-зависимости и тестируем без картинок.
//!
//! ⚠️ Первый приближённый срез (16.1.2b): лицо аппроксимируется верхней частью
//! bbox тела, т.к. WorldState пока не несёт YuNet-bbox лица. 16.5 заменит это на
//! точный face-bbox. Приближение **консервативно** (размывает больше, не меньше) —
//! privacy-first не нарушается.

use crate::config::Privacy;
use mvm_core::BBox;

/// Доля высоты bbox тела сверху, принимаемая за «лицо» (консервативная оценка).
const FACE_BAND_FRAC: f32 = 0.32;

/// Регионы для размытия (нормализованные [0,1] прямоугольники) при данном режиме.
///
/// `people` — пары `(body_bbox, is_anonymous)`. По режимам (§16.6.1):
/// - `full` — ничего не размывается;
/// - `safe` — размываются лица **анонимов** (галерейные сохраняются чёткими);
/// - `anonymous` — размываются лица **всех** людей.
pub fn blur_regions(people: &[(BBox, bool)], mode: Privacy) -> Vec<BBox> {
    people
        .iter()
        .filter(|(_, is_anon)| match mode {
            Privacy::Full => false,
            Privacy::Safe => *is_anon,
            Privacy::Anonymous => true,
        })
        .map(|(bbox, _)| face_band(bbox))
        .collect()
}

/// Верхняя полоса bbox тела как приближение области лица.
fn face_band(b: &BBox) -> BBox {
    let h = (b.y2 - b.y1).max(0.0);
    BBox { x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y1 + h * FACE_BAND_FRAC }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox() -> BBox {
        BBox { x1: 0.3, y1: 0.1, x2: 0.6, y2: 0.9 }
    }

    #[test]
    fn full_mode_blurs_nothing() {
        let people = vec![(bbox(), true), (bbox(), false)];
        assert!(blur_regions(&people, Privacy::Full).is_empty());
    }

    #[test]
    fn safe_mode_blurs_only_anonymous() {
        let people = vec![(bbox(), true /*anon*/), (bbox(), false /*gallery*/)];
        let regions = blur_regions(&people, Privacy::Safe);
        assert_eq!(regions.len(), 1, "only the anonymous face is blurred");
    }

    #[test]
    fn anonymous_mode_blurs_everyone() {
        let people = vec![(bbox(), true), (bbox(), false)];
        assert_eq!(blur_regions(&people, Privacy::Anonymous).len(), 2);
    }

    #[test]
    fn face_band_is_top_portion_of_body() {
        let r = face_band(&bbox());
        assert!((r.x1 - 0.3).abs() < 1e-6 && (r.x2 - 0.6).abs() < 1e-6, "full body width");
        assert!((r.y1 - 0.1).abs() < 1e-6, "starts at body top");
        // Top 32% of a 0.8-tall body → y2 = 0.1 + 0.8*0.32 = 0.356.
        assert!((r.y2 - 0.356).abs() < 1e-3, "top band only, got {}", r.y2);
        assert!(r.y2 < bbox().y2, "never covers the whole body");
    }
}
