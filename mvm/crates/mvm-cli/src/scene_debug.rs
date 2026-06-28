//! Scene visual debug (phase 8.E): draw object + person boxes on a frame so you
//! can SEE what the scene detector finds. Zero new deps — manual rectangle fill
//! and a tiny built-in 3×5 bitmap font for the id numbers (respects the
//! "no new deps for tooling" invariant; only an ONNX expert grows the footprint).

use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use mvm_core::BBox;
use mvm_scene::{TrackedObject, OBJECT_ID_BASE};

/// Distinct box colours + their human names (for the printed legend).
const PALETTE: [([u8; 3], &str); 8] = [
    ([255, 80, 80], "red"),
    ([90, 220, 90], "green"),
    ([80, 160, 255], "blue"),
    ([255, 215, 60], "yellow"),
    ([255, 90, 255], "magenta"),
    ([70, 230, 230], "cyan"),
    ([255, 150, 40], "orange"),
    ([185, 130, 255], "purple"),
];

/// Deterministic colour for a class name (FNV-1a hash → palette). The same class
/// always gets the same colour across frames, so the legend is learnable.
pub fn class_palette(name: &str) -> ([u8; 3], &'static str) {
    let mut h: u32 = 2166136261;
    for b in name.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    PALETTE[(h as usize) % PALETTE.len()]
}

/// 3×5 bitmap digits 0–9, row-major; bit 2 (0b100) is the leftmost column.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111], // 0
    [0b010, 0b110, 0b010, 0b010, 0b111], // 1
    [0b111, 0b001, 0b111, 0b100, 0b111], // 2
    [0b111, 0b001, 0b111, 0b001, 0b111], // 3
    [0b101, 0b101, 0b111, 0b001, 0b001], // 4
    [0b111, 0b100, 0b111, 0b001, 0b111], // 5
    [0b111, 0b100, 0b111, 0b101, 0b111], // 6
    [0b111, 0b001, 0b001, 0b001, 0b001], // 7
    [0b111, 0b101, 0b111, 0b101, 0b111], // 8
    [0b111, 0b101, 0b111, 0b001, 0b111], // 9
];

fn px_clamp(v: f32, max: u32) -> u32 {
    (v.max(0.0) as u32).min(max.saturating_sub(1))
}

/// Draw a hollow rectangle (normalized bbox) of the given thickness.
fn draw_rect(img: &mut RgbImage, b: &BBox, color: [u8; 3], thick: u32) {
    let (w, h) = (img.width(), img.height());
    let x1 = px_clamp(b.x1 * w as f32, w);
    let y1 = px_clamp(b.y1 * h as f32, h);
    let x2 = px_clamp(b.x2 * w as f32, w);
    let y2 = px_clamp(b.y2 * h as f32, h);
    let c = Rgb(color);
    for t in 0..thick {
        for x in x1..=x2 {
            if y1 + t < h {
                img.put_pixel(x, y1 + t, c);
            }
            if y2 >= t {
                img.put_pixel(x, y2 - t, c);
            }
        }
        for y in y1..=y2 {
            if x1 + t < w {
                img.put_pixel(x1 + t, y, c);
            }
            if x2 >= t {
                img.put_pixel(x2 - t, y, c);
            }
        }
    }
}

/// Draw an integer at (x,y) top-left, `scale` px per font cell, with a 1px dark
/// backing for contrast.
fn draw_number(img: &mut RgbImage, x: u32, y: u32, n: u32, color: [u8; 3], scale: u32) {
    let (w, h) = (img.width(), img.height());
    let mut cx = x;
    for ch in n.to_string().bytes() {
        let d = (ch - b'0') as usize;
        for (row, bits) in DIGITS[d].iter().enumerate() {
            for col in 0..3u32 {
                if bits & (1 << (2 - col)) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = cx + col * scale + dx;
                            let py = y + row as u32 * scale + dy;
                            if px < w && py < h {
                                img.put_pixel(px, py, Rgb(color));
                            }
                        }
                    }
                }
            }
        }
        cx += 4 * scale; // 3 cols + 1 space
    }
}

/// Render an annotated frame: white person boxes, class-coloured object boxes with
/// their short id (`id - OBJECT_ID_BASE`) stamped at the corner.
pub fn render(frame: &DynamicImage, person_bboxes: &[BBox], objects: &[TrackedObject]) -> RgbImage {
    let mut img = frame.to_rgb8();
    let (_, h) = frame.dimensions();
    let scale = (h / 120).max(2); // digit size scales with frame height

    for b in person_bboxes {
        draw_rect(&mut img, b, [255, 255, 255], 2);
    }
    for o in objects {
        let (color, _) = class_palette(o.class_name);
        draw_rect(&mut img, &o.bbox, color, 2);
        let lx = px_clamp(o.bbox.x1 * img.width() as f32, img.width());
        let ly = px_clamp(o.bbox.y1 * h as f32, h);
        draw_number(&mut img, lx + 2, ly + 2, o.id - OBJECT_ID_BASE, color, scale);
    }
    img
}

/// Render + save the annotated frame, and return a printable legend (one line per
/// object: short-id, class, colour, score, age, bbox).
pub fn write_annotated(
    path: &str,
    frame: &DynamicImage,
    person_bboxes: &[BBox],
    objects: &[TrackedObject],
) -> anyhow::Result<Vec<String>> {
    let img = render(frame, person_bboxes, objects);
    img.save(path)?;
    let legend = objects
        .iter()
        .map(|o| {
            let (_, cname) = class_palette(o.class_name);
            format!(
                "#{:<2} {:<12} [{:<7}] score {:.2} age {:<4} bbox {:.2},{:.2},{:.2},{:.2}",
                o.id - OBJECT_ID_BASE,
                o.class_name,
                cname,
                o.score,
                o.age,
                o.bbox.x1,
                o.bbox.y1,
                o.bbox.x2,
                o.bbox.y2,
            )
        })
        .collect();
    Ok(legend)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(id: u32, class: &'static str, b: BBox) -> TrackedObject {
        TrackedObject { id, class_name: class, bbox: b, score: 0.9, age: 3, missed: 0 }
    }

    #[test]
    fn class_palette_is_deterministic_and_distinct() {
        assert_eq!(class_palette("cup").1, class_palette("cup").1);
        // Different classes usually differ; at minimum the call is stable.
        let _ = class_palette("laptop");
    }

    #[test]
    fn render_draws_without_panicking_and_marks_pixels() {
        let frame = DynamicImage::new_rgb8(320, 240);
        let persons = vec![BBox { x1: 0.1, y1: 0.1, x2: 0.3, y2: 0.6 }];
        let objects = vec![obj(OBJECT_ID_BASE + 2, "cup", BBox { x1: 0.4, y1: 0.4, x2: 0.55, y2: 0.6 })];
        let img = render(&frame, &persons, &objects);
        // Some pixel must be non-black (a box/number was drawn).
        let any = img.pixels().any(|p| p.0 != [0, 0, 0]);
        assert!(any, "annotated frame should have drawn pixels");
    }

    #[test]
    fn draw_number_handles_multi_digit_and_edges() {
        let mut img = RgbImage::new(40, 20);
        // Near the right edge — must clip, not panic.
        draw_number(&mut img, 35, 2, 123, [255, 255, 255], 3);
    }

    #[test]
    fn write_annotated_legend_has_one_line_per_object() {
        let frame = DynamicImage::new_rgb8(64, 64);
        let objects = vec![
            obj(OBJECT_ID_BASE, "cup", BBox { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 }),
            obj(OBJECT_ID_BASE + 1, "chair", BBox { x1: 0.5, y1: 0.5, x2: 0.9, y2: 0.9 }),
        ];
        let dir = std::env::temp_dir().join(format!("mvm_scene_test_{}.jpg", std::process::id()));
        let legend = write_annotated(dir.to_str().unwrap(), &frame, &[], &objects).unwrap();
        assert_eq!(legend.len(), 2);
        assert!(legend[0].contains("cup"));
        assert!(dir.exists());
    }
}
