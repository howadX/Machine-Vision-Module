//! Scene visual debug (phase 8.E): draw EVERYTHING the detectors produce on a
//! frame, each detector ("agent") in its own colour. Zero new deps — manual
//! rectangle/line/dot fill + a tiny built-in 3×5 bitmap font for ids.
//!
//! This is the groundwork for the Agent Mesh (phase 8.C): one colour per agent,
//! so a glance at /tmp/mvm_scene.jpg shows which detector saw what.

use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use mvm_core::BBox;
use mvm_hands::Hand;
use mvm_scene::{OBJECT_ID_BASE, TrackedObject};

/// Per-agent colours (one detector = one colour). The Agent Mesh will key off this.
pub mod agent {
    /// Fused person track (the merged entity, not a raw detector).
    pub const PERSON: [u8; 3] = [255, 255, 255]; // white
    /// Face detector — YuNet.
    pub const FACE: [u8; 3] = [70, 230, 230]; // cyan
    /// Pose — BlazePose (mvm-posture) skeleton + landmarks.
    pub const POSE: [u8; 3] = [255, 215, 60]; // yellow
    /// Hands — MediaPipe handpose (mvm-hands).
    pub const HANDS: [u8; 3] = [40, 255, 120]; // green
    /// Objects — YOLO26n (mvm-scene).
    pub const OBJECT: [u8; 3] = [255, 150, 40]; // orange

    /// Legend rows: (label, colour).
    pub const LEGEND: [(&str, [u8; 3]); 5] = [
        ("person(fused)", PERSON),
        ("face/YuNet", FACE),
        ("pose/BlazePose", POSE),
        ("hands/handpose", HANDS),
        ("object/YOLO", OBJECT),
    ];
}

/// Everything the per-frame detectors produced, for one annotated frame. Filled by
/// the CLI (`build_body_primary`) so the visualiser draws every agent's output.
#[derive(Default)]
pub struct DetectorOverlay {
    /// YuNet face bboxes (normalized) — all detected, matched or not.
    pub faces: Vec<BBox>,
    /// BlazePose landmarks per body ([x,y]=px, [3]=visibility).
    pub poses: Vec<[[f32; 4]; 33]>,
    /// Hand landmarks (handpose).
    pub hands: Vec<Hand>,
}

/// BlazePose-33 skeleton edges (the main body; faces/feet trimmed for clarity).
const POSE_EDGES: &[(usize, usize)] = &[
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
];

/// 3×5 bitmap digits 0–9, row-major; bit 2 (0b100) is the leftmost column.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b001, 0b001, 0b001],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

fn px_clamp(v: f32, max: u32) -> u32 {
    (v.max(0.0) as u32).min(max.saturating_sub(1))
}

fn put(img: &mut RgbImage, x: i32, y: i32, c: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, Rgb(c));
    }
}

/// Hollow rectangle (normalized bbox), given thickness.
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

/// Filled square dot centred at (x,y) px.
fn draw_dot(img: &mut RgbImage, x: f32, y: f32, color: [u8; 3], r: i32) {
    let (cx, cy) = (x as i32, y as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            put(img, cx + dx, cy + dy, color);
        }
    }
}

/// Line from (x0,y0) to (x1,y1) in px (DDA).
fn draw_line(img: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 3]) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).max(1.0) as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        put(img, (x0 + dx * t) as i32, (y0 + dy * t) as i32, color);
    }
}

/// Draw an integer at (x,y) top-left, `scale` px per font cell.
fn draw_number(img: &mut RgbImage, x: u32, y: u32, n: u32, color: [u8; 3], scale: u32) {
    let mut cx = x;
    for ch in n.to_string().bytes() {
        let d = (ch - b'0') as usize;
        for (row, bits) in DIGITS[d].iter().enumerate() {
            for col in 0..3u32 {
                if bits & (1 << (2 - col)) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            put(
                                img,
                                (cx + col * scale + dx) as i32,
                                (y + row as u32 * scale + dy) as i32,
                                color,
                            );
                        }
                    }
                }
            }
        }
        cx += 4 * scale;
    }
}

const VIS_MIN: f32 = 0.5;

/// Render an annotated frame with every agent's detections in its own colour.
pub fn render(
    frame: &DynamicImage,
    tracks: &[BBox],
    objects: &[TrackedObject],
    overlay: &DetectorOverlay,
) -> RgbImage {
    let mut img = frame.to_rgb8();
    let (fw, fh) = frame.dimensions();
    let scale = (fh / 120).max(2);
    let dot_r = (fh / 240).max(1) as i32;

    // Objects (orange) with short id.
    for o in objects {
        draw_rect(&mut img, &o.bbox, agent::OBJECT, 2);
        let lx = px_clamp(o.bbox.x1 * fw as f32, fw);
        let ly = px_clamp(o.bbox.y1 * fh as f32, fh);
        draw_number(
            &mut img,
            lx + 2,
            ly + 2,
            o.id - OBJECT_ID_BASE,
            agent::OBJECT,
            scale,
        );
    }
    // Pose skeletons (yellow): bones + visible landmark dots.
    for lm in &overlay.poses {
        for &(a, b) in POSE_EDGES {
            if lm[a][3] >= VIS_MIN && lm[b][3] >= VIS_MIN {
                draw_line(
                    &mut img,
                    lm[a][0],
                    lm[a][1],
                    lm[b][0],
                    lm[b][1],
                    agent::POSE,
                );
            }
        }
        for p in lm.iter() {
            if p[3] >= VIS_MIN {
                draw_dot(&mut img, p[0], p[1], agent::POSE, dot_r);
            }
        }
    }
    // Faces (cyan).
    for fb in &overlay.faces {
        draw_rect(&mut img, fb, agent::FACE, 2);
    }
    // Hands (green): box + 21 dots.
    for hand in &overlay.hands {
        draw_rect(
            &mut img,
            &hand.bbox_norm(fw as f32, fh as f32),
            agent::HANDS,
            2,
        );
        for p in &hand.landmarks_px {
            draw_dot(&mut img, p[0], p[1], agent::HANDS, dot_r);
        }
    }
    // Fused person tracks (white) on top.
    for b in tracks {
        draw_rect(&mut img, b, agent::PERSON, 2);
    }
    img
}

/// Render + save the annotated frame; return a printable legend (agent colour key
/// + counts + one line per object).
pub fn write_annotated(
    path: &str,
    frame: &DynamicImage,
    tracks: &[BBox],
    objects: &[TrackedObject],
    overlay: &DetectorOverlay,
) -> anyhow::Result<Vec<String>> {
    let img = render(frame, tracks, objects, overlay);
    img.save(path)?;

    let mut legend: Vec<String> = Vec::new();
    let key: Vec<String> = agent::LEGEND
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    legend.push(format!("agents: {}", key.join(" | ")));
    legend.push(format!(
        "counts: persons {} | faces {} | poses {} | hands {} | objects {}",
        tracks.len(),
        overlay.faces.len(),
        overlay.poses.len(),
        overlay.hands.len(),
        objects.len()
    ));
    for o in objects {
        legend.push(format!(
            "#{:<2} {:<12} score {:.2} age {:<4} bbox {:.2},{:.2},{:.2},{:.2}",
            o.id - OBJECT_ID_BASE,
            o.class_name,
            o.score,
            o.age,
            o.bbox.x1,
            o.bbox.y1,
            o.bbox.x2,
            o.bbox.y2,
        ));
    }
    Ok(legend)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(id: u32, class: &'static str, b: BBox) -> TrackedObject {
        TrackedObject {
            id,
            class_name: class,
            bbox: b,
            score: 0.9,
            age: 3,
            missed: 0,
            lifecycle: mvm_core::types::LifecycleState::Confirmed,
            pos_3d: None,
            pos_conf: 0.0,
            mask: None,
        }
    }

    #[test]
    fn agent_legend_has_all_detectors() {
        let names: Vec<&str> = agent::LEGEND.iter().map(|(n, _)| *n).collect();
        assert!(names.iter().any(|n| n.contains("face")));
        assert!(names.iter().any(|n| n.contains("pose")));
        assert!(names.iter().any(|n| n.contains("hands")));
        assert!(names.iter().any(|n| n.contains("object")));
    }

    #[test]
    fn render_draws_all_layers_without_panicking() {
        let frame = DynamicImage::new_rgb8(320, 240);
        let tracks = vec![BBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.3,
            y2: 0.6,
        }];
        let objects = vec![obj(
            OBJECT_ID_BASE + 2,
            "cup",
            BBox {
                x1: 0.4,
                y1: 0.4,
                x2: 0.55,
                y2: 0.6,
            },
        )];
        let mut overlay = DetectorOverlay::default();
        overlay.faces.push(BBox {
            x1: 0.15,
            y1: 0.12,
            x2: 0.28,
            y2: 0.3,
        });
        let mut lm = [[0.0f32, 0.0, 0.0, 0.9]; 33];
        lm[11] = [100.0, 100.0, 0.0, 0.9];
        lm[12] = [140.0, 100.0, 0.0, 0.9];
        overlay.poses.push(lm);
        let img = render(&frame, &tracks, &objects, &overlay);
        assert!(
            img.pixels().any(|p| p.0 != [0, 0, 0]),
            "should have drawn pixels"
        );
    }

    #[test]
    fn draw_number_handles_multi_digit_and_edges() {
        let mut img = RgbImage::new(40, 20);
        draw_number(&mut img, 35, 2, 123, [255, 255, 255], 3);
    }

    #[test]
    fn write_annotated_legend_reports_counts() {
        let frame = DynamicImage::new_rgb8(64, 64);
        let objects = vec![obj(
            OBJECT_ID_BASE,
            "cup",
            BBox {
                x1: 0.1,
                y1: 0.1,
                x2: 0.2,
                y2: 0.2,
            },
        )];
        let overlay = DetectorOverlay::default();
        let dir = std::env::temp_dir().join(format!("mvm_scene_test_{}.jpg", std::process::id()));
        let legend =
            write_annotated(dir.to_str().unwrap(), &frame, &[], &objects, &overlay).unwrap();
        assert!(legend[0].contains("agents:"));
        assert!(legend.iter().any(|l| l.contains("objects 1")));
        assert!(legend.iter().any(|l| l.contains("cup")));
        assert!(dir.exists());
    }
}
