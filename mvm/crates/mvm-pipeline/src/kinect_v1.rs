//! Optional direct Kinect v1 RGB-D capture through a pinned local libfreenect.
//!
//! This module is compiled only with `kinect-v1`. The shared library is loaded
//! from an explicit prefix at runtime, so ordinary builds and releases retain
//! no Kinect runtime dependency.

use std::collections::VecDeque;
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, RgbImage};
use libloading::Library;
use mvm_core::{
    CalibrationRef, DepthGeometry, DepthStatus, MetricDepthFrame, RegisteredDepthOwners,
    SensorClockDomain, SensorHealth, SensorSourceId, SensorStamp,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::source::{CapturedSensorFrame, SensorSource, monotonic_ns, unix_ms_now};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const RGB_BYTES: usize = WIDTH as usize * HEIGHT as usize * 3;
const DEPTH_PIXELS: usize = WIDTH as usize * HEIGHT as usize;
const QUEUE_CAPACITY: usize = 3;
const GAP_DEVICE_TICKS: u32 = 3_000_000;
const PINNED_LIBRARY: &str = "libfreenect.so.0.7.5";
const READ_PROGRESS_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_METRIC_DEPTH_MM: usize = 10_000;
const NO_OWNER: u32 = RegisteredDepthOwners::NO_OWNER;

#[derive(Debug, Clone)]
pub struct KinectV1Config {
    pub serial: String,
    pub logical_name: String,
    pub calibration_path: PathBuf,
    pub library_prefix: PathBuf,
    pub max_pair_skew_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KinectV1Probe {
    pub library_path: PathBuf,
    pub calibration: CalibrationRef,
    pub usb_camera_present: bool,
    native_intrinsics: [f32; 4],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationProfile {
    schema: String,
    sensor: CalibrationSensor,
    software: CalibrationSoftware,
    factory_calibration: FactoryCalibrationProfile,
    #[serde(default)]
    mount: serde_json::Value,
    #[serde(default)]
    validated_capabilities: serde_json::Value,
    #[serde(default)]
    limits: serde_json::Value,
    #[serde(default)]
    calibrated_at: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FactoryCalibrationProfile {
    authority: String,
    resolution: [u32; 2],
    depth_intrinsics: DepthIntrinsicsProfile,
    zero_plane: serde_json::Value,
    derived_baselines_m: serde_json::Value,
    rgb_depth_registration: String,
    metric_depth: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DepthIntrinsicsProfile {
    fx_px: f32,
    fy_px: f32,
    cx_px: f32,
    cy_px: f32,
    distortion: String,
}

#[derive(Debug, Clone)]
struct LoadedCalibration {
    reference: CalibrationRef,
    native_intrinsics: [f32; 4],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationSensor {
    logical_name: String,
    generation: String,
    camera_serial: String,
    audio_serial: String,
    usb_ids: serde_json::Value,
    stable_rgb_device: String,
    preferred_usb_path_at_calibration: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationSoftware {
    libfreenect_version: String,
    libfreenect_commit: String,
    runtime_prefix: String,
    firmware_path: String,
    firmware_sha256: String,
}

fn load_calibration(config: &KinectV1Config) -> Result<LoadedCalibration> {
    let bytes = std::fs::read(&config.calibration_path).with_context(|| {
        format!(
            "read Kinect calibration {}",
            config.calibration_path.display()
        )
    })?;
    let profile: CalibrationProfile = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "parse Kinect calibration {}",
            config.calibration_path.display()
        )
    })?;
    ensure!(
        profile.schema == "kinect-v1-sensor-calibration/1.0",
        "unsupported Kinect calibration schema {}",
        profile.schema
    );
    ensure!(
        profile.sensor.camera_serial == config.serial,
        "Kinect calibration serial {} does not match configured {}",
        profile.sensor.camera_serial,
        config.serial
    );
    ensure!(
        profile.sensor.logical_name == config.logical_name,
        "Kinect calibration logical name {} does not match configured {}",
        profile.sensor.logical_name,
        config.logical_name
    );
    ensure!(
        profile.software.libfreenect_version == "0.7.5",
        "Kinect calibration requires libfreenect {}, expected 0.7.5",
        profile.software.libfreenect_version
    );
    let _provenance_fields = (
        &profile.sensor.generation,
        &profile.sensor.audio_serial,
        &profile.sensor.usb_ids,
        &profile.sensor.stable_rgb_device,
        &profile.sensor.preferred_usb_path_at_calibration,
        &profile.software.libfreenect_commit,
        &profile.software.runtime_prefix,
        &profile.software.firmware_path,
        &profile.software.firmware_sha256,
        &profile.factory_calibration.authority,
        &profile.factory_calibration.zero_plane,
        &profile.factory_calibration.derived_baselines_m,
        &profile.factory_calibration.rgb_depth_registration,
        &profile.factory_calibration.metric_depth,
        &profile.factory_calibration.depth_intrinsics.distortion,
        &profile.mount,
        &profile.validated_capabilities,
        &profile.limits,
        &profile.calibrated_at,
    );
    let revision_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let reference = CalibrationRef {
        schema: profile.schema,
        camera_serial: profile.sensor.camera_serial,
        revision_sha256,
    };
    reference
        .validate()
        .context("invalid Kinect calibration reference")?;
    ensure!(
        profile.factory_calibration.resolution == [WIDTH, HEIGHT],
        "Kinect native depth intrinsics resolution mismatch"
    );
    let i = profile.factory_calibration.depth_intrinsics;
    ensure!(
        [i.fx_px, i.fy_px, i.cx_px, i.cy_px]
            .iter()
            .all(|value| value.is_finite())
            && i.fx_px > 0.0
            && i.fy_px > 0.0,
        "invalid Kinect native depth intrinsics"
    );
    Ok(LoadedCalibration {
        reference,
        native_intrinsics: [i.fx_px, i.fy_px, i.cx_px, i.cy_px],
    })
}

fn library_path(prefix: &Path) -> PathBuf {
    prefix.join("lib").join(PINNED_LIBRARY)
}

pub fn probe(config: &KinectV1Config) -> Result<KinectV1Probe> {
    ensure!(
        !config.serial.trim().is_empty(),
        "Kinect camera serial is required"
    );
    ensure!(
        !config.logical_name.trim().is_empty(),
        "Kinect logical name is required"
    );
    ensure!(
        config.max_pair_skew_ms > 0,
        "Kinect max_pair_skew_ms must be positive"
    );
    let path = library_path(&config.library_prefix);
    ensure!(
        path.is_file(),
        "pinned Kinect runtime missing: {}",
        path.display()
    );
    let calibration = load_calibration(config)?;
    Ok(KinectV1Probe {
        library_path: path,
        calibration: calibration.reference,
        usb_camera_present: usb_camera_serial_present(&config.serial),
        native_intrinsics: calibration.native_intrinsics,
    })
}

fn usb_camera_serial_present(serial: &str) -> bool {
    std::fs::read_dir("/sys/bus/usb/devices")
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            let root = entry.path();
            read_trimmed(root.join("idVendor")).as_deref() == Some("045e")
                && read_trimmed(root.join("idProduct")).as_deref() == Some("02ae")
                && read_trimmed(root.join("serial")).as_deref() == Some(serial)
        })
}

fn read_trimmed(path: PathBuf) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_owned())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrameMode {
    reserved: c_uint,
    resolution: c_int,
    format: c_int,
    bytes: c_int,
    width: i16,
    height: i16,
    data_bits_per_pixel: i8,
    padding_bits_per_pixel: i8,
    framerate: i8,
    is_valid: i8,
}

#[repr(C)]
struct TimeVal {
    tv_sec: c_long,
    tv_usec: c_long,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RegistrationInfo {
    values: [i32; 29],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RegistrationPadInfo {
    start_lines: u16,
    end_lines: u16,
    cropping_lines: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ZeroPlaneInfo {
    dcmos_emitter_dist: f32,
    dcmos_rcmos_dist: f32,
    reference_distance: f32,
    reference_pixel_size: f32,
}

#[repr(C)]
struct FreenectRegistration {
    reg_info: RegistrationInfo,
    reg_pad_info: RegistrationPadInfo,
    zero_plane_info: ZeroPlaneInfo,
    const_shift: f64,
    raw_to_mm_shift: *mut u16,
    depth_to_rgb_shift: *mut i32,
    registration_table: *mut [i32; 2],
}

#[derive(Debug, Clone)]
struct FactoryRegistration {
    target_offset: i32,
    depth_to_rgb_shift: Vec<i32>,
    registration_table: Vec<[i32; 2]>,
}

impl FactoryRegistration {
    unsafe fn copy_from_api(api: &Api, device: *mut FreenectDevice) -> Result<Self> {
        // SAFETY: the device is open and the pinned ABI returns owned table copies.
        let mut registration = unsafe { (api.copy_registration)(device) };
        ensure!(
            !registration.depth_to_rgb_shift.is_null()
                && !registration.registration_table.is_null(),
            "libfreenect returned incomplete factory registration"
        );
        // SAFETY: libfreenect allocates these fixed-size arrays for the copied registration.
        let depth_to_rgb_shift = unsafe {
            std::slice::from_raw_parts(registration.depth_to_rgb_shift, MAX_METRIC_DEPTH_MM)
        }
        .to_vec();
        // SAFETY: the registration table always contains one pair per native depth pixel.
        let registration_table =
            unsafe { std::slice::from_raw_parts(registration.registration_table, DEPTH_PIXELS) }
                .to_vec();
        let target_offset = i32::from(registration.reg_pad_info.start_lines)
            .checked_mul(HEIGHT as i32)
            .context("invalid Kinect registration padding")?;
        // SAFETY: destroy releases only the API-owned copies after Rust cloned both tables.
        ensure!(
            unsafe { (api.destroy_registration)(&mut registration) } == 0,
            "destroy copied Kinect registration failed"
        );
        Ok(Self {
            target_offset,
            depth_to_rgb_shift,
            registration_table,
        })
    }

    fn register(&self, native_depth_mm: &[u16]) -> RegisteredOutput {
        let mut depth_mm = vec![0_u16; DEPTH_PIXELS];
        let mut owner_index = vec![NO_OWNER; DEPTH_PIXELS];
        let mut collisions = 0_u64;
        let mut out_of_bounds = 0_u64;
        for (source, &depth) in native_depth_mm.iter().take(DEPTH_PIXELS).enumerate() {
            let depth_index = usize::from(depth);
            if depth == 0 || depth_index >= self.depth_to_rgb_shift.len() {
                continue;
            }
            let base = self.registration_table[source];
            let shifted_x = base[0].saturating_add(self.depth_to_rgb_shift[depth_index]);
            if shifted_x < 0 {
                out_of_bounds = out_of_bounds.saturating_add(1);
                continue;
            }
            let nx = shifted_x / 256;
            let ny = base[1] - self.target_offset;
            if nx < 0 || nx >= WIDTH as i32 || ny < 0 || ny >= HEIGHT as i32 {
                out_of_bounds = out_of_bounds.saturating_add(1);
                continue;
            }
            let target = ny as usize * WIDTH as usize + nx as usize;
            if depth_mm[target] != 0 {
                collisions = collisions.saturating_add(1);
            }
            if depth_mm[target] == 0 || depth < depth_mm[target] {
                depth_mm[target] = depth;
                owner_index[target] = source as u32;
            }
        }
        RegisteredOutput {
            depth_mm,
            owner_index,
            collisions,
            out_of_bounds,
        }
    }
}

#[derive(Debug)]
struct RegisteredOutput {
    depth_mm: Vec<u16>,
    owner_index: Vec<u32>,
    collisions: u64,
    out_of_bounds: u64,
}

enum FreenectContext {}
enum FreenectDevice {}
type FrameCallback = unsafe extern "C" fn(*mut FreenectDevice, *mut c_void, c_uint);
type LogCallback = unsafe extern "C" fn(*mut FreenectContext, c_int, *const c_char);

struct Api {
    _library: Library,
    init: unsafe extern "C" fn(*mut *mut FreenectContext, *mut c_void) -> c_int,
    shutdown: unsafe extern "C" fn(*mut FreenectContext) -> c_int,
    set_log_level: unsafe extern "C" fn(*mut FreenectContext, c_int),
    set_log_callback: unsafe extern "C" fn(*mut FreenectContext, Option<LogCallback>),
    select_subdevices: unsafe extern "C" fn(*mut FreenectContext, c_int),
    open_by_serial: unsafe extern "C" fn(
        *mut FreenectContext,
        *mut *mut FreenectDevice,
        *const c_char,
    ) -> c_int,
    close_device: unsafe extern "C" fn(*mut FreenectDevice) -> c_int,
    set_user: unsafe extern "C" fn(*mut FreenectDevice, *mut c_void),
    find_video_mode: unsafe extern "C" fn(c_int, c_int) -> FrameMode,
    find_depth_mode: unsafe extern "C" fn(c_int, c_int) -> FrameMode,
    set_video_mode: unsafe extern "C" fn(*mut FreenectDevice, FrameMode) -> c_int,
    set_depth_mode: unsafe extern "C" fn(*mut FreenectDevice, FrameMode) -> c_int,
    set_video_callback: unsafe extern "C" fn(*mut FreenectDevice, Option<FrameCallback>),
    set_depth_callback: unsafe extern "C" fn(*mut FreenectDevice, Option<FrameCallback>),
    start_video: unsafe extern "C" fn(*mut FreenectDevice) -> c_int,
    start_depth: unsafe extern "C" fn(*mut FreenectDevice) -> c_int,
    stop_video: unsafe extern "C" fn(*mut FreenectDevice) -> c_int,
    stop_depth: unsafe extern "C" fn(*mut FreenectDevice) -> c_int,
    process_events_timeout: unsafe extern "C" fn(*mut FreenectContext, *mut TimeVal) -> c_int,
    copy_registration: unsafe extern "C" fn(*mut FreenectDevice) -> FreenectRegistration,
    destroy_registration: unsafe extern "C" fn(*mut FreenectRegistration) -> c_int,
}

impl Api {
    unsafe fn load(path: &Path) -> Result<Self> {
        // SAFETY: the path is the explicit versioned library validated by `probe`.
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load pinned Kinect runtime {}", path.display()))?;
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                // SAFETY: signatures mirror libfreenect 0.7.5's public C header.
                *unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .with_context(|| format!("resolve libfreenect symbol {}", $name))?
            }};
        }
        Ok(Self {
            init: symbol!(
                "freenect_init",
                unsafe extern "C" fn(*mut *mut FreenectContext, *mut c_void) -> c_int
            ),
            shutdown: symbol!(
                "freenect_shutdown",
                unsafe extern "C" fn(*mut FreenectContext) -> c_int
            ),
            set_log_level: symbol!(
                "freenect_set_log_level",
                unsafe extern "C" fn(*mut FreenectContext, c_int)
            ),
            set_log_callback: symbol!(
                "freenect_set_log_callback",
                unsafe extern "C" fn(*mut FreenectContext, Option<LogCallback>)
            ),
            select_subdevices: symbol!(
                "freenect_select_subdevices",
                unsafe extern "C" fn(*mut FreenectContext, c_int)
            ),
            open_by_serial: symbol!(
                "freenect_open_device_by_camera_serial",
                unsafe extern "C" fn(
                    *mut FreenectContext,
                    *mut *mut FreenectDevice,
                    *const c_char,
                ) -> c_int
            ),
            close_device: symbol!(
                "freenect_close_device",
                unsafe extern "C" fn(*mut FreenectDevice) -> c_int
            ),
            set_user: symbol!(
                "freenect_set_user",
                unsafe extern "C" fn(*mut FreenectDevice, *mut c_void)
            ),
            find_video_mode: symbol!(
                "freenect_find_video_mode",
                unsafe extern "C" fn(c_int, c_int) -> FrameMode
            ),
            find_depth_mode: symbol!(
                "freenect_find_depth_mode",
                unsafe extern "C" fn(c_int, c_int) -> FrameMode
            ),
            set_video_mode: symbol!(
                "freenect_set_video_mode",
                unsafe extern "C" fn(*mut FreenectDevice, FrameMode) -> c_int
            ),
            set_depth_mode: symbol!(
                "freenect_set_depth_mode",
                unsafe extern "C" fn(*mut FreenectDevice, FrameMode) -> c_int
            ),
            set_video_callback: symbol!(
                "freenect_set_video_callback",
                unsafe extern "C" fn(*mut FreenectDevice, Option<FrameCallback>)
            ),
            set_depth_callback: symbol!(
                "freenect_set_depth_callback",
                unsafe extern "C" fn(*mut FreenectDevice, Option<FrameCallback>)
            ),
            start_video: symbol!(
                "freenect_start_video",
                unsafe extern "C" fn(*mut FreenectDevice) -> c_int
            ),
            start_depth: symbol!(
                "freenect_start_depth",
                unsafe extern "C" fn(*mut FreenectDevice) -> c_int
            ),
            stop_video: symbol!(
                "freenect_stop_video",
                unsafe extern "C" fn(*mut FreenectDevice) -> c_int
            ),
            stop_depth: symbol!(
                "freenect_stop_depth",
                unsafe extern "C" fn(*mut FreenectDevice) -> c_int
            ),
            process_events_timeout: symbol!(
                "freenect_process_events_timeout",
                unsafe extern "C" fn(*mut FreenectContext, *mut TimeVal) -> c_int
            ),
            copy_registration: symbol!(
                "freenect_copy_registration",
                unsafe extern "C" fn(*mut FreenectDevice) -> FreenectRegistration
            ),
            destroy_registration: symbol!(
                "freenect_destroy_registration",
                unsafe extern "C" fn(*mut FreenectRegistration) -> c_int
            ),
            _library: library,
        })
    }
}

struct RgbSample {
    bytes: Vec<u8>,
    stamp: SensorStamp,
}

struct DepthSample {
    depth_mm: Vec<u16>,
    owner_index: Vec<u32>,
    stamp: SensorStamp,
    valid_ratio: f32,
}

#[derive(Default)]
struct CallbackState {
    rgb: VecDeque<RgbSample>,
    depth: VecDeque<DepthSample>,
    rgb_sequence: u64,
    depth_sequence: u64,
    last_rgb_timestamp: Option<u32>,
    last_depth_timestamp: Option<u32>,
    health: SensorHealth,
    registration: Option<FactoryRegistration>,
}

impl CallbackState {
    fn observe_timestamp(
        last: &mut Option<u32>,
        current: u32,
        gaps: &mut u64,
        health: &mut SensorHealth,
    ) {
        if let Some(previous) = *last {
            let delta = current.wrapping_sub(previous);
            if delta == 0 {
                health.duplicate_timestamps = health.duplicate_timestamps.saturating_add(1);
            } else if delta > u32::MAX / 2 {
                health.out_of_order_timestamps = health.out_of_order_timestamps.saturating_add(1);
            } else if delta > GAP_DEVICE_TICKS {
                *gaps = gaps.saturating_add(1);
            }
        }
        *last = Some(current);
    }
}

unsafe extern "C" fn rgb_callback(dev: *mut FreenectDevice, data: *mut c_void, timestamp: c_uint) {
    let _ = dev;
    if data.is_null() {
        return;
    }
    let state_ptr = CALLBACK_STATE.load(Ordering::Acquire);
    if state_ptr.is_null() {
        return;
    }
    // SAFETY: callbacks run on the owning event thread; drop clears this pointer
    // only after both streams have stopped.
    let state = unsafe { &mut *state_ptr };
    let sequence = state.rgb_sequence;
    let stamp = kinect_stamp(sequence, timestamp, SensorClockDomain::KinectV1Rgb);
    let mut gaps = state.health.rgb_gaps;
    CallbackState::observe_timestamp(
        &mut state.last_rgb_timestamp,
        timestamp,
        &mut gaps,
        &mut state.health,
    );
    state.health.rgb_gaps = gaps;
    // SAFETY: RGB medium mode supplies exactly 640*480*3 bytes during callback.
    let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), RGB_BYTES) }.to_vec();
    state.rgb_sequence = sequence.saturating_add(1);
    if state.rgb.len() == QUEUE_CAPACITY {
        state.rgb.pop_front();
        state.health.rgb_only = state.health.rgb_only.saturating_add(1);
    }
    state.rgb.push_back(RgbSample { bytes, stamp });
}

unsafe extern "C" fn depth_callback(
    dev: *mut FreenectDevice,
    data: *mut c_void,
    timestamp: c_uint,
) {
    let _ = dev;
    if data.is_null() {
        return;
    }
    let state_ptr = CALLBACK_STATE.load(Ordering::Acquire);
    if state_ptr.is_null() {
        return;
    }
    // SAFETY: callbacks run on the owning event thread; drop clears this pointer
    // only after both streams have stopped.
    let state = unsafe { &mut *state_ptr };
    let sequence = state.depth_sequence;
    let stamp = kinect_stamp(sequence, timestamp, SensorClockDomain::KinectV1Depth);
    let mut gaps = state.health.depth_gaps;
    CallbackState::observe_timestamp(
        &mut state.last_depth_timestamp,
        timestamp,
        &mut gaps,
        &mut state.health,
    );
    state.health.depth_gaps = gaps;
    // SAFETY: native metric medium mode supplies exactly 640*480 u16 pixels.
    let native_depth_mm = unsafe { std::slice::from_raw_parts(data.cast::<u16>(), DEPTH_PIXELS) };
    let Some(registration) = state.registration.as_ref() else {
        state.health.resync_errors = state.health.resync_errors.saturating_add(1);
        return;
    };
    let registered = registration.register(native_depth_mm);
    let depth_mm = registered.depth_mm;
    let owner_index = registered.owner_index;
    let _registration_diagnostics = (registered.collisions, registered.out_of_bounds);
    let valid = depth_mm.iter().filter(|&&value| value != 0).count();
    state.depth_sequence = sequence.saturating_add(1);
    if state.depth.len() == QUEUE_CAPACITY {
        state.depth.pop_front();
        state.health.depth_dropped = state.health.depth_dropped.saturating_add(1);
    }
    state.depth.push_back(DepthSample {
        depth_mm,
        owner_index,
        stamp,
        valid_ratio: valid as f32 / DEPTH_PIXELS as f32,
    });
}

// Direct Kinect capture is an explicit single-owner backend. libfreenect invokes
// these callbacks synchronously from `process_events_timeout`.
static CALLBACK_STATE: AtomicPtr<CallbackState> = AtomicPtr::new(ptr::null_mut());

unsafe extern "C" fn log_callback(
    _context: *mut FreenectContext,
    _level: c_int,
    message: *const c_char,
) {
    if message.is_null() {
        return;
    }
    // SAFETY: libfreenect provides a NUL-terminated message for the duration of
    // this callback.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let lower = text.to_ascii_lowercase();
    if lower.contains("resync") || lower.contains("lost too many packets") {
        let state_ptr = CALLBACK_STATE.load(Ordering::Acquire);
        if !state_ptr.is_null() {
            // SAFETY: logging and frame callbacks run on the owning event thread.
            let state = unsafe { &mut *state_ptr };
            state.health.resync_errors = state.health.resync_errors.saturating_add(1);
        }
    }
    eprint!("{text}");
}

fn kinect_stamp(sequence: u64, raw: u32, domain: SensorClockDomain) -> SensorStamp {
    SensorStamp {
        source_sequence: sequence,
        host_monotonic_ns: monotonic_ns(),
        wall_clock_unix_ms: unix_ms_now(),
        clock_domain: domain,
        device_timestamp_raw: Some(raw),
    }
}

pub struct KinectV1Source {
    api: Api,
    context: *mut FreenectContext,
    device: *mut FreenectDevice,
    state: Box<CallbackState>,
    source: SensorSourceId,
    calibration: CalibrationRef,
    native_intrinsics: [f32; 4],
    max_pair_skew_ns: u64,
    video_started: bool,
    depth_started: bool,
}

// The source and its C handles are owned and used by the capture thread only.
unsafe impl Send for KinectV1Source {}

impl KinectV1Source {
    pub fn open(config: KinectV1Config) -> Result<Self> {
        let probe = probe(&config)?;
        ensure!(
            probe.usb_camera_present,
            "Kinect camera serial {} is not present",
            config.serial
        );
        // SAFETY: symbol signatures and pinned library were validated above.
        let api = unsafe { Api::load(&probe.library_path) }?;
        let mut context = ptr::null_mut();
        // SAFETY: libfreenect initializes the output context pointer.
        ensure!(
            unsafe { (api.init)(&mut context, ptr::null_mut()) } >= 0,
            "freenect_init failed"
        );
        // Camera only: this backend must not claim or command motor/audio.
        unsafe {
            (api.set_log_level)(context, 2);
            (api.set_log_callback)(context, Some(log_callback));
            (api.select_subdevices)(context, 0x02);
        }
        let serial = CString::new(config.serial.as_str()).context("Kinect serial contains NUL")?;
        let mut device = ptr::null_mut();
        if unsafe { (api.open_by_serial)(context, &mut device, serial.as_ptr()) } < 0 {
            unsafe { (api.shutdown)(context) };
            bail!(
                "open Kinect camera serial {} failed (missing, busy, or permission denied)",
                config.serial
            );
        }
        let video_mode = unsafe { (api.find_video_mode)(1, 0) };
        let depth_mode = unsafe { (api.find_depth_mode)(1, 5) };
        if video_mode.is_valid == 0
            || depth_mode.is_valid == 0
            || video_mode.width != WIDTH as i16
            || video_mode.height != HEIGHT as i16
            || depth_mode.width != WIDTH as i16
            || depth_mode.height != HEIGHT as i16
        {
            unsafe {
                (api.close_device)(device);
                (api.shutdown)(context);
            }
            bail!(
                "libfreenect 0.7.5 does not expose required 640x480 RGB + native metric-depth modes"
            );
        }
        if unsafe { (api.set_video_mode)(device, video_mode) } < 0
            || unsafe { (api.set_depth_mode)(device, depth_mode) } < 0
        {
            unsafe {
                (api.close_device)(device);
                (api.shutdown)(context);
            }
            bail!("configure Kinect RGB-D modes failed");
        }
        // SAFETY: the open serial-bound device owns stable EEPROM calibration.
        let registration = match unsafe { FactoryRegistration::copy_from_api(&api, device) } {
            Ok(registration) => registration,
            Err(error) => {
                unsafe {
                    (api.close_device)(device);
                    (api.shutdown)(context);
                }
                return Err(error);
            }
        };
        let mut state = Box::<CallbackState>::default();
        state.registration = Some(registration);
        if CALLBACK_STATE
            .compare_exchange(
                ptr::null_mut(),
                &mut *state,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            unsafe {
                (api.close_device)(device);
                (api.shutdown)(context);
            }
            bail!("another Kinect v1 source already owns the process callback state");
        }
        unsafe {
            (api.set_user)(device, (&mut *state as *mut CallbackState).cast());
            (api.set_video_callback)(device, Some(rgb_callback));
            (api.set_depth_callback)(device, Some(depth_callback));
        }
        if unsafe { (api.start_depth)(device) } < 0 {
            CALLBACK_STATE.store(ptr::null_mut(), Ordering::Release);
            unsafe {
                (api.close_device)(device);
                (api.shutdown)(context);
            }
            bail!("start Kinect native metric-depth stream failed");
        }
        if unsafe { (api.start_video)(device) } < 0 {
            CALLBACK_STATE.store(ptr::null_mut(), Ordering::Release);
            unsafe {
                (api.stop_depth)(device);
                (api.close_device)(device);
                (api.shutdown)(context);
            }
            bail!("start Kinect RGB stream failed");
        }
        Ok(Self {
            api,
            context,
            device,
            state,
            source: SensorSourceId {
                backend: "kinect_v1".into(),
                logical_name: config.logical_name,
                serial: Some(config.serial),
            },
            calibration: probe.calibration,
            native_intrinsics: probe.native_intrinsics,
            max_pair_skew_ns: config.max_pair_skew_ms.saturating_mul(1_000_000),
            video_started: true,
            depth_started: true,
        })
    }

    fn take_packet(&mut self) -> Option<CapturedSensorFrame> {
        let (rgb, depth, skew) =
            take_strict_pair(&mut self.state, self.max_pair_skew_ns, monotonic_ns())?;
        let owners = RegisteredDepthOwners {
            registered_width: WIDTH,
            registered_height: HEIGHT,
            native_width: WIDTH,
            native_height: HEIGHT,
            native_fx_px: self.native_intrinsics[0],
            native_fy_px: self.native_intrinsics[1],
            native_cx_px: self.native_intrinsics[2],
            native_cy_px: self.native_intrinsics[3],
            owner_index: depth.owner_index,
            calibration_revision: self.calibration.revision_sha256.clone(),
        };
        debug_assert!(owners.validate().is_ok());
        let metric_depth = Arc::new(MetricDepthFrame {
            width: WIDTH,
            height: HEIGHT,
            depth_mm: depth.depth_mm,
            invalid_value: 0,
            stamp: depth.stamp,
            geometry: DepthGeometry::RegisteredToRgb,
            calibration_revision: self.calibration.revision_sha256.clone(),
            valid_ratio: depth.valid_ratio,
            rgb_depth_skew_ns: skew,
        });
        let image =
            RgbImage::from_raw(WIDTH, HEIGHT, rgb.bytes).expect("callback RGB size is fixed");
        Some(CapturedSensorFrame {
            rgb: Arc::new(DynamicImage::ImageRgb8(image)),
            rgb_stamp: rgb.stamp,
            source: self.source.clone(),
            depth_status: DepthStatus::Paired,
            metric_depth: Some(metric_depth),
            registered_depth_owners: Some(Arc::new(owners)),
            calibration: Some(self.calibration.clone()),
            health: self.state.health.clone(),
        })
    }
}

fn take_strict_pair(
    state: &mut CallbackState,
    max_pair_skew_ns: u64,
    now_ns: u64,
) -> Option<(RgbSample, DepthSample, u64)> {
    loop {
        let rgb = state.rgb.front()?;
        let newest_depth_ns = state.depth.back().map(|d| d.stamp.host_monotonic_ns);
        let ready = newest_depth_ns.is_some_and(|ns| ns >= rgb.stamp.host_monotonic_ns)
            || now_ns.saturating_sub(rgb.stamp.host_monotonic_ns) > max_pair_skew_ns;
        if !ready {
            return None;
        }
        let best = state
            .depth
            .iter()
            .enumerate()
            .min_by_key(|(_, depth)| rgb.stamp.host_skew_ns(&depth.stamp))
            .map(|(index, depth)| (index, rgb.stamp.host_skew_ns(&depth.stamp)));
        let Some((index, skew)) = best.filter(|(_, skew)| *skew <= max_pair_skew_ns) else {
            state.rgb.pop_front();
            state.health.rgb_only = state.health.rgb_only.saturating_add(1);
            continue;
        };
        let rgb = state.rgb.pop_front().expect("front RGB exists");
        for _ in 0..index {
            state.depth.pop_front();
            state.health.depth_dropped = state.health.depth_dropped.saturating_add(1);
        }
        let depth = state.depth.pop_front().expect("selected depth exists");
        state.health.paired = state.health.paired.saturating_add(1);
        return Some((rgb, depth, skew));
    }
}

impl SensorSource for KinectV1Source {
    fn read(&mut self) -> Result<CapturedSensorFrame> {
        let deadline = Instant::now() + READ_PROGRESS_TIMEOUT;
        loop {
            if let Some(packet) = self.take_packet() {
                return Ok(packet);
            }
            let mut timeout = TimeVal {
                tv_sec: 0,
                tv_usec: 250_000,
            };
            let rc = unsafe { (self.api.process_events_timeout)(self.context, &mut timeout) };
            if rc < 0 && rc != -10 {
                self.state.health.resync_errors = self.state.health.resync_errors.saturating_add(1);
                bail!("Kinect event processing failed: {rc}");
            }
            if Instant::now() >= deadline {
                self.state.health.resync_errors = self.state.health.resync_errors.saturating_add(1);
                bail!(
                    "Kinect RGB-D packet timeout after {}s (rgb queued {}, depth queued {}, health {})",
                    READ_PROGRESS_TIMEOUT.as_secs(),
                    self.state.rgb.len(),
                    self.state.depth.len(),
                    serde_json::to_string(&self.state.health)?
                );
            }
        }
    }

    fn dimensions(&self) -> (u32, u32) {
        (WIDTH, HEIGHT)
    }
    fn device(&self) -> &str {
        &self.source.logical_name
    }
}

impl Drop for KinectV1Source {
    fn drop(&mut self) {
        unsafe {
            if self.video_started {
                (self.api.stop_video)(self.device);
            }
            if self.depth_started {
                (self.api.stop_depth)(self.device);
            }
            CALLBACK_STATE.store(ptr::null_mut(), Ordering::Release);
            (self.api.set_user)(self.device, ptr::null_mut());
            (self.api.close_device)(self.device);
            (self.api.shutdown)(self.context);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_runtime_path_is_versioned() {
        assert_eq!(
            library_path(Path::new("/opt/freenect")),
            PathBuf::from("/opt/freenect/lib/libfreenect.so.0.7.5")
        );
    }

    #[test]
    fn calibration_revision_hashes_exact_bytes_and_checks_identity() {
        let config = KinectV1Config {
            serial: "A00364911518102A".into(),
            logical_name: "kinect-v1-main".into(),
            calibration_path: PathBuf::from("/home/dawoh/.config/kinect-v1/calibration.json"),
            library_prefix: PathBuf::from("/home/dawoh/.local/opt/libfreenect-0.7.5"),
            max_pair_skew_ms: 20,
        };
        if config.calibration_path.exists() {
            let reference = load_calibration(&config).unwrap();
            assert_eq!(reference.reference.camera_serial, config.serial);
            assert_eq!(reference.reference.revision_sha256.len(), 64);
        }
    }

    fn synthetic_registration() -> FactoryRegistration {
        let mut registration_table = vec![[i32::MAX / 2, 0]; DEPTH_PIXELS];
        registration_table[0] = [10 * 256, 20];
        registration_table[1] = [10 * 256, 20];
        registration_table[2] = [-256, 20];
        registration_table[3] = [WIDTH as i32 * 256, 20];
        FactoryRegistration {
            target_offset: 0,
            depth_to_rgb_shift: vec![0; MAX_METRIC_DEPTH_MM],
            registration_table,
        }
    }

    fn test_stamp(ns: u64, domain: SensorClockDomain) -> SensorStamp {
        SensorStamp {
            source_sequence: 0,
            host_monotonic_ns: ns,
            wall_clock_unix_ms: 1,
            clock_domain: domain,
            device_timestamp_raw: Some(1),
        }
    }

    fn rgb_sample(ns: u64) -> RgbSample {
        RgbSample {
            bytes: Vec::new(),
            stamp: test_stamp(ns, SensorClockDomain::KinectV1Rgb),
        }
    }

    fn depth_sample(ns: u64) -> DepthSample {
        DepthSample {
            depth_mm: Vec::new(),
            owner_index: Vec::new(),
            stamp: test_stamp(ns, SensorClockDomain::KinectV1Depth),
            valid_ratio: 0.0,
        }
    }

    #[test]
    fn factory_registration_keeps_nearest_owner_and_marks_unknown_elsewhere() {
        let mut native = vec![0_u16; DEPTH_PIXELS];
        native[0] = 1_500;
        native[1] = 900;
        let output = synthetic_registration().register(&native);
        let target = 20 * WIDTH as usize + 10;
        assert_eq!(output.depth_mm[target], 900);
        assert_eq!(output.owner_index[target], 1);
        assert_eq!(output.collisions, 1);
        assert_eq!(output.owner_index[target + 1], NO_OWNER);
    }

    #[test]
    fn factory_registration_rejects_invalid_and_out_of_bounds_samples() {
        let mut native = vec![0_u16; DEPTH_PIXELS];
        native[0] = MAX_METRIC_DEPTH_MM as u16;
        native[2] = 1_000;
        native[3] = 1_000;
        let output = synthetic_registration().register(&native);
        assert_eq!(output.out_of_bounds, 2);
        assert!(output.depth_mm.iter().all(|&depth| depth == 0));
        assert!(output.owner_index.iter().all(|&owner| owner == NO_OWNER));
    }

    #[test]
    fn strict_pairing_drops_expired_rgb_instead_of_publishing_rgb_only() {
        let mut state = CallbackState::default();
        state.rgb.push_back(rgb_sample(10));
        state.rgb.push_back(rgb_sample(100));
        state.depth.push_back(depth_sample(105));
        let (rgb, _, skew) = take_strict_pair(&mut state, 20, 200).unwrap();
        assert_eq!(rgb.stamp.host_monotonic_ns, 100);
        assert_eq!(skew, 5);
        assert_eq!(state.health.rgb_only, 1);
        assert_eq!(state.health.paired, 1);
        assert!(state.rgb.is_empty());
        assert!(state.depth.is_empty());
    }

    #[test]
    fn strict_pairing_waits_for_possible_future_depth() {
        let mut state = CallbackState::default();
        state.rgb.push_back(rgb_sample(100));
        state.depth.push_back(depth_sample(90));
        assert!(take_strict_pair(&mut state, 20, 110).is_none());
        assert_eq!(state.rgb.len(), 1);
        assert_eq!(state.depth.len(), 1);
        assert_eq!(state.health.rgb_only, 0);
    }
}
