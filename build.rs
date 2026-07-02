fn main() {
    // Link libv4l2 from system (available as .so.0 without -dev package)
    println!("cargo:rustc-link-search=native={}/.local/lib", std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rustc-link-lib=v4l2");
}
