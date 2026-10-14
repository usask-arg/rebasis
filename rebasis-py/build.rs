//! Lets plain `cargo build` link the extension module on macOS (`-undefined dynamic_lookup`, so
//! the Python symbols resolve when the interpreter loads it); maturin passes the same flags.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
