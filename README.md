# SPX_GEX
Derived from a project developed during the Applied Data Science course led by Dr. Ami Gates at Oregon State University. 
Ref: https://gatesboltonanalytics.com/?page_id=136

Some background on GEX.
What is GEX: https://support.spotgamma.com/hc/en-us/articles/15214161607827-GEX-Gamma-Exposure-Explained-What-It-Is-and-How-SpotGamma-Uses-It
GEX formula: https://flashalpha.com/articles/how-to-calculate-gamma-exposure-gex-formula-worked-example

# Adding Rust back-end and framework: https://doc.rust-lang.org/book/title-page.html
Falling in love with beautiful Rust! 
Turning my project into a hybrid of Python and Rust (PyO3) with Maturin bindings... 

Install rust and build tools in brand new shell terminal (not in project IDE).
Deactivate conda or other python environments.
`conda deactivate`

Install rust
`curl --proto '=https' --tlsv1.2 -sSf https://rustup.rs | sh`

Update/install C compiler (gcc)
`sudo apt update && sudo apt install build-essential -y`

In project directory, init rust project
`cargo init --lib`  : this creates Cargo.toml and /src/lib.rs files in project dir.

Edit Cargo.toml to lower case project name (pick any name!) and add dependencies:  https://doc.rust-lang.org/cargo/reference/manifest.html
`[package]
name = "spx_gex_engine"
version = "0.1.0"
edition = "2024"

[lib]
name = "spx_gex_engine"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
# Tokio is the async runtime engine needed for network operations
tokio = { version = "1", features = ["full"] }
# Reqwest handles making the HTTP requests asynchronously
reqwest = { version = "0.12", features = ["json"] }
# Serde parses the incoming API JSON data into type-safe Rust structures
serde = { version = "1.0", features = ["derive"] }
`

Write test rust code in src/lib.rs 
and run 
`cargo run`




