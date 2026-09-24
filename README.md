# SPX_GEX
Derived from a project developed during the Applied Data Science course led by Dr. Ami Gates at Oregon State University. 
Ref: https://gatesboltonanalytics.com/?page_id=136

Some background on GEX.

What is GEX: https://support.spotgamma.com/hc/en-us/articles/15214161607827-GEX-Gamma-Exposure-Explained-What-It-Is-and-How-SpotGamma-Uses-It

GEX formula: https://flashalpha.com/articles/how-to-calculate-gamma-exposure-gex-formula-worked-example

### Adding Rust back-end and framework: https://doc.rust-lang.org/book/title-page.html
Falling in love with beautiful Rust! \
Turning my project into a hybrid of Python and Rust (PyO3) with Maturin bindings...

First steps with Rust:\
Deactivate conda or other python environments.\
Install rust and build tools in brand new shell terminal (not in project IDE).\
`curl --proto '=https' --tlsv1.2 -sSf https://rustup.rs | sh`

Update/install C compiler (gcc)
`sudo apt update && sudo apt install build-essential -y`

In project directory, init rust project
`cargo init --lib`  : this creates Cargo.toml and /src/lib.rs files in project dir.

Edit Cargo.toml: add project name and dependencies:  https://doc.rust-lang.org/cargo/reference/manifest.html
<img width="620" height="308" alt="image" src="https://github.com/user-attachments/assets/512e0428-019a-4ff7-a638-02889f39af1a" />

Write test rust code in src/lib.rs 
and run 
`cargo run`




