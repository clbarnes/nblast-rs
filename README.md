# nblast-rs
[![Build Status](https://travis-ci.com/clbarnes/nblast-rs.svg?branch=master)](https://travis-ci.com/clbarnes/nblast-rs)

Rust implementation of NBLAST, originally published [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4961245/) and implemented [here](https://github.com/natverse/nat.nblast/), with python bindings.

This repository acts as a cargo workspace, and contains two crates:

- [nblast-rs](./nblast-rs), a cargo project representing the `nblast` rust crate
- [nblast-py](./nblast-py), a maturin project representing the `pynblast` python project (which binds to rust)

To build and test the python project in `nblast-py` (which should not be deployed as a crate), use `maturin develop` and `pytest` in the subdirectory.

See [crates.io](https://crates.io/crates/nblast) and [docs.rs](https://docs.rs/nblast) for the rust project,
or [PyPI](https://pypi.org/project/pynblast) for the python package.

## Development

Use [rustup](https://rustup.rs/) to install the rust compiler specified in [rust-toolchain](./rust-toolchain).

### nblast-rs

In the `nblast-rs` subdirectory

- Use `cargo build` / `cargo test` / `cargo bench` etc.

### nblast-py

In the `nblast-py` subdirectory

- Install the development dependencies with `pip install -r requirements.txt`
- Compile the rust component and install locally with `maturin develop` (add `--release` for an optimised build)

## Notes

Installing the python wheel from a source distribution (sdist) is currently not supported and will likely fail.
See [here](./nblast-py/README.rst) for details on how to build locally if wheels are not available for your platform.
