# nblast-rs

WIP

Fast, safe implementation of NBLAST, originally published [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4961245/) and implemented [here](https://github.com/natverse/nat.nblast/), with python bindings.

This repository acts as a cargo workspace, and contains two crates:

- [nblast-rs](./nblast-rs), a cargo project representing the `nblast` rust crate
- [nblast-py](./nblast-py), a maturin project representing the `pynblast` python project (which binds to rust)

In the root directory, `cargo build` / `cargo test` will only build/test the crate in `nblast-rs`.
To build and test the python project in `nblast-py` (which should not be deployed as a crate), use `maturin develop` and `pytest` in the subdirectory.

See [crates.io](https://crates.io/crates/rand) and [docs.rs](https://docs.rs/nblast) for the rust project,
or [PyPI](https://pypi.org/project/pynblast) and [ReadTheDocs]() for the released projects,
