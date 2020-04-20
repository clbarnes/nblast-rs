#!/bin/sh

set -ex

### Setup Rust toolchain #######################################################

RUST_VERSION=${RUST_VERSION:-$TRAVIS_RUST_VERSION}

curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=$RUST_VERSION
export PATH=$PATH:$HOME/.cargo/bin
