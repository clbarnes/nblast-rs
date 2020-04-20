=========
nblast-rs
=========

Python bindings for nblast rust library.

* Free software: GPL3 license

Building
--------

If wheels are not available for your platform, they can be built locally using `maturin <https://github.com/PyO3/maturin>`_
and an appropriate rust compiler.
Assuming:

* the rust version specified in ``rust-toolchain`` at the root of the repository is installed (see `rustup <https://rustup.rs/>`_)
* the required python version is on the ``PATH`` as ``python``
* maturin is installed (``pip install maturin``)

::

    git clone https://github.com/clbarnes/nblast-rs
    cd nblast-rs/nblast-py
    maturin build --release -i python

will produce a wheel in ``nblast-rs/nblast-py/target/wheels``.
