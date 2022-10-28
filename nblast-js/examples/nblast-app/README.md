# nblast-app

A simple example to show NBLAST running in the browser.

## Usage

1. Run `make pkg` in the ancestory `nblast-js` directory.
2. Run `npm install` in this directory.
3. Run an HTTP server in this directory (e.g. [`simple-http-server`](https://crates.io/crates/simple-http-server)).
4. Navigate to the `index.html` (e.g. http://localhost:8000/index.html).
5. Select a score matrix (e.g. `data/smat_fcwb.csv` the parent workspace) and two point clouds (e.g. from `data/points/*.csv`).
6. Press the button!

## Known problems

It's *extremely* slow.
