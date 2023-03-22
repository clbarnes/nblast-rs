.PHONY: pkg
pkg-debug: clean-pkg
	wasm-pack build --target no-modules

.PHONY:
pkg: clean-pkg
	wasm-pack build --release --target no-modules

.PHONY: clean-pkg
clean-pkg:
	rm -rf pkg/
