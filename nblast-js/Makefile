.PHONY: pkg
pkg-debug:
	wasm-pack build --target web

.PHONY:
pkg:
	wasm-pack build --release --target web

.PHONY: clean-pkg
clean-pkg:
	rm -rf pkg/
