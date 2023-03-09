release-patch:
	cd nblast-rs && cargo release patch
	cd nblast-py && cargo release patch
	cd nblast-js && cargo release patch

release-minor:
	cd nblast-rs && cargo release minor
	cd nblast-py && cargo release minor
	cd nblast-js && cargo release minor

release-major:
	cd nblast-rs && cargo release major
	cd nblast-py && cargo release major
	cd nblast-js && cargo release major
