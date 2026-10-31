.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest

.PHONY: test-latest
test-latest:
	env -u UV_FROZEN uv run \
		--no-cache --isolated --no-project \
		--with . --with pytest --with absl-py \
		--with mujoco-menagerie==2026.9.4 --upgrade pytest

.PHONY: test-all
test-all: check test

.PHONY: coverage
coverage:
	uv run coverage run --source=src -m pytest .
	uv run coverage report

.PHONY: doc
doc:
	rm -rf docs/_build
	uv run --group doc sphinx-build docs docs/_build -W

.PHONY: doc-live
doc-live:
	rm -rf docs/_build
	uv run --group doc sphinx-autobuild docs docs/_build --watch src/mink

.PHONY: build
build:
	rm -rf dist/
	uv build
	uv export --frozen --no-dev --no-emit-project --no-hashes --no-header --no-annotate --output-file dist/runtime-requirements.txt
	uv run --no-cache --isolated --no-project --with-requirements dist/runtime-requirements.txt --with pytest --with absl-py --with mujoco-menagerie==2026.9.4 --with dist/*.whl pytest tests/
	uv run --no-cache --isolated --no-project --with-requirements dist/runtime-requirements.txt --with pytest --with absl-py --with mujoco-menagerie==2026.9.4 --with dist/*.tar.gz pytest tests/
	@ls -lh dist/*.whl | awk '{print "Wheel size: " $$5}'
	@echo "Build and test successful"
