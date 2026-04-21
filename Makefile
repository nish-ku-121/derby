# ----------------------------
# Makefile for Docker + Poetry workflow (portable)
# ----------------------------

# Default Python image (should match Dockerfile)
PYTHON_IMAGE ?= python:3.10-slim

# Host port to expose Jupyter on (container still listens on 8888)
JUPYTER_PORT ?= 8888

# Default pytest target. Can be overridden with:
#   make test TEST=derby/tests/test_utils.py
#   make test TEST=derby/tests/test_utils.py::TestUtils::test_kth_largest_1
TEST ?= derby/tests

# Detect platform
UNAME_S := $(shell uname -s)

# Set correct current directory path for Docker
ifeq ($(OS),Windows_NT)
    ifeq ($(UNAME_S),MINGW64_NT-10.0)
        # Git Bash on Windows
        PWD_PATH := $(shell pwd -W | sed 's#\\#/#g')
    else
        # CMD or PowerShell on Windows
        PWD_PATH := $(PWD)
    endif
else
    # Linux/macOS
    PWD_PATH := $(PWD)
endif

# ----------------------------
# Targets
# ----------------------------

.PHONY: lockfile build shell run jupyter
.PHONY: test

# Generate poetry.lock inside Docker only when pyproject.toml changes.
# Run poetry lock inside a throwaway container; always 'touch' the lockfile so
# its timestamp is newer than pyproject.toml even if no changes were needed.
# This prevents make from rerunning this step on every invocation.
poetry.lock: pyproject.toml
	docker run --rm -v "$(PWD_PATH):/app" $(PYTHON_IMAGE) bash -c "cd /app && pip install poetry==2.1.4 && poetry lock && touch poetry.lock"

# Convenience target to force (re)locking regardless of timestamps
lockfile:
	$(MAKE) -B poetry.lock

# Build Docker image (ensure lockfile is up to date first)
build: poetry.lock
	docker build -t derby-app .

# Run container with live code mounting
shell: build
	docker run -it --rm -v "$(PWD_PATH):/app" derby-app bash

# Generalized Docker run target
run:
	docker run --rm -v "$(PWD_PATH):/app" derby-app bash -lc "cd /app && poetry run $(ARGS)"

# Run pytest inside the Docker image (simple interface).
# Usage:
#   make test                              # run entire suite (quiet)
#   make test TEST=derby/tests/test_utils.py
#   make test TEST=derby/tests/test_utils.py::TestClass::test_method
# If you need custom flags occasionally: make run ARGS="pytest -vv -k pattern"
test: build
	docker run --rm -v "$(PWD_PATH):/app" derby-app bash -lc "cd /app && poetry run pytest '$(TEST)' -q"

# Run Jupyter Lab inside Docker with Poetry env (mounts repo and exposes port)
jupyter: build
	docker run --rm -it \
		-p $(JUPYTER_PORT):8888 \
		-v "$(PWD_PATH):/app" \
		derby-app bash -lc "cd /app && \
		poetry run python -m ipykernel install --user --name derby-poetry --display-name 'Python (derby)' || true && \
		poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''"
