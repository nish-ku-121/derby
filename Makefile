# ----------------------------
# Makefile for Docker + Poetry workflow (portable)
# ----------------------------

# Default Python image (should match Dockerfile)
PYTHON_IMAGE ?= python:3.10-slim

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

.PHONY: docker-lockfile docker-build docker-shell docker-run docker-jupyter

# Generate poetry.lock inside Docker only when pyproject.toml changes
poetry.lock: pyproject.toml
	docker run --rm -v "$(PWD_PATH):/app" $(PYTHON_IMAGE) bash -c "cd /app && pip install poetry==2.1.4 && poetry lock"

# Convenience target to force (re)locking regardless of timestamps
docker-lockfile:
	$(MAKE) -B poetry.lock

# Build Docker image (ensure lockfile is up to date first)
docker-build: poetry.lock
	docker build -t derby-app .

# Run container with live code mounting
docker-shell: docker-build
	docker run -it --rm -v "$(PWD_PATH):/app" derby-app bash

# Generalized Docker run target
docker-run:
	docker run --rm -v "$(PWD_PATH):/app" derby-app $(ARGS)

# Run Jupyter Lab inside Docker with Poetry env (mounts repo and exposes port)
docker-jupyter: docker-build
	docker run --rm -it \
		-p 8888:8888 \
		-v "$(PWD_PATH):/app" \
		derby-app bash -lc "cd /app && \
		poetry run python -m ipykernel install --user --name derby-poetry --display-name 'Python (derby)' || true && \
		poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''"
