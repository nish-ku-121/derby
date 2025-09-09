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

.PHONY: docker-lockfile docker-build docker-shell docker-run

# Generate poetry.lock inside Docker (cross-platform)
docker-lockfile:
	docker run --rm -v "$(PWD_PATH):/app" $(PYTHON_IMAGE) bash -c "cd /app && pip install poetry==2.1.4 && poetry lock"

# Build Docker image
docker-build:
	docker build -t derby-app .

# Run container with live code mounting
docker-shell: docker-build
	docker run -it --rm -v "$(PWD_PATH):/app" derby-app bash

# Generalized Docker run target
docker-run:
	docker run --rm -v "$(PWD_PATH):/app" derby-app $(ARGS)
