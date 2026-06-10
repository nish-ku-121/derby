# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Set HOME and PATH first
ENV HOME=/root
ENV PATH="$HOME/.local/bin:$PATH"

# System dependencies for scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=2.1.4
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Set workdir to /app
WORKDIR /app

##
# Optimize caching: install dependencies based only on lockfile first.
# This way, code changes in the repo don't bust the heavy dependency layer.
##

# Copy only pyproject + lock to build dependency layer
COPY pyproject.toml poetry.lock* ./

# Install all dependencies including dev tools (Jupyter, etc.)
# but skip installing the project itself to keep this layer stable
RUN poetry install --no-interaction --no-ansi --with dev --no-root

# Now copy project source and minimal required files
COPY derby ./derby

# Install the local package in editable mode so runtime mounts override if needed
RUN poetry install --no-interaction --no-ansi --only-root

# Copy the remainder (configs, scripts); .dockerignore keeps context slim
COPY . .

# Default command
CMD ["poetry", "run", "python"]
