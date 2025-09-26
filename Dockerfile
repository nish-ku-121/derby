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

# Copy only pyproject files first to leverage Docker layer caching for deps
COPY pyproject.toml poetry.lock* ./

# Copy the package directory so Poetry can install the project (previous error
# came from not having the 'derby' directory present during install)
COPY derby ./derby

# Install dependencies AND the root package (harmless for dev; live source
# mounted at runtime will still override installed copy)
RUN poetry install --no-interaction --no-ansi --with dev

# Copy the remainder of the repo (configs, scripts, notebooks, etc.)
COPY . .

# Default command
CMD ["poetry", "run", "python"]
