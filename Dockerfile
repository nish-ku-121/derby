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

# Copy only pyproject.toml and poetry.lock first for caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev by default)
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the code
COPY . .

# Default command
CMD ["poetry", "run", "python"]
