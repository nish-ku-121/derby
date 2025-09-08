# Makefile for Docker + Poetry workflow

# Build Docker image
docker-build:
	docker build -t derby-app .


# Run Poetry lock in the custom Docker image, generating poetry.lock on host
docker-poetry-install: docker-build
	docker run --rm -v $(PWD):/app -w /app derby-app poetry lock


# Run container with live code mounting
docker-shell: docker-build
	docker run -it --rm -v $(PWD):/app derby-app bash
