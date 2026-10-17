# Production Dockerfile for card-vision REST microservice
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies for OpenCV and wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY card_vision/ ./card_vision/

# Build wheel distribution
RUN pip install --no-cache-dir hatchling \
    && python -m hatchling build

# Final lightweight runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies for OpenCV and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r cardvision && useradd -r -g cardvision -s /bin/false -d /app cardvision

# Copy wheel from builder and install with server dependencies
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl "card-vision[server]" \
    && rm -rf /tmp/*.whl

USER cardvision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["card-vision", "serve", "--host", "0.0.0.0", "--port", "8000"]
