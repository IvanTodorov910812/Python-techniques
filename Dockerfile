# Multi-stage build for optimal image size
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install wheels
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime (minimal image)
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CACHE_DIR=/app/cache

# 🔥 CRITICAL: Create cache directory for persistent storage
RUN mkdir -p /app/cache

# Pre-download spaCy model to avoid startup timeout
RUN python -m spacy download en_core_web_sm

# Pre-compute ESCO embeddings to avoid slow startup on Render free tier
# This is critical: free tier has limited CPU/RAM, so compute embeddings at build time
RUN python precompute_embeddings.py || echo "Note: Embeddings will be computed on first run"

# Make model download script executable
RUN chmod +x download_models.sh

# VOLUME for persistent cache (will be mounted by Render)
VOLUME ["/app/cache"]

# Health check (extended timeout for model loading)
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/_stcore/health', timeout=10)" || exit 1

# Default command (Streamlit with optimized settings)
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8000", \
     "--server.address=0.0.0.0", \
     "--client.showErrorDetails=true", \
     "--logger.level=info"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDER DEPLOYMENT CONFIGURATION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# In render.yaml or Render web settings:
#
# services:
#   - type: web
#     name: cv-matcher
#     env: docker
#     dockerfilePath: Dockerfile
#     envVars:
#       - key: CACHE_DIR
#         value: /app/cache
#     disk:
#       - name: embedding_cache
#         mountPath: /app/cache
#         sizeGB: 2
#
# This ensures cache files (esco_embeddings.pkl, embedding_cache.pkl)
# persist across container restarts and deployments!
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Optional: For GPU support on Render (if using CUDA)
# Replace base image: FROM nvidia/cuda:12.1-runtime-ubuntu22.04
# Then add: RUN apt-get install -y python3.11 python3-pip
