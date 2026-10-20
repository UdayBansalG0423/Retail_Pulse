# ==============================================================================
# RetailPulse — Production Dockerfile
# Optimized for minimal image layer size, low RAM footprint, and instant startup
# ==============================================================================

# 1. Base Image: Slim Debian Bookworm with Python 3.11 (~130 MB base)
FROM python:3.11-slim-bookworm AS runtime

# 2. Environment Variables for Memory Optimization & Headless Operation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Glibc memory allocator tuning: Return heap memory back to OS aggressively
    MALLOC_TRIM_THRESHOLD_=65536 \
    # Thread pool bounds: Prevents XGBoost / NumPy thread spawning overhead
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    # Streamlit headless defaults
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 3. Install minimal system packages (libgomp1 for OpenMP/XGBoost, curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        dos2unix && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 4. Install Python dependencies using layer cache
COPY requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Create unprivileged application user for container security
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# 6. Copy application source code and pre-trained artifacts
COPY --chown=appuser:appgroup . /app

# Normalize line endings and permissions for entrypoint script
RUN dos2unix /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# 7. Switch to non-root user
USER appuser

# 8. Container Healthcheck on Streamlit internal health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
