# Build stage - for compiling dependencies
FROM python:3.11-slim-bookworm as builder

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install TA-Lib (technical analysis library)
RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar xz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf ta-lib

# --- Runtime stage ---
FROM python:3.11-slim-bookworm

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
WORKDIR /app
COPY . .

# Security hardening
RUN chmod 755 /app && \
    find . -type f -exec chmod 644 {} \; && \
    chmod 755 docker-entrypoint.sh && \
    adduser --disabled-password --gecos "" trader && \
    chown -R trader:trader /app

# Switch to non-root user
USER trader

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PROMETHEUS_MULTIPROC_DIR=/tmp \
    UVICORN_WORKERS=4 \
    TZ=UTC

#

# Entrypoint with graceful shutdown handling
ENTRYPOINT ["tini", "--", "/app/docker-entrypoint.sh"]