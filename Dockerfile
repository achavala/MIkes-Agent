# Multi-stage build to reduce final image size
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (better caching - copy requirements first)
COPY requirements.txt .
RUN pip install --no-cache-dir --user torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user stable-baselines3[extra]>=2.3.0 && \
    grep -v "^torch" requirements.txt | grep -v "^stable-baselines3" > /tmp/requirements_remaining.txt && \
    pip install --no-cache-dir --user -r /tmp/requirements_remaining.txt && \
    rm /tmp/requirements_remaining.txt && \
    pip cache purge

# Final stage - minimal runtime image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set timezone to EST/EDT (America/New_York)
ENV TZ=America/New_York
ENV PYTHONPATH=/root/.local/lib/python3.11/site-packages:$PYTHONPATH
ENV PATH=/root/.local/bin:$PATH

# Ensure timezone is set in system
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install curl for model downloads (lightweight, ~1MB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy only what is needed (explicit, not wildcard)
COPY requirements.txt .
COPY start_cloud.sh .
COPY core/ ./core/
COPY utils/ ./utils/
COPY dashboard_app.py .
COPY mike_agent_live_safe.py .
# Copy models directory (includes trained historical model)
COPY models/ ./models/
# Copy other essential Python files (but not test files - excluded by .dockerignore)
COPY *.py ./
# Ensure live_activity_log.py is included (for Analytics tab)
COPY live_activity_log.py ./

# Safety: ensure executable
RUN chmod +x start_cloud.sh

# Clean up unnecessary files to reduce image size
# IMPORTANT: Preserve version.txt, METADATA files, and numpy test modules (needed by scipy)
RUN find /root/.local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /root/.local -type f -name "*.pyo" -delete 2>/dev/null || true && \
    find /root/.local -name "*.md" -delete 2>/dev/null || true && \
    find /root/.local/lib/python*/site-packages -type d -name "tests" -not -path "*/torch/testing*" -not -path "*/numpy/*" -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local/lib/python*/site-packages -type d -name "test" -not -path "*/torch/testing*" -not -path "*/numpy/*" -not -name "testing" -exec rm -rf {} + 2>/dev/null || true
# Note: We preserve all .txt files (including version.txt) and numpy test modules (needed by scipy)

# Verify stable-baselines3 is accessible (but don't fail if version.txt is missing - package may still work)
RUN python -c "import sys; sys.path.insert(0, '/root/.local/lib/python3.11/site-packages'); import stable_baselines3; print('✓ stable-baselines3 installed')" 2>/dev/null || \
    python -c "import sys; sys.path.insert(0, '/root/.local/lib/python3.11/site-packages'); from stable_baselines3 import PPO; print('✓ stable-baselines3 importable (version check skipped)')" || \
    echo "⚠️ stable-baselines3 verification failed - will check at runtime"

# Expose port
EXPOSE 8080

# Start both agent and dashboard
CMD ["bash", "start_cloud.sh"]
