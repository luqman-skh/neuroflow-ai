FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered streaming logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED="False" \
    HF_HUB_DISABLE_TELEMETRY="1" \
    HF_HUB_OFFLINE="1" \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    SQLITE_DB_PATH="/app/data/neuroflow.db"

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project specification and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code and default datasets
COPY src/ ./src/
COPY data/ ./data/

# Ensure data directory exists with write permissions for SQLite persistence
RUN mkdir -p /app/data

# Expose Gradio UI port
EXPOSE 7860

# Health check to ensure Gradio web server is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default execution command
CMD ["python", "-m", "neuroflow.ui.app"]
