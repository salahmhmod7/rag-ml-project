FROM python:3.10-slim

# --- Prevent buffered stdout / stderr and enable utf-8 ---
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV PYTHONPATH=/home/app

# --- System dependencies (minimal, lean) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# --- Create app user ---
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

# --- Optimize layer caching: copy requirements FIRST, then pip install ---
COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy application code ---
COPY --chown=app:app . /home/app/

# --- Switch to non-root user ---
USER app

# --- Expose FastAPI port ---
EXPOSE 8000

# --- Entrypoint ---
# Use host.docker.internal for Ollama connectivity from container to host OS.
# On Docker Desktop: use host.docker.internal to reach localhost (the host machine).
# On Linux/Mac with SSH tunneling: use host IP or adjust --network host.
#
# To connect to Ollama running on the host from inside the container:
#   - Docker Desktop: set OLLAMA_BASE_URL=http://host.docker.internal:11434
#   - Linux/macOS: use --network host flag or host IP gateway
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]