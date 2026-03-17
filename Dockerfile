# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim
 
# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app
 
# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
 
# ── Copy project files ────────────────────────────────────────────────────────
COPY app/ ./app/
COPY run.py .
 
# ── ChromaDB storage directory ────────────────────────────────────────────────
RUN mkdir -p /app/chroma_db
 
# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000
 
# ── Start command ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
