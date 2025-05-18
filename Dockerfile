# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    cmake \
    g++ \
    wget \
    unzip \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git lfs
RUN git lfs install

# Copy project files
COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/
COPY src/ ./src/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir uv && \
    uv venv && \
    uv pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install -e .

# Create necessary directories and download resources
RUN mkdir -p logs cache && \
    wget https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h/resolve/main/vi_lm_4grams.bin.zip && \
    unzip -q vi_lm_4grams.bin.zip -d ./cache/ && \
    git clone https://huggingface.co/khanhld/chunkformer-large-vie ./packages/former/chunkformer-large-vie

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "fastapi", "run", "src/voice_recog/app.py"] 