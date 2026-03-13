FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support (for GTX 1070)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt "prefect>=2.14,<3"

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed outputs/plots

# Set Python path
ENV PYTHONPATH=/app
