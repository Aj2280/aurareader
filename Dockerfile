FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install core runtime dependencies directly to avoid conflict at build time
RUN pip install --no-cache-dir \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    "transformers>=4.40.0,<5.0.0" \
    "gradio>=6.0.0" \
    "fastapi" \
    "requests"

# Copy the rest of the application code
COPY . .

# Expose the Gradio port
EXPOSE 7860

# Set environment variable for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Start application
CMD ["python", "app.py"]
