# Use an official PyTorch base image (CUDA included)
FROM python:3.11-slim

#set working dir inside the container
WORKDIR /app 

# Install system dependencies in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (copy deps first from host to container)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/

# docker listens to these ports
EXPOSE 8888 

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]