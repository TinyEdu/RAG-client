# Base image with Python and CUDA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI (assuming it has a Linux CLI available)
RUN wget -O /usr/local/bin/ollama https://ollama.com/download/cli/ollama-linux && \
    chmod +x /usr/local/bin/ollama

# Install Python dependencies
RUN pip3 install --upgrade pip

# Install model dependencies (if any additional ones are required)
# e.g., RUN pip3 install torch transformers

# Pull the model (assuming you have a command to pull the model via Ollama CLI)
RUN ollama pull llama3.1-8b

# Expose any necessary ports (adjust as necessary)
EXPOSE 8080

# Set the entrypoint to the Ollama command that runs the model server or inference
CMD ["ollama", "run", "llama3.1-8b"]
