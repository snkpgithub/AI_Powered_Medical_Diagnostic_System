# Use the official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     build-essential     libglib2.0-0     libsm6     libxext6     libxrender-dev     && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision fastapi uvicorn pillow

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
