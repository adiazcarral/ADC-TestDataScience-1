# Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY src/ src/
COPY models/ models/  # if you save your model files here

# Set environment variables
ENV PYTHONPATH=/app

# Run the API
CMD ["uvicorn", "src.adc_testdatascience_1.app:app", "--host", "0.0.0.0", "--port", "8000"]

