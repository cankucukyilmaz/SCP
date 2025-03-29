# Use official PyTorch image as base
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY SCP/app.py /app/app.py
COPY SCP/index.html /app/index.html
COPY SCP/src /app/src
COPY SCP/saved_models /app/saved_models
COPY SCP/requirements.txt /app/requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Run the application
CMD ["python", "app.py"]
