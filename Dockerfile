# Use an official PyTorch image as a base
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app, saved_models, source code, configuration, and input directory
COPY app.py .
COPY saved_models/ ./saved_models
COPY src/ ./src
COPY config.yaml .

# Expose the port the app will run on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]