from src.data_loader import *
from src.model import *
from src.utils import *

from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os

config = load_config("config.yaml")

# Initialize Flask app
app = Flask(__name__)

# Define image transformations
mean = [0.539342462170622, 0.4856492361751947, 0.43324973198145483]
std = [0.2270848481387292, 0.22017275884417153, 0.22362268174675773]

transform = transforms.Compose([
    transforms.Resize((config["height"], config["width"])),  # Resize to model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=mean, std=std),  # Normalize using dataset statistics
])

# Load the model
def load_model(model_path, device):
    """
    Load a pre-trained model from the specified path.
    """
    print(f"Loading model from: {model_path}")

    # Ensure the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = ResNet18().to(device)  # Initialize the model
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
    model.eval()  # Set the model to evaluation mode

    print("Model loaded successfully.")
    return model

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model_path = "saved_models/resnet18_20250320_180043.pth"
try:
    model = load_model(model_path, device)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image uploads and return predictions.
    """
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if a file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Apply transformations and prepare the image for the model
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Get predictions from the model
        with torch.no_grad():
            output = model(img)

        # Convert logits to probabilities
        probabilities = F.softmax(output, dim=1).squeeze()  # Remove batch dimension

        # Get the top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)  # Get top 3 probabilities and their indices
        top_probs = top_probs.tolist()  # Convert to list
        top_indices = top_indices.tolist()  # Convert to list

        # Map indices to labels
        top_labels = [config["classes"][i] for i in top_indices]

        # Combine probabilities and labels
        top_predictions = [
            {"label": label, "probability": round(prob * 100, 2)}
            for label, prob in zip(top_labels, top_probs)
        ]

        # Return the result as JSON
        return jsonify({
            'predictions': top_predictions
        })

    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)