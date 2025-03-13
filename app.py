import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model import *
from src.data_loader import *
from src.utils import *

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import io

config = load_config("config.yaml")

app = Flask(__name__)

def load_model(model_path):
    model = ResNet18(num_classes=10)  # Update num_classes if your dataset has a different number of classes
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model1 = load_model('models/model1.pth')
model2 = load_model('models/model2.pth')
model3 = load_model('models/model3.pth')

mean, std = compute_mean_std(config["input_dir"])

transform = transforms.Compose([
    transforms.Resize((config["height"], config["width"])),  # Match the input size of the model
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),  # Pre-trained ResNet normalization
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image and apply transformations
        img = Image.open(io.BytesIO(file.read()))
        img = transform(img).unsqueeze(0)

        # Get predictions from all models
        with torch.no_grad():
            output1 = model1(img)
            output2 = model2(img)
            output3 = model3(img)

        # Convert logits to probabilities
        prob_output1 = F.softmax(output1, dim=1)
        prob_output2 = F.softmax(output2, dim=1)
        prob_output3 = F.softmax(output3, dim=1)

        # Average the probabilities
        avg_probabilities = (prob_output1 + prob_output2 + prob_output3) / 3

        # Get the final prediction (class with highest probability)
        final_prediction = avg_probabilities.argmax(dim=1).item()

        return jsonify({'predicted_class': final_prediction, 'probabilities': avg_probabilities.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
