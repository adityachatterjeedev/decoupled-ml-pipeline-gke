from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

app = Flask(__name__)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the model with a wait-loop
device = torch.device("cpu")
model = Net().to(device)
MODEL_PATH = "/mnt/model/mnist_cnn.pth"

def load_model():
    while not os.path.exists(MODEL_PATH):
        print("Waiting for model file to be created by training job...")
        time.sleep(10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")

load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        # Convert list to tensor and reshape for CNN: (Batch, Channel, H, W)
        img_tensor = torch.tensor(data).float().view(1, 1, 28, 28).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.argmax(dim=1, keepdim=True).item()
            
        return jsonify({'prediction': prediction, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)