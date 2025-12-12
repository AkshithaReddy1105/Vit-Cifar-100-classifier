from flask import Flask, request, jsonify, render_template_string
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import io
import json
import os

app = Flask(__name__)

# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

class ModelPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Check which model file exists and load accordingly
        model_loaded = False
        model_name = "pretrained"
        
        if os.path.exists('vit_cifar100.pth'):
            # Try loading different architectures based on your training
            architectures = [
                ('resnet34', 'ResNet34'),
                ('vit_small_patch16_224', 'ViT-Small'),
                ('resnet50', 'ResNet50'),
            ]
            
            for arch, name in architectures:
                try:
                    print(f"Attempting to load {name} model...")
                    self.model = timm.create_model(arch, pretrained=False, num_classes=100)
                    self.model.load_state_dict(torch.load('vit_cifar100.pth', map_location=self.device))
                    print(f"✓ Successfully loaded {name} trained model!")
                    model_name = name
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"  Failed to load as {name}: {str(e)[:50]}...")
                    continue
        
        if not model_loaded:
            print("⚠️  No trained weights found. Using pretrained ResNet34 from ImageNet...")
            self.model = timm.create_model('resnet34', pretrained=True, num_classes=100)
            model_name = "ImageNet-pretrained ResNet34"
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        
        print(f"Model loaded: {self.model_name}")
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        predictions = []
        for i in range(5):
            predictions.append({
                'class': CIFAR100_CLASSES[top5_idx[0][i].item()],
                'probability': float(top5_prob[0][i].item())
            })
        
        return predictions

predictor = ModelPredictor()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vision Transformer - CIFAR-100 Classifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .model-info {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .model-info strong {
            color: #667eea;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9fa;
        }
        .upload-area:hover {
            background: #e9ecef;
            border-color: #5568d3;
        }
        .upload-area p {
            margin: 10px 0;
            color: #666;
        }
        .upload-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }
        #preview {
            max-width: 100%;
            max-height: 400px;
            margin: 20px auto;
            display: none;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 10px;
            transition: transform 0.2s;
            font-weight: 600;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        #results {
            margin-top: 30px;
            display: none;
        }
        #results h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .prediction {
            background: linear-gradient(to right, #f8f9fa, white);
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: transform 0.2s;
        }
        .prediction:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .prediction:nth-child(1) { border-left-color: #667eea; }
        .prediction:nth-child(2) { border-left-color: #764ba2; }
        .prediction:nth-child(3) { border-left-color: #8b5cf6; }
        .prediction:nth-child(4) { border-left-color: #a78bfa; }
        .prediction:nth-child(5) { border-left-color: #c4b5fd; }
        .prediction strong {
            color: #333;
            font-size: 1.1em;
        }
        .probability {
            font-weight: bold;
            color: #667eea;
            font-size: 1.2em;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 CIFAR-100 Image Classifier</h1>
        <p class="subtitle">AI-Powered Image Recognition System</p>
        
        <div class="model-info">
            <strong>🧠 Model:</strong> """ + predictor.model_name + """
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📁</div>
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="loadImage(event)">
            <p><strong>Click to upload image</strong></p>
            <p style="font-size: 0.9em;">or drag and drop</p>
            <p style="font-size: 0.8em; margin-top: 10px;">Supports: JPG, PNG, WebP</p>
        </div>
        
        <img id="preview" />
        <button id="predictBtn" onclick="predict()" style="display: none;">
            🔍 Classify Image
        </button>
        
        <div id="results">
            <h2>📊 Top 5 Predictions:</h2>
            <div id="predictionsList"></div>
        </div>
        
        <div class="footer">
            <p>Powered by Deep Learning • CIFAR-100 Dataset • 100 Classes</p>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        function loadImage(event) {
            selectedFile = event.target.files[0];
            if (!selectedFile) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('preview');
                const predictBtn = document.getElementById('predictBtn');
                
                preview.src = e.target.result;
                preview.style.display = 'block';
                predictBtn.style.display = 'block';
                
                // Hide previous results
                document.getElementById('results').style.display = 'none';
            };
            reader.readAsDataURL(selectedFile);
        }
        
        async function predict() {
            if (!selectedFile) return;
            
            const predictBtn = document.getElementById('predictBtn');
            const originalText = predictBtn.innerHTML;
            
            predictBtn.innerHTML = '<span class="loading"></span> Analyzing...';
            predictBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Prediction failed');
                }
                
                const data = await response.json();
                displayResults(data.predictions);
            } catch (error) {
                alert('Error: ' + error.message);
                console.error('Prediction error:', error);
            } finally {
                predictBtn.innerHTML = originalText;
                predictBtn.disabled = false;
            }
        }
        
        function displayResults(predictions) {
            const resultsDiv = document.getElementById('results');
            const listDiv = document.getElementById('predictionsList');
            
            listDiv.innerHTML = '';
            
            predictions.forEach((pred, idx) => {
                const div = document.createElement('div');
                div.className = 'prediction';
                div.style.opacity = '0';
                div.innerHTML = `
                    <strong>${idx + 1}. ${pred.class.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong>
                    <span class="probability">${(pred.probability * 100).toFixed(1)}%</span>
                `;
                listDiv.appendChild(div);
                
                // Animate in
                setTimeout(() => {
                    div.style.transition = 'opacity 0.3s';
                    div.style.opacity = '1';
                }, idx * 100);
            });
            
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        
        // Drag and drop
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#5568d3';
            uploadArea.style.background = '#e9ecef';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#667eea';
            uploadArea.style.background = '#f8f9fa';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#667eea';
            uploadArea.style.background = '#f8f9fa';
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                const input = document.getElementById('fileInput');
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                input.files = dataTransfer.files;
                
                loadImage({ target: input });
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        image_bytes = image_file.read()
        predictions = predictor.predict(image_bytes)
        return jsonify({'predictions': predictions})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'device': str(predictor.device),
        'model': predictor.model_name
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Flask Server Starting...")
    print("="*60)
    print(f"Model: {predictor.model_name}")
    print(f"Device: {predictor.device}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=8080, debug=False)