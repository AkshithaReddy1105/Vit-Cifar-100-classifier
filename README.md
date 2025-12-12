# Vision Transformer CIFAR-100 Classifier

An AI-powered image classification web application that uses deep learning to classify images into 100 different categories from the CIFAR-100 dataset. The application features a modern web interface and can be deployed to Google Cloud Platform with a single command.

## Features

- **Multiple Model Support**: ResNet34, ResNet50, and Vision Transformer (ViT) architectures
- **High Accuracy**: Achieves 82.84% test accuracy on CIFAR-100 dataset
- **Modern Web Interface**: Clean, responsive UI with drag-and-drop image upload
- **Real-time Predictions**: Fast inference with top-5 predictions and confidence scores
- **Docker Support**: Containerized application for easy deployment
- **Cloud-Ready**: One-command deployment to Google Cloud Run
- **Health Monitoring**: Built-in health check endpoint for monitoring

## Demo

The application provides:
- Drag-and-drop image upload interface
- Real-time classification with confidence scores
- Top 5 predictions for each image
- Support for 100 different object categories

## CIFAR-100 Classes

The model can classify images into 100 categories including:
- **Animals**: apple, aquarium_fish, bear, beaver, bee, butterfly, camel, cattle, chimpanzee, and more
- **Vehicles**: bicycle, bus, motorcycle, pickup_truck, train, tractor, and more
- **Plants**: maple_tree, oak_tree, orchid, poppy, rose, sunflower, tulip, and more
- **Objects**: bed, bottle, chair, clock, keyboard, lamp, telephone, and more

## Model Performance

Based on training results with ResNet architecture:
- **Best Test Accuracy**: 82.84%
- **Training Time**: ~24 minutes (10 epochs)
- **Final Training Accuracy**: 90.92%
- **Model Architecture**: ResNet with transfer learning

## Requirements

### Local Development
```
Python 3.10+
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.2
flask>=2.3.2
Pillow>=10.0.0
gunicorn>=21.2.0
```

### Deployment
- Docker
- Google Cloud SDK (for GCP deployment)
- GCP Project with billing enabled

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AkshithaReddy1105/Vit-Cifar-100-classifier.git
cd Vit-Cifar-100-classifier
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training Your Own Model

Train a new model on CIFAR-100 dataset:

```bash
python train_vit.py
```

The training script will:
1. Download the CIFAR-100 dataset automatically
2. Prompt you to choose a model architecture (ResNet34, ResNet50, or ViT-Small)
3. Ask for the number of training epochs
4. Train the model and save the best weights as `vit_cifar100.pth`
5. Generate training metrics in `results.json`

**Model Options:**
- **ResNet34** (Recommended): Fast training, good accuracy (~83%)
- **ResNet50**: More parameters, potentially higher accuracy
- **ViT-Small**: Vision Transformer, requires more training time

### Running Locally

#### Option 1: Flask Development Server
```bash
python app.py
```
Then open http://localhost:8080 in your browser.

#### Option 2: Docker
```bash
# Build the image
docker build -t vit-classifier .

# Run the container
docker run -p 8080:8080 vit-classifier
```

### Testing the Deployment

Test the application with automated tests:

```bash
# Test local deployment
python test_deployment.py http://localhost:8080

# Test remote deployment
python test_deployment.py https://your-service-url.run.app
```

## Deployment to Google Cloud Platform

### Prerequisites
1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Create a GCP project with billing enabled
3. Ensure Docker is installed

### One-Command Deployment

```bash
./deploy.sh
```

The deployment script will:
1. Prompt for your GCP project ID and region
2. Enable required Google Cloud APIs
3. Build and test the Docker image
4. Push the image to Google Container Registry
5. Deploy to Cloud Run
6. Provide the live URL

### Manual Deployment

```bash
# Set your project ID
export PROJECT_ID=your-project-id
export REGION=us-central1

# Build and tag image
docker build -t gcr.io/$PROJECT_ID/vit-classifier .

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/vit-classifier

# Deploy to Cloud Run
gcloud run deploy vit-classifier \
  --image gcr.io/$PROJECT_ID/vit-classifier \
  --platform managed \
  --region $REGION \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated \
  --port 8080
```

## API Endpoints

### GET /
Web interface for image classification

### POST /predict
Classify an uploaded image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "predictions": [
    {
      "class": "apple",
      "probability": 0.8542
    },
    {
      "class": "orange",
      "probability": 0.0823
    }
  ]
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "model": "ResNet34"
}
```

## Project Structure

```
.
├── app.py                  # Flask web application
├── train_vit.py           # Model training script
├── test_deployment.py     # Deployment testing script
├── deploy.sh              # GCP deployment automation script
├── Dockerfile             # Docker container configuration
├── requirements.txt       # Python dependencies
├── vit_cifar100.pth      # Trained model weights (82.84 MB)
├── results.json          # Training metrics and results
└── data/                 # CIFAR-100 dataset (auto-downloaded)
```

## Model Architecture

The application supports three architectures via the `timm` library:

1. **ResNet34**:
   - Residual Network with 34 layers
   - Pretrained on ImageNet, fine-tuned on CIFAR-100
   - Fast inference, good accuracy

2. **ResNet50**:
   - Deeper ResNet variant with 50 layers
   - More parameters for potentially better accuracy

3. **ViT-Small**:
   - Vision Transformer architecture
   - Patch size: 16x16
   - Input size: 224x224

## Training Details

- **Dataset**: CIFAR-100 (50,000 training + 10,000 test images)
- **Image Size**: 224x224 (upscaled from 32x32)
- **Batch Size**: 128
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.05)
- **Scheduler**: Cosine Annealing
- **Data Augmentation**:
  - Random horizontal flip
  - Random crop with padding
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation)
- **Normalization**: CIFAR-100 mean and std values

## Performance Optimization

The application includes several optimizations:
- **Transfer Learning**: Uses ImageNet pretrained weights
- **Mixed Precision**: GPU training with automatic mixed precision
- **Data Loading**: Multi-threaded data loading with pin_memory
- **Model Caching**: Trained weights saved and reused
- **Production Server**: Gunicorn with multiple workers

## Troubleshooting

### Model Loading Issues
If the app fails to load the trained model, it will automatically fall back to ImageNet pretrained weights. Check that `vit_cifar100.pth` exists and matches the architecture.

### Memory Issues
If you encounter OOM errors during training:
- Reduce batch size in `train_vit.py`
- Use ResNet34 instead of ResNet50 or ViT
- Enable gradient checkpointing

### Cloud Deployment Issues
Check the logs:
```bash
gcloud run services logs tail vit-classifier --region us-central1
```

## Future Improvements

- [ ] Add support for custom image uploads beyond CIFAR-100 classes
- [ ] Implement model ensemble for higher accuracy
- [ ] Add Grad-CAM visualization for interpretability
- [ ] Support for additional datasets
- [ ] Mobile app deployment
- [ ] API rate limiting and authentication
- [ ] Model versioning and A/B testing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- CIFAR-100 dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- timm library by Ross Wightman for pretrained models
- PyTorch team for the deep learning framework
- Google Cloud Platform for hosting infrastructure

## Contact

For questions or issues, please open an issue on GitHub or contact the repository owner.

---

**Made with PyTorch, Flask, and deployed on Google Cloud Platform**
