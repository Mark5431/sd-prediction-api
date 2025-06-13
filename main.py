from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import io
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapping
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
idx_to_label = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

# Define model loader
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "mobilenet":
        model_ft = models.mobilenet_v2(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
        model_ft.classifier[1] = nn.Linear(model_ft.last_channel, num_classes)
    else:
        raise ValueError("Invalid model name")
    return model_ft, input_size

# Init model
model_name = "mobilenet"
num_classes = 7
feature_extract = False
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Device
USE_GPU = True
device = torch.device('cuda:0' if USE_GPU and torch.cuda.is_available() else 'cpu')
model = model_ft.to(device)

# Load trained model
model.load_state_dict(torch.load('skin_cancer_model (1).pth', map_location=device, weights_only=False))
model.eval()

# Image preprocessing
norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# Grad-CAM hooks
gradients = []
activations = []

target_layer = model.features[-1]
def forward_hook(module, input, output):
    activations.append(output)
def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_backward_hook(backward_hook)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        input_tensor = test_transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Prepare response
        response = {
            "predicted_class": idx_to_label[predicted.item()],
            "lesion_type": lesion_type_dict[idx_to_label[predicted.item()]],
            "confidence_scores": {
                lesion_type_dict[idx_to_label[idx]]: prob.item() * 100
                for idx, prob in enumerate(probabilities[0])
            }
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gradcam/")
async def gradcam(file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        input_tensor = test_transform(image).unsqueeze(0).to(device)

        # Forward pass
        model.zero_grad()
        outputs = model(input_tensor)
        predicted = torch.argmax(outputs, 1)
        class_score = outputs[0, predicted.item()]
        class_score.backward()

        # Extract activation & gradient
        activation = activations[0].squeeze().detach()
        gradient = gradients[0].squeeze().detach()
        weights = gradient.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activation).sum(dim=0)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        cam = cam.cpu().numpy()

        # Generate heatmap
        heatmap = cv2.resize(cam, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Original image for overlay
        orig_img = image.resize((224, 224))
        orig_np = np.array(orig_img)
        if orig_np.ndim == 2:  # grayscale
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)

        overlay = cv2.addWeighted(orig_np, 0.5, heatmap, 0.5, 0)

        # Convert overlay to bytes
        _, buffer = cv2.imencode('.jpg', overlay)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Clean buffers and remove hooks on shutdown
@app.on_event("shutdown")
def cleanup():
    activations.clear()
    gradients.clear()
    forward_handle.remove()
    backward_handle.remove()