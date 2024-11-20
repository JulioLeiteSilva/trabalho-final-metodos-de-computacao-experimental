from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import torch

# Carregar modelo pré-treinado
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

model.eval()

# Transformações necessárias para o modelo
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_ssim(original, distorted):
    # Redimensionar a imagem distorcida para combinar com a original
    if original.shape != distorted.shape:
        distorted = cv2.resize(distorted, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    distorted_gray = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)
    return ssim(original_gray, distorted_gray)



def calculate_psnr(original, distorted):
    # Redimensionar a imagem distorcida para combinar com a original
    if original.shape != distorted.shape:
        distorted = cv2.resize(distorted, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    
    mse = np.mean((original - distorted) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def classify_image(image):
    # Converter imagem OpenCV para PIL
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image_pil).unsqueeze(0)  # Adicionar dimensão batch
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()  # Retorna o índice da classe prevista
