from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Carregar o modelo ResNet50 pré-treinado
MODEL = ResNet50(weights="imagenet")

def classify_image(img):
    """
    Classifica uma imagem usando o modelo ResNet50.
    Retorna o rótulo da classe com maior probabilidade e a confiança (%).
    """
    try:
        # Pré-processar a imagem
        x = cv2.resize(img, (224, 224))  # Redimensionar para 224x224
        x = x[:, :, ::-1].astype(np.float32)  # Converter de BGR para RGB
        x = np.expand_dims(x, axis=0)  # Adicionar dimensão batch
        x = preprocess_input(x)  # Pré-processamento específico do ResNet50
        
        # Fazer a predição
        preds = MODEL.predict(x)
        classes = decode_predictions(preds, top=1)[0]  # Retorna a classe top-1
        label, confidence = classes[0][1], classes[0][2] * 100  # Nome e confiança (%)
        return label, confidence
    except Exception as e:
        print(f"Erro na classificação: {e}")
        return "Unknown", 0.0

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
