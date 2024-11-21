from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
# from scipy.stats import entropy

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
    if original.shape != distorted.shape:
        distorted = cv2.resize(distorted, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    return ssim(original, distorted, channel_axis=2) * 100

def calculate_psnr(original, distorted):
    # Redimensionar a imagem distorcida para combinar com a original
    if original.shape != distorted.shape:
        distorted = cv2.resize(distorted, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    
    mse = np.mean((original - distorted) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# def calculate_entropy(image):
#     # Calcula a entropia da imagem com base no histograma normalizado.
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Calcula o histograma da imagem (256 níveis de intensidade)
#     hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    
#     # Normaliza o histograma para obter uma distribuição de probabilidade
#     hist_prob = hist / hist.sum()
    
#     # Calcula a entropia usando a fórmula de Shannon
#     entropy = -np.sum(hist_prob * np.log2(hist_prob + 1e-10))  # Evita log(0) adicionando 1e-10
    
#     return entropy
