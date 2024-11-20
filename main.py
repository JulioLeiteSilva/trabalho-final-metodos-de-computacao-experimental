import os
import cv2
import pandas as pd
from functions.distortions import *
from functions.metrics import *

# Mapeamento de identificadores do ImageNet para índices numéricos
imagenette_class_mapping = {
    'n01440764': 0,  # Classe 0
    'n02102040': 1,  # Classe 1
    'n02979186': 2,  # Classe 2
    'n03000684': 3,  # Classe 3
    'n03028079': 4,  # Classe 4
    'n03394916': 5,  # Classe 5
    'n03417042': 6,  # Classe 6
    'n03425413': 7,  # Classe 7
    'n03445777': 8,  # Classe 8
    'n03888257': 9   # Classe 9
}

def ensure_directory_exists(file_path):
    """Garantir que o diretório do arquivo exista. Se não, crie-o."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_images(input_folder, output_csv, distortions):
    results = []

    header = ['Class', 'Image', 'Original (SSIM)', 'Original (PSNR)', 'Original (Identified Class)', 'Original (Confidence)', 'Original (Correct)']
    distortion_columns = []
    for distortion_name in distortions.keys():
        distortion_columns.append(f"{distortion_name} (SSIM)")
        distortion_columns.append(f"{distortion_name} (PSNR)")
        distortion_columns.append(f"{distortion_name} (Identified Class)")
        distortion_columns.append(f"{distortion_name} (Confidence)")
        distortion_columns.append(f"{distortion_name} (Correct)")
    header.extend(distortion_columns)

    print(f"Iniciando o processamento de imagens na pasta: {input_folder}")

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            print(f"Processando a classe: {class_folder}")
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)

                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Arquivo ignorado (não é uma imagem válida): {image_file}")
                    continue

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Erro ao carregar a imagem: {image_path}")
                    continue

                expected_class = imagenette_class_mapping[class_folder]

                original_label, original_confidence = classify_image(image)
                original_correct = int(original_label in imagenette_class_mapping)
                row = {
                    'Class': expected_class,
                    'Image': image_file,
                    'Original (SSIM)': 1.0,
                    'Original (PSNR)': float('inf'),
                    'Original (Identified Class)': original_label,
                    'Original (Confidence)': original_confidence,
                    'Original (Correct)': original_correct
                }

                for distortion_name, (distortion_fn, _) in distortions.items():
                    distorted = distortion_fn(image)

                    if distorted.ndim != image.ndim:
                        distorted = cv2.cvtColor(distorted, cv2.COLOR_GRAY2BGR)

                    ssim_score = calculate_ssim(image, distorted)
                    psnr_score = calculate_psnr(image, distorted)
                    distorted_label, distorted_confidence = classify_image(distorted)
                    correct = int(distorted_label in imagenette_class_mapping)
                    row[f"{distortion_name} (SSIM)"] = ssim_score
                    row[f"{distortion_name} (PSNR)"] = psnr_score
                    row[f"{distortion_name} (Identified Class)"] = distorted_label
                    row[f"{distortion_name} (Confidence)"] = distorted_confidence
                    row[f"{distortion_name} (Correct)"] = correct

                print(f"Imagem processada: {image_file}")
                results.append(row)

    ensure_directory_exists(output_csv)
    pd.DataFrame(results, columns=header).to_csv(output_csv, index=False)
    print(f"Resultados salvos em: {output_csv}")

distortions = {
    'Compression_60': (lambda img: apply_compression(img, 60), 'Quality=60'),
    'Compression_10': (lambda img: apply_compression(img, 10), 'Quality=10'),
    'Resize_128x128': (lambda img: apply_resizing(img, 128, 128), 'Width=128, Height=128'),
    'Gaussian_Noise_0.5': (lambda img: apply_gaussian_noise(img, 0.5), 'Sigma=0.5'),
    'Canny': (apply_canny, 'Default Parameters'),
    'Grayscale': (convert_to_grayscale, 'No Parameters'),
    'Crop_1.5': (lambda img: apply_cropping(img, 1.5), 'Zoom=1.5')
}

assert os.path.exists('../imagenette2/train'), "Caminho para o dataset de treino não encontrado!"
assert os.path.exists('../imagenette2/val'), "Caminho para o dataset de validação não encontrado!"

process_images('../imagenette2/train', 'output/train/results.csv', distortions)
process_images('../imagenette2/val', 'output/val/results.csv', distortions)
