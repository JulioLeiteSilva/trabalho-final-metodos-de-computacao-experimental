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
    
    distorted_images_dir = "distortedImages"
    ensure_directory_exists(distorted_images_dir)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            print(f"Processando a classe: {class_folder}")
            class_output_dir = os.path.join(distorted_images_dir, class_folder)
            ensure_directory_exists(class_output_dir)

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
                    row[f"{distortion_name} (PSNR)"] =  psnr_score
                    row[f"{distortion_name} (Identified Class)"] = distorted_label
                    row[f"{distortion_name} (Confidence)"] = distorted_confidence
                    row[f"{distortion_name} (Correct)"] = correct

                    distortion_dir = os.path.join(class_output_dir, distortion_name)
                    ensure_directory_exists(distortion_dir)
                    distorted_image_path = os.path.join(distortion_dir, image_file)
                    cv2.imwrite(distorted_image_path, distorted)

                print(f"Imagem processada: {image_file}")
                results.append(row)

    ensure_directory_exists(output_csv)
    pd.DataFrame(results, columns=header).to_csv(output_csv, index=False)
    print(f"Resultados salvos em: {output_csv}")

distortions = {
    'Compression_70': (lambda img: apply_compression(img, 70), 'Quality=70'),
    'Compression_20': (lambda img: apply_compression(img, 20), 'Quality=20'),
    'Resize_128x128': (lambda img: apply_resizing(img, 128, 128), 'Width=128, Height=128'),
    'Gaussian_Noise_25': (lambda img: gaussian_noise(img, mean=10, std=10), 'Mean=0, Std=10'),
    'Gaussian_Noise_50': (lambda img: gaussian_noise(img, mean=0, std=50), 'Mean=0, Std=50'),
    'Canny': (apply_canny, 'Default Parameters'),
    'Grayscale': (convert_to_grayscale, 'No Parameters'),
    'Crop_1.5': (lambda img: apply_cropping(img, 1.5), 'Zoom=1.5')
}

assert os.path.exists('../imagenette2/train'), "Caminho para o dataset de treino não encontrado!"
assert os.path.exists('../imagenette2/val'), "Caminho para o dataset de validação não encontrado!"

#process_images('../imagenette2/train', 'output/train/results.csv', distortions)
process_images('../imagenette2/val', 'output/val/results.csv', distortions)
