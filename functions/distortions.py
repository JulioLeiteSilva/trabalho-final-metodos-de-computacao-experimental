import cv2
import numpy as np

def apply_compression(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(compressed, cv2.IMREAD_COLOR)

def apply_resizing(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def apply_canny(image):
    return cv2.Canny(image, 100, 200)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_cropping(image, zoom):
    height, width = image.shape[:2]
    new_width, new_height = int(width / zoom), int(height / zoom)
    x_offset, y_offset = (width - new_width) // 2, (height - new_height) // 2
    cropped = image[y_offset:y_offset + new_height, x_offset:x_offset + new_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
