import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

def create_base_image(size=(224, 224)):
    """CREA LA IMAGEN BASE"""
    gray_value = np.random.randint(150, 200)
    img = np.ones((*size, 3), dtype=np.uint8) * gray_value
    noise = np.random.randint(-20, 20, (*size, 3), dtype=np.int16)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def add_scratches(img):
    """AÑADE RASGUÑOS SINTETICOS"""
    for _ in range(np.random.randint(1, 4)):
        start = (np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0]))
        end = (np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0]))
        color = (np.random.randint(50, 100),) * 3
        thickness = np.random.randint(1, 3)
        cv2.line(img, start, end, color, thickness)
    return img

def add_blobs(img):
    """AÑADE MANCHAS SINTETICAS"""
    for _ in range(np.random.randint(3, 8)):
        center = (np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0]))
        radius = np.random.randint(3, 15)
        color = (np.random.randint(70, 120),) * 3
        cv2.circle(img, center, radius, color, -1)
    return img

def preprocess_image(image, target_size=(224, 224)):
    """PREPROCESA LA IMAGEN PARA EL MODELO"""
    img = np.array(image)
    if len(img.shape) == 2:  # Convertir escala de grises a RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def load_model(model_path='models/defect_model.h5'):
    """CARGA EL MODELO ENTRENADO"""
    return tf.keras.models.load_model(model_path)

def predict_defects(model, image):
    """REALIZA LA PREDICCION"""
    preprocessed = preprocess_image(image)
    predictions = model.predict(preprocessed)[0]
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]
    return class_id, confidence