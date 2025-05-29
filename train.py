import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from utils import create_base_image, add_scratches, add_blobs

def generate_synthetic_data(output_dir="synthetic_data", num_samples=400):
    """Genera datos sintéticos para entrenamiento"""
    classes = ['good', 'scratch', 'blob', 'mixed']
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    
    """SECCION PARA LAS IMAGENES BUENAS"""
    for i in range(num_samples // 4):
        img = create_base_image()
        cv2.imwrite(f"{output_dir}/good/good_{i}.jpg", img)
    
    """SECCION PARA LAS IMAGENES CON RASGUÑOS"""
    for i in range(num_samples // 4):
        img = create_base_image()
        img = add_scratches(img)
        cv2.imwrite(f"{output_dir}/scratch/scratch_{i}.jpg", img)
    
    """SECCION PARA LAS IMAGENES CON MANCHAS"""
    for i in range(num_samples // 4):
        img = create_base_image()
        img = add_blobs(img)
        cv2.imwrite(f"{output_dir}/blob/blob_{i}.jpg", img)
    
    """SECCION PARA LAS IMAGENES CON DEFECTOS"""
    for i in range(num_samples // 4):
        img = create_base_image()
        img = add_scratches(img)
        img = add_blobs(img)
        cv2.imwrite(f"{output_dir}/mixed/mixed_{i}.jpg", img)

def create_detection_model():
    """FUNCION PARA CREAR EL MODELO DE DETECCION"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(4, activation='softmax')  
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """ENTRENA Y GUARDA EL MODELO"""
    
    generate_synthetic_data()
    
    """PREPARA EL GENERADOR DE DATOS"""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        'synthetic_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        'synthetic_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        subset='validation'
    )
    
    """CREA Y ENTRENA EL MODELO"""
    model = create_detection_model()
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15
    )
    
    """GUARDA EL MODELO"""
    os.makedirs('models', exist_ok=True)
    model.save('models/defect_model.h5')
    print("✅ Modelo entrenado y guardado como 'models/defect_model.h5'")

if __name__ == "__main__":
    train_model()