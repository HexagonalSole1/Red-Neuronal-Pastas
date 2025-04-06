
# core/data/dataset.py
"""
Manejo y preparación de datasets
"""
import os
import glob
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from config.model_config import MIN_SAMPLES_PER_CLASS, IMAGE_SIZE
from core.data.preprocessing import resize_and_normalize_image
from core.utils.file_utils import ensure_dir

def prepare_dataset(data_dir, test_split=0.2, min_samples=MIN_SAMPLES_PER_CLASS, image_size=IMAGE_SIZE):
    """
    Prepara el conjunto de datos a partir de un directorio de imágenes organizadas por clase
    
    Args:
        data_dir: Directorio con las imágenes por clase
        test_split: Proporción para el conjunto de prueba
        min_samples: Número mínimo de imágenes por clase
        image_size: Tamaño objetivo de las imágenes (ancho, alto)
        
    Returns:
        X: Array de imágenes preprocesadas
        y: Etiquetas codificadas en one-hot
        class_names: Lista con los nombres de las clases
    """
    X = []  # Datos de imágenes
    y = []  # Etiquetas
    class_names = []  # Nombres de clases
    valid_class_indices = []  # Índices de clases válidas
    
    print("\n=== Preparando conjunto de datos ===")
    
    # Verificar que exista el directorio
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No se encontró el directorio {data_dir}")
    
    # Obtener las clases (carpetas)
    folders = [f for f in sorted(os.listdir(data_dir)) 
              if os.path.isdir(os.path.join(data_dir, f))]
    
    if not folders:
        raise ValueError(f"No se encontraron carpetas de clases en {data_dir}")
    
    # Crear directorios para división train/test
    train_base_dir = os.path.join("data", "processed", "train")
    test_base_dir = os.path.join("data", "processed", "test")
    ensure_dir(train_base_dir)
    ensure_dir(test_base_dir)
    
    # Procesar cada clase
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(data_dir, folder)
        
        print(f"\nProcesando clase: {folder} (índice {idx})")
        
        # Crear directorios para entrenamiento y prueba para esta clase
        train_dir = os.path.join(train_base_dir, folder)
        test_dir = os.path.join(test_base_dir, folder)
        ensure_dir(train_dir)
        ensure_dir(test_dir)
        
        # Obtener todas las imágenes
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        # Verificar si hay suficientes imágenes
        if len(image_files) < min_samples:
            print(f"⚠️ Advertencia: La clase '{folder}' solo tiene {len(image_files)} imágenes, "
                  f"se requieren al menos {min_samples}. Esta clase será ignorada.")
            continue
        
        print(f"Encontradas {len(image_files)} imágenes")
        
        # Dividir en entrenamiento y prueba
        if len(image_files) >= 2:
            # Calcular número de muestras para prueba
            n_test = max(1, int(len(image_files) * test_split))
            n_train = len(image_files) - n_test
            
            print(f"Dividiendo en {n_train} imágenes de entrenamiento y {n_test} de prueba")
            
            # Dividir archivos
            if len(image_files) < 10:
                np.random.seed(42)
                np.random.shuffle(image_files)
                train_files = image_files[:-n_test]
                test_files = image_files[-n_test:]
            else:
                train_files, test_files = train_test_split(
                    image_files, test_size=n_test/len(image_files), random_state=42
                )
        else:
            # Si solo hay una imagen, usar para entrenamiento y prueba
            train_files = image_files
            test_files = image_files
            print("⚠️ Advertencia: Solo hay una imagen, se usará para entrenamiento y prueba")
        
        # Copiar archivos a carpetas de train/test
        for file in train_files:
            dest = os.path.join(train_dir, os.path.basename(file))
            shutil.copy(file, dest)
        
        for file in test_files:
            dest = os.path.join(test_dir, os.path.basename(file))
            shutil.copy(file, dest)
        
        # Procesar cada imagen
        class_X = []  # Imágenes de esta clase
        class_y = []  # Etiquetas de esta clase
        
        for img_path in image_files:
            try:
                # Cargar y procesar imagen
                img = Image.open(img_path).convert('RGB')
                img_array = resize_and_normalize_image(img, image_size)
                
                class_X.append(img_array)
                class_y.append(idx)
                
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
        
        # Solo añadir la clase si se procesaron suficientes imágenes
        if len(class_X) >= min_samples:
            X.extend(class_X)
            y.extend(class_y)
            class_names.append(folder)
            valid_class_indices.append(idx)
            print(f"✓ Clase '{folder}' añadida con {len(class_X)} imágenes")
        else:
            print(f"⚠️ Advertencia: No se pudieron procesar suficientes imágenes para '{folder}', "
                  f"se requieren al menos {min_samples}. Esta clase será ignorada.")
    
    # Verificar si hay suficientes clases
    if len(class_names) == 0:
        raise ValueError("No se encontraron clases con suficientes imágenes para entrenar")
    
    # Ajustar índices si algunas clases fueron ignoradas
    if len(valid_class_indices) < len(folders):
        # Crear un mapa de índice original a nuevo índice
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_class_indices)}
        # Remapear las etiquetas
        y = [idx_map[label] for label in y]
    
    # Convertir a arrays de numpy
    X = np.array(X)
    y = np.array(y)
    
    # Codificar etiquetas en one-hot
    y_encoded = to_categorical(y, num_classes=len(class_names))
    
    print(f"\n✅ Dataset preparado: {X.shape[0]} imágenes en {len(class_names)} clases")
    
    return X, y_encoded, class_names

