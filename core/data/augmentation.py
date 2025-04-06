# core/data/augmentation.py
"""
Técnicas de aumento de datos para mejorar el entrenamiento
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config.model_config import DATA_AUGMENTATION_CONFIG

def create_data_augmentation():
    """
    Crea un generador de aumento de datos con la configuración predefinida
    
    Returns:
        ImageDataGenerator configurado
    """
    return ImageDataGenerator(**DATA_AUGMENTATION_CONFIG)

def apply_augmentation(X_train, y_train, batch_size=32, custom_config=None):
    """
    Aplica aumento de datos al conjunto de entrenamiento
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        batch_size: Tamaño de lote
        custom_config: Configuración personalizada (opcional)
        
    Returns:
        Generador de datos aumentados
    """
    config = custom_config if custom_config else DATA_AUGMENTATION_CONFIG
    datagen = ImageDataGenerator(**config)
    
    return datagen.flow(X_train, y_train, batch_size=batch_size)

def create_tf_dataset(X, y, batch_size=32, augment=True, shuffle=True, buffer_size=1000):
    """
    Crea un dataset de TensorFlow optimizado para el entrenamiento
    
    Args:
        X: Datos de entrada
        y: Etiquetas
        batch_size: Tamaño de lote
        augment: Si aplicar aumento de datos
        shuffle: Si mezclar los datos
        buffer_size: Tamaño del buffer para mezcla
        
    Returns:
        tf.data.Dataset optimizado
    """
    # Crear dataset básico
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Mezclar si es necesario
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    # Aplicar aumento de datos si es necesario
    if augment:
        # Creamos funciones de aumento que operan sobre tensores
        def augment_fn(x, y):
            # Ejemplo de aumentos aleatorios
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, 0.1)
            x = tf.image.random_contrast(x, 0.8, 1.2)
            return x, y
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimizaciones finales
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset