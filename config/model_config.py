# config/model_config.py
"""
Configuración del modelo de clasificación
"""

# Parámetros de la imagen
IMAGE_SIZE = (224, 224)
IMG_HEIGHT, IMG_WIDTH = IMAGE_SIZE
CHANNELS = 3

# Parámetros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 5

# Parámetros de validación cruzada
K_FOLDS = 5

# Configuración de aumento de datos
DATA_AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2
}

# Configuración del modelo CNN
MODEL_CONFIG = {
    'base_model': 'MobileNetV2',
    'weights': 'imagenet',
    'include_top': False,
    'dense_layers': [128, 64],
    'dropout_rate': 0.2,
    'activation': 'relu',
    'final_activation': 'softmax'
}