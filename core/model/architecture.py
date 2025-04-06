# core/model/architecture.py
"""
Definiciones de arquitecturas de modelos
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from config.model_config import MODEL_CONFIG, IMAGE_SIZE, CHANNELS

def create_base_model(input_shape=None, config=None):
    """
    Crea el modelo base pre-entrenado
    
    Args:
        input_shape: Forma de entrada (alto, ancho, canales)
        config: Configuración personalizada
        
    Returns:
        Modelo base pre-entrenado
    """
    if input_shape is None:
        input_shape = (*IMAGE_SIZE, CHANNELS)
    
    if config is None:
        config = MODEL_CONFIG
    
    model_name = config.get('base_model', 'MobileNetV2')
    weights = config.get('weights', 'imagenet')
    include_top = config.get('include_top', False)
    
    if model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=include_top,
            weights=weights
        )
    elif model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=include_top,
            weights=weights
        )
    elif model_name == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=include_top,
            weights=weights
        )
    else:
        raise ValueError(f"Modelo base no soportado: {model_name}")
    
    # Congelar el modelo base para fine-tuning
    base_model.trainable = False
    
    return base_model

def create_classifier_model(num_classes, input_shape=None, config=None):
    """
    Crea un modelo completo de clasificación
    
    Args:
        num_classes: Número de clases a clasificar
        input_shape: Forma de entrada (alto, ancho, canales)
        config: Configuración personalizada
        
    Returns:
        Modelo completo compilado
    """
    if input_shape is None:
        input_shape = (*IMAGE_SIZE, CHANNELS)
    
    if config is None:
        config = MODEL_CONFIG
    
    # Obtener el modelo base
    base_model = create_base_model(input_shape, config)
    
    # Obtener parámetros de configuración
    dense_layers = config.get('dense_layers', [128, 64])
    dropout_rate = config.get('dropout_rate', 0.2)
    activation = config.get('activation', 'relu')
    final_activation = config.get('final_activation', 'softmax')
    
    # Construir el modelo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
    ])
    
    # Añadir capas densas según la configuración
    for units in dense_layers:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout_rate))
    
    # Añadir la capa de salida
    model.add(layers.Dense(num_classes, activation=final_activation))
    
    return model

def compile_model(model, learning_rate=0.001, metrics=None):
    """
    Compila un modelo con los parámetros especificados
    
    Args:
        model: Modelo a compilar
        learning_rate: Tasa de aprendizaje
        metrics: Lista de métricas (por defecto ['accuracy'])
        
    Returns:
        Modelo compilado
    """
    if metrics is None:
        metrics = ['accuracy']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    return model
