# core/model/trainer.py
"""
Funcionalidad para entrenamiento de modelos
"""
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config.model_config import EPOCHS, BATCH_SIZE, LEARNING_RATE, K_FOLDS
from core.data.augmentation import apply_augmentation, create_tf_dataset

def train_model(model, X_train, y_train, X_val=None, y_val=None, 
                epochs=EPOCHS, batch_size=BATCH_SIZE, use_augmentation=True):
    """
    Entrena un modelo con los datos proporcionados
    
    Args:
        model: Modelo a entrenar
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_val: Datos de validación (opcional)
        y_val: Etiquetas de validación (opcional)
        epochs: Número de épocas
        batch_size: Tamaño de lote
        use_augmentation: Si utilizar aumento de datos
        
    Returns:
        Historial de entrenamiento
    """
    # Definir callbacks para mejorar el entrenamiento
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3, min_lr=0.00001)
    ]
    
    # Preparar datos de entrenamiento con aumento si es necesario
    if use_augmentation:
        if tf.executing_eagerly():
            # Usar tf.data API para mejor rendimiento
            train_dataset = create_tf_dataset(
                X_train, y_train, batch_size=batch_size, augment=True
            )
            
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = create_tf_dataset(
                    X_val, y_val, batch_size=batch_size, augment=False, shuffle=False
                )
            
            # Entrenar el modelo
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Usar ImageDataGenerator para versiones más antiguas
            train_generator = apply_augmentation(X_train, y_train, batch_size)
            
            # Entrenar el modelo
            history = model.fit(
                train_generator,
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val) if X_val is not None else None,
                callbacks=callbacks,
                verbose=1
            )
    else:
        # Entrenar sin aumento de datos
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
    
    return history

def train_with_cross_validation(X, y, num_classes, architecture_fn, 
                               k_folds=K_FOLDS, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Entrena un modelo usando validación cruzada
    
    Args:
        X: Datos completos
        y: Etiquetas completas
        num_classes: Número de clases
        architecture_fn: Función que crea el modelo
        k_folds: Número de particiones
        epochs: Número de épocas por fold
        batch_size: Tamaño de lote
        
    Returns:
        histories: Lista de historiales de entrenamiento
        val_accuracies: Lista de precisiones de validación
        best_model: Mejor modelo entrenado
    """
    # Ajustar k_folds si hay pocas muestras
    sample_count = X.shape[0]
    if sample_count < k_folds * 2:
        new_k_folds = max(2, sample_count // 2)
        print(f"⚠️ Advertencia: Demasiados folds ({k_folds}) para {sample_count} muestras.")
        print(f"Ajustando a {new_k_folds} folds para evitar errores.")
        k_folds = new_k_folds
    
    # Definir la validación cruzada
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    histories = []
    val_accuracies = []
    best_accuracy = 0
    best_model = None
    
    # Iterar sobre los folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f'Entrenando fold {fold+1}/{k_folds}')
        
        # Dividir los datos para este fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Crear un nuevo modelo para este fold
        model = architecture_fn(num_classes)
        
        # Entrenar el modelo
        history = train_model(
            model, X_train, y_train, X_val, y_val, 
            epochs=epochs, batch_size=batch_size
        )
        
        # Evaluar el modelo
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f'Fold {fold+1}: Precisión de validación = {val_accuracy:.4f}')
        
        # Guardar resultados
        histories.append(history)
        val_accuracies.append(val_accuracy)
        
        # Actualizar mejor modelo si corresponde
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
    
    # Imprimir resumen
    mean_accuracy = np.mean(val_accuracies)
    std_accuracy = np.std(val_accuracies)
    print(f"\nResultados de validación cruzada ({k_folds} folds):")
    print(f"Precisión media: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return histories, val_accuracies, best_model