# services/training_service.py
"""
Servicio para el entrenamiento de modelos
"""
import os
import numpy as np
import tensorflow as tf

from config.app_config import MODELS_DIR, OUTPUT_DIR, CLASS_NAMES_PATH
from config.model_config import EPOCHS, BATCH_SIZE, K_FOLDS, LEARNING_RATE
from core.data.dataset import prepare_dataset
from core.model.architecture import create_classifier_model, compile_model
from core.model.trainer import train_with_cross_validation
from core.model.evaluation import (
    evaluate_model, plot_confusion_matrix, plot_training_history,
    save_classification_report, generate_model_summary
)
from core.utils.visualization import visualize_class_examples
from core.utils.file_utils import ensure_dir, convert_all_heic_in_directory

class TrainingService:
    """Servicio para entrenar y evaluar modelos de clasificación"""
    
    def __init__(self, data_dir='data/raw'):
        """
        Inicializa el servicio de entrenamiento
        
        Args:
            data_dir: Directorio con los datos de entrenamiento
        """
        self.data_dir = data_dir
        ensure_dir(MODELS_DIR)
        ensure_dir(OUTPUT_DIR)
    
    def train_model(self, epochs=EPOCHS, batch_size=BATCH_SIZE, k_folds=K_FOLDS, 
                    learning_rate=LEARNING_RATE, save_model_path=None):
        """
        Entrena un nuevo modelo con los datos disponibles
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño de lote
            k_folds: Número de folds para validación cruzada
            learning_rate: Tasa de aprendizaje
            save_model_path: Ruta para guardar el modelo (por defecto, 'models/best_model.h5')
            
        Returns:
            Resultados del entrenamiento
        """
        # Establecer ruta de guardado por defecto si no se especifica
        if save_model_path is None:
            save_model_path = os.path.join(MODELS_DIR, 'best_model.h5')
        
        # Convertir imágenes HEIC si es necesario
        convert_all_heic_in_directory(self.data_dir)
        
        # Preparar el dataset
        X, y, class_names = prepare_dataset(self.data_dir)
        num_classes = len(class_names)
        
        # Visualizar ejemplos de las clases
        visualize_class_examples(X, np.argmax(y, axis=1), class_names,
                               output_path=os.path.join(OUTPUT_DIR, 'class_examples.png'))
        
        # Definir función para crear modelo
        def create_model(num_classes):
            model = create_classifier_model(num_classes)
            return compile_model(model, learning_rate=learning_rate)
        
        # Entrenar con validación cruzada
        histories, val_accuracies, best_model = train_with_cross_validation(
            X, y, num_classes, create_model, k_folds, epochs, batch_size
        )
        
        # Guardar el mejor modelo
        best_model.save(save_model_path)
        print(f"Mejor modelo guardado en: {save_model_path}")
        
        # Evaluar en todo el conjunto de datos
        eval_results = evaluate_model(best_model, X, y)
        
        # Guardar resultados
        plot_confusion_matrix(
            eval_results['y_true'], 
            eval_results['y_pred'], 
            class_names,
            output_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        )
        
        plot_training_history(
            histories, 
            k_folds,
            output_path=os.path.join(OUTPUT_DIR, 'training_history.png')
        )
        
        save_classification_report(
            eval_results['y_true'],
            eval_results['y_pred'],
            class_names,
            output_path=os.path.join(OUTPUT_DIR, 'classification_report.csv')
        )
        
        generate_model_summary(
            best_model,
            class_names,
            val_accuracies,
            output_path=os.path.join(OUTPUT_DIR, 'model_summary.txt')
        )
        
        # Guardar los nombres de las clases
        with open(CLASS_NAMES_PATH, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        # Preparar resultados
        results = {
            'accuracy': float(np.mean(val_accuracies)),
            'std_accuracy': float(np.std(val_accuracies)),
            'num_classes': num_classes,
            'classes': class_names,
            'model_path': save_model_path,
            'class_names_path': CLASS_NAMES_PATH
        }
        
        return results
