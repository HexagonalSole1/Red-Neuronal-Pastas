
# core/model/evaluation.py
"""
Evaluación y métricas para modelos
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        
    Returns:
        Métricas de evaluación
    """
    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Obtener predicciones
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calcular métricas adicionales
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'y_pred': y_pred,
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'report': report_dict
    }

def plot_confusion_matrix(y_true, y_pred, class_names, output_path=None):
    """
    Genera y guarda la matriz de confusión
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones
        class_names: Nombres de las clases
        output_path: Ruta para guardar la imagen
    
    Returns:
        Matriz de confusión
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    
    # Guardar si se especificó una ruta
    if output_path:
        plt.savefig(output_path)
        print(f"Matriz de confusión guardada en: {output_path}")
    
    # Cerrar figura para liberar memoria
    plt.close()
    
    return cm

def plot_training_history(histories, k_folds=None, output_path=None):
    """
    Grafica el historial de entrenamiento
    
    Args:
        histories: Historial o lista de historiales
        k_folds: Número de folds (si es validación cruzada)
        output_path: Ruta para guardar la gráfica
    """
    if k_folds is None:
        # Un solo historial
        history = histories
        plt.figure(figsize=(12, 5))
        
        # Gráfico de precisión
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='entrenamiento')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='validación')
        plt.title('Precisión durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='entrenamiento')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='validación')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
    else:
        # Múltiples historiales (validación cruzada)
        plt.figure(figsize=(15, 10))
        
        for i in range(min(k_folds, len(histories))):
            # Gráfico de precisión para cada fold
            plt.subplot(2, 3, i+1)
            plt.plot(histories[i].history['accuracy'], label='train')
            if 'val_accuracy' in histories[i].history:
                plt.plot(histories[i].history['val_accuracy'], label='validation')
            plt.title(f'Fold {i+1}')
            plt.xlabel('Épocas')
            plt.ylabel('Precisión')
            plt.legend()
    
    plt.tight_layout()
    
    # Guardar si se especificó una ruta
    if output_path:
        plt.savefig(output_path)
        print(f"Historial de entrenamiento guardado en: {output_path}")
    
    plt.close()

def save_classification_report(y_true, y_pred, class_names, output_path):
    """
    Guarda un reporte de clasificación detallado
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones
        class_names: Nombres de las clases
        output_path: Ruta para guardar el reporte
    """
    # Generar reporte
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Convertir a DataFrame
    df_report = pd.DataFrame(report).transpose()
    
    # Guardar como CSV
    df_report.to_csv(output_path)
    print(f"Reporte de clasificación guardado en: {output_path}")
    
    return df_report

def generate_model_summary(model, class_names, val_accuracies=None, output_path=None):
    """
    Genera un resumen del modelo con información relevante
    
    Args:
        model: Modelo entrenado
        class_names: Nombres de las clases
        val_accuracies: Lista de precisiones de validación si se usó validación cruzada
        output_path: Ruta para guardar el resumen
    """
    # Crear resumen
    summary = {
        'num_classes': len(class_names),
        'classes': class_names,
        'model_type': model.__class__.__name__,
        'input_shape': model.input_shape[1:]
    }
    
    # Añadir métricas de validación cruzada si están disponibles
    if val_accuracies is not None:
        summary['cross_validation'] = {
            'folds': len(val_accuracies),
            'mean_accuracy': float(np.mean(val_accuracies)),
            'std_accuracy': float(np.std(val_accuracies)),
            'min_accuracy': float(np.min(val_accuracies)),
            'max_accuracy': float(np.max(val_accuracies))
        }
    
    # Guardar como texto si se especificó una ruta
    if output_path:
        with open(output_path, 'w') as f:
            # Guardar información básica
            f.write(f"Número de clases: {summary['num_classes']}\n")
            f.write(f"Clases: {', '.join(summary['classes'])}\n")
            f.write(f"Tipo de modelo: {summary['model_type']}\n")
            f.write(f"Tamaño de entrada: {summary['input_shape'][0]}x{summary['input_shape'][1]}\n")
            
            # Guardar información de validación cruzada si está disponible
            if 'cross_validation' in summary:
                cv = summary['cross_validation']
                f.write(f"Número de folds: {cv['folds']}\n")
                f.write(f"Precisión promedio: {cv['mean_accuracy']:.4f}\n")
                f.write(f"Desviación estándar: {cv['std_accuracy']:.4f}\n")
        
        print(f"Resumen del modelo guardado en: {output_path}")
    
    return summary