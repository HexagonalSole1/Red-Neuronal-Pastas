
# core/utils/visualization.py
"""
Utilidades para visualización de datos e imágenes
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def visualize_class_examples(X, y, class_names, samples_per_class=3, output_path=None):
    """
    Visualiza ejemplos de cada clase
    
    Args:
        X: Datos de imágenes
        y: Etiquetas (no codificadas)
        class_names: Nombres de las clases
        samples_per_class: Número de ejemplos por clase
        output_path: Ruta para guardar la visualización
    """
    num_classes = len(class_names)
    plt.figure(figsize=(15, 2 * num_classes))
    
    for class_idx in range(num_classes):
        # Obtener imágenes de esta clase
        indices = np.where(y == class_idx)[0]
        
        # Si hay suficientes imágenes, mostrar samples_per_class ejemplos
        if len(indices) >= samples_per_class:
            sample_indices = indices[:samples_per_class]
            
            for i, sample_idx in enumerate(sample_indices):
                plt.subplot(num_classes, samples_per_class, class_idx * samples_per_class + i + 1)
                plt.imshow(X[sample_idx])
                plt.title(class_names[class_idx])
                plt.axis('off')
    
    plt.tight_layout()
    
    # Guardar si se especificó una ruta
    if output_path:
        # Asegurar que el directorio exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Ejemplos de clases guardados en: {output_path}")
    
    plt.close()

def plot_learning_curves(history, output_path=None):
    """
    Grafica las curvas de aprendizaje de un entrenamiento
    
    Args:
        history: Historial de entrenamiento
        output_path: Ruta para guardar la gráfica
    """
    plt.figure(figsize=(12, 4))
    
    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    
    # Guardar si se especificó una ruta
    if output_path:
        # Asegurar que el directorio exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Curvas de aprendizaje guardadas en: {output_path}")
    
    plt.close()

def create_prediction_visualization(image_path, prediction, confidence, output_path=None):
    """
    Crea una visualización de una predicción
    
    Args:
        image_path: Ruta a la imagen
        prediction: Clase predicha
        confidence: Confianza de la predicción (0-1)
        output_path: Ruta para guardar la visualización
        
    Returns:
        PIL.Image: Imagen con la visualización
    """
    # Cargar imagen
    img = Image.open(image_path).convert('RGB')
    
    # Crear figura
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    # Añadir título con predicción
    plt.title(f"Predicción: {prediction}\nConfianza: {confidence:.2%}", 
              fontsize=14, color='black', pad=10)
    
    # Añadir barra de confianza
    ax = plt.gca()
    ax.axis('off')
    
    # Añadir barra de confianza en la parte inferior
    bar_height = 0.03
    bar_y = 0.05
    bar_color = plt.cm.RdYlGn(confidence)
    
    # Barra de fondo (gris)
    ax.add_patch(plt.Rectangle((0.1, bar_y), 0.8, bar_height, 
                           transform=ax.transAxes, facecolor='lightgray'))
    
    # Barra de confianza (coloreada)
    ax.add_patch(plt.Rectangle((0.1, bar_y), 0.8 * confidence, bar_height, 
                           transform=ax.transAxes, facecolor=bar_color))
    
    # Texto de confianza
    ax.text(0.5, bar_y - 0.02, f"{confidence:.1%}", 
            transform=ax.transAxes, ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Guardar si se especificó una ruta
    if output_path:
        # Asegurar que el directorio exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualización guardada en: {output_path}")
    
    # Convertir figura a imagen
    plt.close()
    
    if output_path:
        return Image.open(output_path)
    
    return None

def plot_class_distribution(class_counts, class_names=None, output_path=None):
    """
    Grafica la distribución de clases en el conjunto de datos
    
    Args:
        class_counts: Diccionario con conteo de clases {clase: cantidad}
        class_names: Lista de nombres de clases (opcional)
        output_path: Ruta para guardar la gráfica
    """
    if class_names is None:
        class_names = list(class_counts.keys())
    
    # Extraer cantidades en el mismo orden que los nombres
    counts = [class_counts.get(name, 0) for name in class_names]
    
    plt.figure(figsize=(12, 6))
    
    # Crear gráfico de barras
    ax = sns.barplot(x=class_names, y=counts)
    
    # Añadir etiquetas y título
    plt.title('Distribución de Clases', fontsize=16)
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Cantidad de Imágenes', fontsize=12)
    
    # Rotar etiquetas del eje x si hay muchas clases
    if len(class_names) > 6:
        plt.xticks(rotation=45, ha='right')
    
    # Añadir valores sobre las barras
    for i, count in enumerate(counts):
        ax.text(i, count + max(counts) * 0.01, str(count), 
                horizontalalignment='center', fontsize=10)
    
    plt.tight_layout()
    
    # Guardar si se especificó una ruta
    if output_path:
        # Asegurar que el directorio exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Distribución de clases guardada en: {output_path}")
    
    plt.close()