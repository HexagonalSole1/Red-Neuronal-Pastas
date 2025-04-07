# services/prediction_service.py
"""
Servicio para realizar predicciones con modelos entrenados
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from config.app_config import DEFAULT_MODEL_PATH, CLASS_NAMES_PATH, OUTPUT_DIR
from core.data.preprocessing import resize_and_normalize_image, convert_heic_to_jpg

class PredictionService:
    """Servicio para realizar predicciones con modelos de clasificación"""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        """
        Inicializa el servicio de predicción
        
        Args:
            model_path: Ruta al modelo guardado
            class_names_path: Ruta al archivo con nombres de clases
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = []
        
        # Verificar si el modelo y los nombres de clase existen
        self._check_model_files()
    
    def _check_model_files(self):
        """Verifica si los archivos del modelo existen"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")
        
        if not os.path.exists(self.class_names_path):
            raise FileNotFoundError(f"No se encontró el archivo de clases en {self.class_names_path}")
    
    def load_model(self):
        """Carga el modelo si no está ya cargado"""
        if self.model is None:
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Modelo cargado desde: {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Error al cargar el modelo: {e}")
        
        return self.model
    
    def load_class_names(self):
        """Carga los nombres de las clases"""
        if not self.class_names:
            try:
                with open(self.class_names_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                print(f"Nombres de clases cargados: {len(self.class_names)} clases")
            except Exception as e:
                raise RuntimeError(f"Error al cargar nombres de clases: {e}")
        
        return self.class_names
    
    def predict_image(self, image_path, top_k=3, visualize=False, output_path=None):
        """
        Predice la clase de una imagen
        
        Args:
            image_path: Ruta a la imagen
            top_k: Número de predicciones principales a devolver
            visualize: Si mostrar la imagen con la predicción
            output_path: Ruta para guardar la visualización
            
        Returns:
            Diccionario con resultados de la predicción
        """
        # Verificar si la imagen existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen en {image_path}")
        
        # Cargar el modelo y los nombres de clases
        self.load_model()
        self.load_class_names()
        
        # Procesar la imagen según su formato
        if image_path.lower().endswith(('.heic', '.heif')):
            # Convertir HEIC a JPG
            jpg_path = convert_heic_to_jpg(image_path)
            if jpg_path is None:
                raise RuntimeError("No se pudo convertir la imagen HEIC a JPG")
            
            # Cargar la imagen convertida
            img = Image.open(jpg_path).convert('RGB')
        else:
            # Cargar imagen directamente
            img = Image.open(image_path).convert('RGB')
        
        # Preprocesar la imagen
        img_array = resize_and_normalize_image(img)
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de lote
        
        # Realizar la predicción
        predictions = self.model.predict(img_array)[0]
        
        # Obtener los índices ordenados por confianza (descendente)
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        # Preparar resultados
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'class': self.class_names[idx],
                'confidence': float(predictions[idx])
            })
        
        # Visualizar si se solicita
        if visualize or output_path:
            # Determinar la ruta de salida si no se proporciona
            if output_path is None and visualize:
                output_dir = os.path.join(OUTPUT_DIR, 'predictions')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, 
                    f"pred_{os.path.basename(image_path).split('.')[0]}.png"
                )
            
            if output_path:
                # Crear visualización
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.title(f'Predicción: {results[0]["class"]}\nConfianza: {results[0]["confidence"]:.2%}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                print(f"Visualización guardada en: {output_path}")
        
        return {
            'top_prediction': results[0],
            'all_predictions': results,
            'visualization_path': output_path if output_path else None
        }
    
    def batch_predict(self, image_paths, visualize=False, output_dir=None):
        """
        Realiza predicciones en lote para múltiples imágenes
        
        Args:
            image_paths: Lista de rutas a imágenes
            visualize: Si crear visualizaciones
            output_dir: Directorio para guardar visualizaciones
            
        Returns:
            Lista de resultados de predicción
        """
        results = []
        
        for image_path in image_paths:
            try:
                # Determinar ruta de salida si es necesario
                output_path = None
                if visualize and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir, 
                        f"pred_{os.path.basename(image_path).split('.')[0]}.png"
                    )
                
                # Realizar predicción
                prediction = self.predict_image(
                    image_path, 
                    visualize=visualize, 
                    output_path=output_path
                )
                
                # Añadir información de la imagen
                prediction['image_path'] = image_path
                results.append(prediction)
                
            except Exception as e:
                print(f"Error al procesar {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results

