
# api/routes.py
"""
Rutas para la API REST
"""
import os
import base64
import uuid
import io
from PIL import Image
from flask import request, jsonify, current_app

from api import api_bp
from config.app_config import TEMP_UPLOADS_DIR, allowed_file
from core.data.preprocessing import convert_heic_to_jpg
from services.prediction_service import PredictionService
from services.class_service import ClassService

# Servicios
prediction_service = PredictionService()
class_service = ClassService()

@api_bp.route('/info', methods=['GET'])
def get_info():
    """Devuelve información sobre el modelo (API)"""
    try:
        # Obtener las clases disponibles
        class_names = prediction_service.load_class_names()
        
        # Información del modelo
        model_info = {
            'name': current_app.config.get('APP_NAME', 'Food Classification System'),
            'num_classes': len(class_names),
            'classes': class_names,
            'version': '2.0.0'
        }
        
        # Si existe el resumen del modelo, añadir información adicional
        model_summary_path = os.path.join(current_app.config['OUTPUT_DIR'], 'model_summary.txt')
        if os.path.exists(model_summary_path):
            summary_data = {}
            with open(model_summary_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        summary_data[key.strip()] = value.strip()
            
            # Añadir datos del resumen
            if 'Precisión promedio' in summary_data:
                model_info['accuracy'] = summary_data['Precisión promedio']
            
            if 'Tamaño de entrada' in summary_data:
                model_info['input_size'] = summary_data['Tamaño de entrada']
        
        return jsonify({
            'status': 'ok',
            'model': model_info
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Realiza una predicción con una imagen enviada"""
    try:
        # Verificar el tipo de contenido
        content_type = request.headers.get('Content-Type', '')
        
        # Inicializar variable para la imagen procesada
        img = None
        temp_filepath = None
        
        # Caso 1: application/json - imagen en base64
        if 'application/json' in content_type:
            if 'image' not in request.json:
                return jsonify({'status': 'error', 'message': 'No se envió ninguna imagen'}), 400
            
            # Decodificar la imagen en base64
            image_data = request.json['image']
            # Eliminar el prefijo 'data:image/jpeg;base64,' si existe
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            
            # Convertir de base64 a bytes
            image_bytes = base64.b64decode(image_data)
            
            # Guardar temporalmente para detectar si es HEIC
            temp_filepath = os.path.join(TEMP_UPLOADS_DIR, f"temp_{uuid.uuid4()}.bin")
            with open(temp_filepath, 'wb') as f:
                f.write(image_bytes)
            
            # Verificar si es HEIC por los primeros bytes (signature)
            is_heic = False
            with open(temp_filepath, 'rb') as f:
                header = f.read(12)
                # Verificar bytes característicos de HEIC/HEIF
                if b'ftyp' in header and (b'heic' in header or b'heif' in header or b'mif1' in header):
                    is_heic = True
            
            # Procesar según el tipo de imagen
            if is_heic:
                try:
                    jpg_filepath = convert_heic_to_jpg(temp_filepath)
                    if jpg_filepath:
                        img = Image.open(jpg_filepath).convert('RGB')
                    else:
                        raise ValueError("No se pudo convertir la imagen HEIC")
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)}), 400
                finally:
                    # Limpiar archivos temporales
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
            else:
                # Procesar como imagen normal
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Caso 2: multipart/form-data - archivo de imagen
        elif 'multipart/form-data' in content_type or request.files:
            if 'file' not in request.files:
                return jsonify({'status': 'error', 'message': 'No se envió ningún archivo'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'No se seleccionó ningún archivo'}), 400
            
            # Verificar si es un formato permitido
            if file and allowed_file(file.filename):
                # Guardar el archivo temporalmente
                filename = str(uuid.uuid4()) + '_' + file.filename
                temp_filepath = os.path.join(TEMP_UPLOADS_DIR, filename)
                file.save(temp_filepath)
                
                # Verificar si es HEIC/HEIF
                is_heic = file.filename.lower().endswith(('.heic', '.heif'))
                
                # Procesar según el formato
                if is_heic:
                    jpg_filepath = convert_heic_to_jpg(temp_filepath)
                    if jpg_filepath:
                        img = Image.open(jpg_filepath).convert('RGB')
                    else:
                        return jsonify({'status': 'error', 'message': 'No se pudo convertir la imagen HEIC'}), 400
                else:
                    img = Image.open(temp_filepath).convert('RGB')
            else:
                return jsonify({'status': 'error', 'message': 'Formato de archivo no permitido'}), 400
        
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Tipo de contenido no soportado. Use application/json o multipart/form-data'
            }), 415
        
        # Realizar la predicción usando el servicio
        if img:
            # Guardar la imagen temporal para la predicción
            if temp_filepath is None:
                temp_filepath = os.path.join(TEMP_UPLOADS_DIR, f"temp_{uuid.uuid4()}.jpg")
                img.save(temp_filepath, 'JPEG')
            
            # Obtener predicción
            try:
                result = prediction_service.predict_image(temp_filepath, top_k=5)
                
                # Limpiar archivo temporal
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                
                # Devolver resultados
                return jsonify({
                    'status': 'ok',
                    'prediction': result['top_prediction']['class'],
                    'confidence': result['top_prediction']['confidence'],
                    'all_predictions': result['all_predictions']
                })
            
            except Exception as e:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                return jsonify({'status': 'error', 'message': str(e)}), 500
            
        return jsonify({'status': 'error', 'message': 'Error al procesar la imagen'}), 400
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/classes', methods=['GET'])
def get_classes():
    """Devuelve la lista de clases disponibles"""
    try:
        # Obtener estadísticas detalladas de clases
        stats = class_service.get_class_stats()
        
        # Simplificar para la respuesta
        classes = []
        for name, info in stats.items():
            classes.append({
                'name': name,
                'total_images': info['total_images'],
                'error': info.get('error', None)
            })
        
        return jsonify({
            'status': 'ok',
            'count': len(classes),
            'classes': classes
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Comprueba si el servidor está funcionando"""
    return jsonify({'status': 'ok', 'message': 'El servidor está en funcionamiento'})

@api_bp.route('/model_status', methods=['GET'])
def model_status():
    """Comprueba el estado del modelo"""
    try:
        # Verificar si los archivos existen
        model_path = current_app.config.get('DEFAULT_MODEL_PATH')
        class_names_path = current_app.config.get('CLASS_NAMES_PATH')
        
        model_exists = os.path.exists(model_path)
        class_names_exist = os.path.exists(class_names_path)
        
        # Intentar cargar el modelo
        model_loaded = False
        if model_exists:
            try:
                prediction_service.load_model()
                model_loaded = True
            except:
                pass
        
        status = {
            'model_file_exists': model_exists,
            'class_names_file_exists': class_names_exist,
            'model_loaded': model_loaded
        }
        
        if all(status.values()):
            return jsonify({'status': 'ok', 'details': status})
        else:
            return jsonify({'status': 'warning', 'details': status})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

