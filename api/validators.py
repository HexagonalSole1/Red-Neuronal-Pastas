
# api/validators.py
"""
Validación de datos para la API
"""
import os
from functools import wraps
from flask import request, jsonify
from config.app_config import allowed_file, TEMP_UPLOADS_DIR

def validate_image_upload(f):
    """
    Decorador para validar la subida de imágenes
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Verificar si hay archivos
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No se envió ningún archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No se seleccionó ningún archivo'}), 400
        
        # Verificar el tipo de archivo
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Tipo de archivo no permitido'}), 400
        
        return f(*args, **kwargs)
    
    return decorated_function

def validate_json_image(f):
    """
    Decorador para validar las imágenes enviadas como JSON (base64)
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Verificar si es JSON
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Se esperaba Content-Type: application/json'}), 400
        
        # Verificar si contiene el campo 'image'
        if 'image' not in request.json:
            return jsonify({'status': 'error', 'message': 'No se envió ninguna imagen'}), 400
        
        # Verificar que el campo 'image' no esté vacío
        image_data = request.json['image']
        if not image_data:
            return jsonify({'status': 'error', 'message': 'El campo image está vacío'}), 400
        
        return f(*args, **kwargs)
    
    return decorated_function
