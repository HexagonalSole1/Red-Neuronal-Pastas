
# web/routes.py
"""
Rutas para la interfaz web
"""
import os
import uuid
from flask import (
    render_template, redirect, url_for, flash, 
    request, send_from_directory, current_app
)
from werkzeug.utils import secure_filename

from web import web_bp
from config.app_config import (
    TEMP_UPLOADS_DIR, STATIC_UPLOADS_DIR, allowed_file
)
from services.prediction_service import PredictionService
from services.class_service import ClassService
from core.data.preprocessing import convert_heic_to_jpg

# Inicializar servicios
prediction_service = PredictionService()
class_service = ClassService()

@web_bp.route('/')
def index():
    """Página principal de la aplicación web"""
    # Obtener información del modelo
    try:
        class_names = prediction_service.load_class_names()
    except:
        class_names = []
    
    # Información a mostrar en la página
    model_info = {
        'num_classes': len(class_names),
        'classes': class_names,
        'image_size': f"{current_app.config['IMG_HEIGHT']}x{current_app.config['IMG_WIDTH']}"
    }
    
    # Leer información del modelo si está disponible
    summary_path = os.path.join(current_app.config['OUTPUT_DIR'], 'model_summary.txt')
    if os.path.exists(summary_path):
        summary_data = {}
        try:
            with open(summary_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        summary_data[key.strip()] = value.strip()
            model_info['summary'] = summary_data
        except:
            pass
    
    return render_template('index.html', info=model_info)

@web_bp.route('/predict', methods=['GET', 'POST'])
def web_predict():
    """Página para realizar predicciones desde la web"""
    if request.method == 'POST':
        # Verificar si hay un archivo en la solicitud
        if 'file' not in request.files:
            flash('No se seleccionó ningún archivo')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No se seleccionó ningún archivo')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Crear nombre único para el archivo
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(TEMP_UPLOADS_DIR, filename)
            file.save(filepath)
            
            try:
                # Verificar si es un archivo HEIC/HEIF para convertirlo
                is_heic = filepath.lower().endswith(('.heic', '.heif'))
                
                if is_heic:
                    jpg_filepath = convert_heic_to_jpg(filepath)
                    if jpg_filepath:
                        filepath = jpg_filepath  # Usar el JPG para el resto del proceso
                    else:
                        flash('No se pudo convertir la imagen HEIC/HEIF')
                        return redirect(request.url)
                
                # Realizar predicción
                result = prediction_service.predict_image(filepath, top_k=3)
                
                # Guardar la imagen para mostrarla en los resultados
                display_filename = os.path.basename(filepath)
                if is_heic:
                    # Si era HEIC, usar el nombre del JPG convertido
                    display_filename = os.path.basename(filepath)
                
                # Copiar a directorio de subidas estáticas
                save_path = os.path.join(STATIC_UPLOADS_DIR, display_filename)
                os.system(f"cp {filepath} {save_path}")
                
                # Preparar datos para la plantilla
                prediction_data = {
                    'prediction': result['top_prediction']['class'],
                    'confidence': result['top_prediction']['confidence'] * 100,  # Convertir a porcentaje
                    'results': result['all_predictions'],
                    'image_path': f"uploads/{display_filename}"
                }
                
                return render_template('result.html', **prediction_data)
                
            except Exception as e:
                flash(f'Error en la predicción: {str(e)}')
                return redirect(request.url)
            finally:
                # Limpiar archivo temporal
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            flash('Tipo de archivo no permitido')
            return redirect(request.url)
    
    # GET request - mostrar formulario
    return render_template('predict.html')

@web_bp.route('/classes')
def web_classes():
    """Página que muestra todas las clases disponibles"""
    # Obtener estadísticas de clases
    stats = class_service.get_class_stats()
    
    # Preparar datos para la plantilla
    class_names = list(stats.keys())
    
    # Determinar si hay ejemplos de imágenes para cada clase
    class_examples = {}
    for class_name, info in stats.items():
        if info.get('path') and info.get('total_images', 0) > 0:
            # Hay imágenes para esta clase, buscar una para mostrar
            class_path = info['path']
            for ext in ['.jpg', '.jpeg', '.png']:
                for file in os.listdir(class_path):
                    if file.lower().endswith(ext):
                        # Tenemos una imagen de ejemplo
                        class_examples[class_name] = True
                        break
                if class_name in class_examples:
                    break
    
    return render_template('classes.html', classes=class_names, examples=class_examples)

@web_bp.route('/about')
def about():
    """Página con información sobre el proyecto"""
    return render_template('about.html')

@web_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Servir archivos subidos (imágenes)"""
    return send_from_directory('static/uploads', filename)

