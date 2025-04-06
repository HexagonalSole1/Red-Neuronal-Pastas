# config/app_config.py
"""
Configuración centralizada de la aplicación
"""
import os

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
TEMP_UPLOADS_DIR = os.path.join(BASE_DIR, 'temp_uploads')
STATIC_UPLOADS_DIR = os.path.join(BASE_DIR, 'static', 'uploads')

# Configuración de la aplicación web
APP_NAME = "Food Classification System"
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_secret_key_12345')
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic', 'heif'}

# Configuración para almacenamiento de archivos
def allowed_file(filename):
    """Verifica si la extensión del archivo es permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rutas de archivos importantes
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.h5')
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, 'class_names.txt')
MODEL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'model_summary.txt')

# Garantizar que existan los directorios necesarios
def ensure_directories():
    """Crea los directorios necesarios si no existen"""
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUT_DIR,
        TEMP_UPLOADS_DIR,
        STATIC_UPLOADS_DIR
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    return True
