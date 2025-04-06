# app.py
"""
Punto de entrada principal para la aplicación de clasificación de alimentos
Incluye servidor API REST y interfaz web
"""
import os
import argparse
import sys
from flask import Flask

from config.app_config import (
    APP_NAME, HOST, PORT, SECRET_KEY, TEMP_UPLOADS_DIR, 
    STATIC_UPLOADS_DIR, DEFAULT_MODEL_PATH, CLASS_NAMES_PATH, 
    OUTPUT_DIR, ensure_directories
)
from api import api_bp
from web import web_bp

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(
        description='Servidor API y Web para el clasificador de alimentos'
    )
    parser.add_argument(
        '--port', type=int, default=PORT, 
        help=f'Puerto del servidor (default: {PORT})'
    )
    parser.add_argument(
        '--host', default=HOST, 
        help=f'Host del servidor (default: {HOST})'
    )
    parser.add_argument(
        '--debug', action='store_true', 
        help='Ejecutar el servidor en modo debug'
    )
    return parser.parse_args()

def create_app():
    """
    Crea y configura la aplicación Flask
    
    Returns:
        app: Aplicación Flask configurada
    """
    app = Flask(__name__)
    
    # Configuración
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    app.config['APP_NAME'] = APP_NAME
    
    # Directorios de la aplicación
    app.config['TEMP_UPLOADS_DIR'] = TEMP_UPLOADS_DIR
    app.config['STATIC_UPLOADS_DIR'] = STATIC_UPLOADS_DIR
    app.config['DEFAULT_MODEL_PATH'] = DEFAULT_MODEL_PATH
    app.config['CLASS_NAMES_PATH'] = CLASS_NAMES_PATH
    app.config['OUTPUT_DIR'] = OUTPUT_DIR
    
    # Configuración de imágenes
    from config.model_config import IMG_HEIGHT, IMG_WIDTH
    app.config['IMG_HEIGHT'] = IMG_HEIGHT
    app.config['IMG_WIDTH'] = IMG_WIDTH
    
    # Registrar blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(web_bp)
    
    return app

def check_model_files():
    """
    Verifica si existen los archivos del modelo
    
    Returns:
        bool: True si los archivos existen
    """
    model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    class_names_exist = os.path.exists(CLASS_NAMES_PATH)
    
    if not model_exists:
        print(f"⚠️ Advertencia: No se encontró el modelo en {DEFAULT_MODEL_PATH}")
    
    if not class_names_exist:
        print(f"⚠️ Advertencia: No se encontró el archivo de clases en {CLASS_NAMES_PATH}")
    
    return model_exists and class_names_exist

def main():
    """Función principal"""
    # Analizar argumentos
    args = parse_args()
    
    # Asegurar que existan los directorios necesarios
    ensure_directories()
    
    # Verificar archivos del modelo
    if not check_model_files():
        print("\n⚠️ No se encontraron los archivos del modelo.")
        print("Por favor, entrene el modelo primero ejecutando:")
        print("   python scripts/train.py")
        print("\nEl servidor se iniciará, pero no podrá realizar predicciones hasta que entrene el modelo.")
    
    # Crear la aplicación
    app = create_app()
    
    # Mostrar información de inicio
    print("\n" + "="*80)
    print(f" {APP_NAME} - Servidor API y Web ".center(80, "="))
    print("="*80)
    print(f"📡 API REST: http://{args.host}:{args.port}/api")
    print(f"🌐 Interfaz Web: http://{args.host}:{args.port}/")
    print(f"🧪 Endpoints API disponibles:")
    print(f"   - GET  /api/info")
    print(f"   - POST /api/predict")
    print(f"   - GET  /api/classes")
    print(f"   - GET  /api/health")
    print(f"   - GET  /api/model_status")
    print("="*80)
    
    # Iniciar el servidor
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido por el usuario")
        return 0
    except Exception as e:
        print(f"\n❌ Error al iniciar el servidor: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())