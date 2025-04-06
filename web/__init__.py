# web/__init__.py
"""
Inicialización del módulo de interfaz web
"""
from flask import Blueprint

# Crear blueprint para la web
web_bp = Blueprint('web', __name__)

# Importar rutas
from web.routes import *  # noqa
