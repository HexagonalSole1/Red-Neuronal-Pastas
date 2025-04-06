# api/__init__.py
"""
Inicialización del módulo de API REST
"""
from flask import Blueprint

# Crear blueprint para la API
api_bp = Blueprint('api', __name__)

# Importar rutas
from api.routes import *  # noqa