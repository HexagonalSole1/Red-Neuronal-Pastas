#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controller: Define el servidor Flask y las rutas
"""

import os
from flask import Flask, render_template, flash, request, redirect, url_for
import logging

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'  # Cambia esto por una clave segura

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Página principal"""
    # Intentamos obtener las clases del modelo (si existe)
    try:
        # En un caso real, cargaríamos esto desde el modelo
        classes = ['Bucatini', 'Conchiglie', 'Espagueti', 'Estrellitas', 
                  'Farfalle', 'Fettuccine', 'Fideos', 'Fusilli', 
                  'Lasagna', 'Macarrones', 'Orecchiette', 'Penne', 'Rigatoni']
        
        model_info = {
            'num_classes': len(classes),
            'classes': classes,
            'image_size': "224x224",
            'summary': {'Precisión promedio': '0.9968'}
        }
    except Exception as e:
        logger.error(f"Error al cargar clases: {e}")
        model_info = {
            'num_classes': 0,
            'classes': [],
            'image_size': "224x224"
        }
    
    return render_template('index.html', info=model_info)

@app.route('/predict', methods=['GET', 'POST'])
def web_predict():
    """Página para subir imagen y hacer predicción"""
    if request.method == 'POST':
        # Aquí iría la lógica de predicción
        flash('Funcionalidad de predicción no implementada aún')
        return redirect(url_for('index'))
    return render_template('predict.html')

@app.route('/classes')
def web_classes():
    """Página que muestra las clases disponibles"""
    # En un caso real, cargaríamos esto desde el modelo
    classes = ['Bucatini', 'Conchiglie', 'Espagueti', 'Estrellitas', 
              'Farfalle', 'Fettuccine', 'Fideos', 'Fusilli', 
              'Lasagna', 'Macarrones', 'Orecchiette', 'Penne', 'Rigatoni']
    
    return render_template('classes.html', classes=classes, examples={})

@app.route('/about')
def about():
    """Página con información sobre el proyecto"""
    return render_template('about.html')

@app.route('/api/info')
def api_info():
    """Endpoint API para información del modelo"""
    return {
        'status': 'ok',
        'model': {
            'name': 'Clasificador de Pastas',
            'num_classes': 13,
            'classes': ['Bucatini', 'Conchiglie', 'Espagueti', 'Estrellitas', 
                       'Farfalle', 'Fettuccine', 'Fideos', 'Fusilli', 
                       'Lasagna', 'Macarrones', 'Orecchiette', 'Penne', 'Rigatoni'],
            'version': '1.0.0'
        }
    }

@app.route('/api/health')
def api_health():
    """Endpoint API para verificar salud del servidor"""
    return {'status': 'ok', 'message': 'Servidor operativo'}

def run_server(custom_host='0.0.0.0', custom_port=5000, debug=False):
    """
    Inicia el servidor Flask
    
    Args:
        custom_host: Host para escuchar (default: 0.0.0.0)
        custom_port: Puerto para escuchar (default: 5000)
        debug: Modo de depuración (default: False)
    """
    try:
        app.run(host=custom_host, port=custom_port, debug=debug)
    except Exception as e:
        logger.error(f"Error al iniciar el servidor: {e}")
        raise