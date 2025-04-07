#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar el entrenamiento real con las importaciones correctas
"""
import os
import sys
import argparse

# Añadir el directorio raíz al path de Python
directorio_raiz = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, directorio_raiz)

# Crear directorio services si no existe
services_dir = os.path.join(directorio_raiz, 'services')
if not os.path.exists(services_dir):
    os.makedirs(services_dir)
    print(f"✅ Directorio 'services' creado")

# Asegurar que existe el archivo __init__.py en services
services_init = os.path.join(services_dir, '__init__.py')
if not os.path.exists(services_init):
    with open(services_init, 'w') as f:
        f.write('# services/__init__.py\n"""Inicialización del módulo de servicios"""\n')
    print(f"✅ Archivo services/__init__.py creado")

# Crear directorios de datos si no existen
data_dirs = ['data', 'data/raw', 'data/entrenamiento', 'data/prueba', 'output', 'models']
for dir_path in data_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Importar el servicio de entrenamiento
from services.training_service import TrainingService

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo de clasificación')
    parser.add_argument('--data-dir', default='data/raw', 
                        help='Directorio con datos de entrenamiento (default: data/raw)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Número de épocas (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Tamaño de lote (default: 32)')
    parser.add_argument('--k-folds', type=int, default=5, 
                        help='Número de folds para validación cruzada (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=0.001, 
                        help='Tasa de aprendizaje (default: 0.001)')
    parser.add_argument('--model-path', default=None, 
                        help='Ruta para guardar el modelo (default: models/best_model.h5)')
    
    return parser.parse_args()

def main():
    """Función principal de entrenamiento"""
    args = parse_args()
    
    # Verificar que el directorio de datos exista
    if not os.path.isdir(args.data_dir):
        print(f"⚠️ Advertencia: No se encontró el directorio de datos {args.data_dir}")
        print(f"Se creará el directorio. Por favor, asegúrate de colocar tus imágenes organizadas en subcarpetas por clase.")
        os.makedirs(args.data_dir, exist_ok=True)
        print("Ejemplo: data/raw/clase1/, data/raw/clase2/, etc.")
        return 1
    
    # Verificar si hay clases (subdirectorios) en el directorio de datos
    subdirs = [d for d in os.listdir(args.data_dir) 
              if os.path.isdir(os.path.join(args.data_dir, d))]
    
    if not subdirs:
        print(f"⚠️ Advertencia: No hay clases (subdirectorios) en {args.data_dir}")
        print("Debes crear al menos 2 carpetas, una para cada clase")
        print("Ejemplo: data/raw/clase1/, data/raw/clase2/, etc.")
        return 1
    
    print("\n" + "="*80)
    print("   ENTRENAMIENTO DE MODELO DE CLASIFICACIÓN DE IMÁGENES   ".center(80, "="))
    print("="*80)
    print(f"Directorio de datos: {args.data_dir}")
    print(f"Clases encontradas: {len(subdirs)}")
    print(f"Nombres de clases: {', '.join(subdirs)}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Validación cruzada: {args.k_folds} folds")
    print(f"Learning rate: {args.learning_rate}")
    print("="*80 + "\n")
    
    # Crear servicio de entrenamiento
    training_service = TrainingService(data_dir=args.data_dir)
    
    try:
        # Iniciar entrenamiento real
        print("🚀 Iniciando entrenamiento real...\n")
        results = training_service.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            k_folds=args.k_folds,
            learning_rate=args.learning_rate,
            save_model_path=args.model_path
        )
        
        # Mostrar resultados
        print("\n" + "="*80)
        print("   RESULTADOS DEL ENTRENAMIENTO   ".center(80, "="))
        print("="*80)
        print(f"Precisión promedio: {results['accuracy']:.4f}")
        print(f"Desviación estándar: {results['std_accuracy']:.4f}")
        print(f"Número de clases: {results['num_classes']}")
        print(f"Clases: {', '.join(results['classes'])}")
        print(f"Modelo guardado en: {results['model_path']}")
        print(f"Nombres de clases guardados en: {results['class_names_path']}")
        print("="*80 + "\n")
        
        print("🎉 Entrenamiento completado con éxito.")
        print("Para iniciar el servidor y realizar predicciones ejecuta:")
        print("  python app.py")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())