# scripts/train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para entrenar un nuevo modelo de clasificación
"""
import argparse
import sys
import os

from services.training_service import TrainingService
from config.app_config import RAW_DATA_DIR
from config.model_config import EPOCHS, BATCH_SIZE, K_FOLDS, LEARNING_RATE

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo de clasificación')
    parser.add_argument('--data-dir', default=RAW_DATA_DIR, 
                        help=f'Directorio con datos de entrenamiento (default: {RAW_DATA_DIR})')
    parser.add_argument('--epochs', type=int, default=EPOCHS, 
                        help=f'Número de épocas (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, 
                        help=f'Tamaño de lote (default: {BATCH_SIZE})')
    parser.add_argument('--k-folds', type=int, default=K_FOLDS, 
                        help=f'Número de folds para validación cruzada (default: {K_FOLDS})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, 
                        help=f'Tasa de aprendizaje (default: {LEARNING_RATE})')
    parser.add_argument('--model-path', default=None, 
                        help='Ruta para guardar el modelo (default: models/best_model.h5)')
    
    return parser.parse_args()

def main():
    """Función principal de entrenamiento"""
    args = parse_args()
    
    # Verificar que el directorio de datos exista
    if not os.path.isdir(args.data_dir):
        print(f"Error: No se encontró el directorio de datos {args.data_dir}")
        return 1
    
    # Verificar si hay clases (subdirectorios) en el directorio de datos
    subdirs = [d for d in os.listdir(args.data_dir) 
              if os.path.isdir(os.path.join(args.data_dir, d))]
    
    if not subdirs:
        print(f"Error: No hay clases (subdirectorios) en {args.data_dir}")
        print("Debe crear al menos 2 carpetas, una para cada clase")
        return 1
    
    print("\n===============================================================")
    print("   ENTRENAMIENTO DE MODELO DE CLASIFICACIÓN DE ALIMENTOS")
    print("===============================================================")
    print(f"Directorio de datos: {args.data_dir}")
    print(f"Clases encontradas: {len(subdirs)}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Validación cruzada: {args.k_folds} folds")
    print(f"Learning rate: {args.learning_rate}")
    print("===============================================================\n")
    
    # Crear servicio de entrenamiento
    training_service = TrainingService(data_dir=args.data_dir)
    
    try:
        # Iniciar entrenamiento
        print("Iniciando entrenamiento...\n")
        results = training_service.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            k_folds=args.k_folds,
            learning_rate=args.learning_rate,
            save_model_path=args.model_path
        )
        
        # Mostrar resultados
        print("\n===============================================================")
        print("   RESULTADOS DEL ENTRENAMIENTO")
        print("===============================================================")
        print(f"Precisión promedio: {results['accuracy']:.4f}")
        print(f"Desviación estándar: {results['std_accuracy']:.4f}")
        print(f"Número de clases: {results['num_classes']}")
        print(f"Clases: {', '.join(results['classes'])}")
        print(f"Modelo guardado en: {results['model_path']}")
        print(f"Nombres de clases guardados en: {results['class_names_path']}")
        print("===============================================================\n")
        
        print("Entrenamiento completado con éxito.")
        print("Para iniciar el servidor y realizar predicciones ejecute:")
        print("  python app.py")
        
        return 0
    
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())