
# scripts/predict.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para hacer predicciones con el modelo entrenado
"""
import argparse
import sys
import os

from services.prediction_service import PredictionService
from config.app_config import DEFAULT_MODEL_PATH, CLASS_NAMES_PATH, OUTPUT_DIR

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='Predicción con el modelo entrenado')
    parser.add_argument('image_path', help='Ruta a la imagen para predecir')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, 
                        help=f'Ruta al modelo guardado (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--classes', default=CLASS_NAMES_PATH, 
                        help=f'Ruta a los nombres de clases (default: {CLASS_NAMES_PATH})')
    parser.add_argument('--visualize', action='store_true', 
                        help='Mostrar la imagen con la predicción')
    parser.add_argument('--output', default=None, 
                        help='Ruta para guardar la visualización (default: output/predictions/)')
    
    return parser.parse_args()

def main():
    """Función principal para predicción"""
    args = parse_args()
    
    # Verificar que la imagen exista
    if not os.path.exists(args.image_path):
        print(f"Error: No se encontró la imagen en {args.image_path}")
        return 1
    
    # Verificar que el modelo exista
    if not os.path.exists(args.model):
        print(f"Error: No se encontró el modelo en {args.model}")
        return 1
    
    # Verificar que el archivo de clases exista
    if not os.path.exists(args.classes):
        print(f"Error: No se encontró el archivo de clases en {args.classes}")
        return 1
    
    # Determinar ruta de salida para visualización
    output_path = args.output
    if args.visualize and not output_path:
        output_dir = os.path.join(OUTPUT_DIR, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, 
            f"pred_{os.path.basename(args.image_path).split('.')[0]}.png"
        )
    
    try:
        # Crear servicio de predicción
        prediction_service = PredictionService(
            model_path=args.model, 
            class_names_path=args.classes
        )
        
        # Realizar predicción
        result = prediction_service.predict_image(
            args.image_path, 
            visualize=args.visualize, 
            output_path=output_path
        )
        
        # Mostrar resultados
        print("\n===============================================================")
        print("   RESULTADO DE LA PREDICCIÓN")
        print("===============================================================")
        print(f"Clase predicha: {result['top_prediction']['class']}")
        print(f"Confianza: {result['top_prediction']['confidence']:.2%}")
        print("===============================================================")
        
        # Mostrar otras predicciones si hay más de una
        if len(result['all_predictions']) > 1:
            print("\nOtras posibilidades:")
            for pred in result['all_predictions'][1:]:
                print(f"  - {pred['class']}: {pred['confidence']:.2%}")
        
        # Mostrar ruta de la visualización si se guardó
        if result.get('visualization_path'):
            print(f"\nVisualización guardada en: {result['visualization_path']}")
        
        return 0
    
    except Exception as e:
        print(f"\nError durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
