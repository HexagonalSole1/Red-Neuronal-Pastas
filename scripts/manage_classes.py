# scripts/manage_classes.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gestionar clases del conjunto de datos
"""
import argparse
import sys
import os

from services.class_service import ClassService
from config.app_config import RAW_DATA_DIR

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='Gestión de clases para el clasificador')
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando list: listar clases
    list_parser = subparsers.add_parser('list', help='Listar clases disponibles')
    
    # Comando add: añadir clase
    add_parser = subparsers.add_parser('add', help='Añadir nueva clase')
    add_parser.add_argument('source_dir', help='Directorio con imágenes de la nueva clase')
    add_parser.add_argument('--name', help='Nombre para la nueva clase (por defecto, nombre del directorio)')
    
    # Comando remove: eliminar clase
    remove_parser = subparsers.add_parser('remove', help='Eliminar una clase existente')
    remove_parser.add_argument('class_name', help='Nombre de la clase a eliminar')
    
    # Comando rename: renombrar clase
    rename_parser = subparsers.add_parser('rename', help='Renombrar una clase existente')
    rename_parser.add_argument('old_name', help='Nombre actual de la clase')
    rename_parser.add_argument('new_name', help='Nuevo nombre para la clase')
    
    # Comando stats: mostrar estadísticas
    stats_parser = subparsers.add_parser('stats', help='Mostrar estadísticas de las clases')
    stats_parser.add_argument('--class-name', help='Nombre de la clase específica (opcional)')
    
    return parser.parse_args()

def main():
    """Función principal para gestión de clases"""
    args = parse_args()
    
    # Crear servicio de clases
    class_service = ClassService()
    
    # Si no se especificó comando, mostrar ayuda
    if not args.command:
        print("Error: Debe especificar un comando")
        print("Ejecute 'python manage_classes.py --help' para ver los comandos disponibles")
        return 1
    
    try:
        # Comando: list
        if args.command == 'list':
            classes = class_service.get_available_classes()
            
            if not classes:
                print("No hay clases disponibles")
                return 0
            
            print("\n===============================================================")
            print("   CLASES DISPONIBLES")
            print("===============================================================")
            for i, class_name in enumerate(classes, 1):
                print(f"{i}. {class_name}")
            print(f"\nTotal: {len(classes)} clases")
            
            return 0
        
        # Comando: add
        elif args.command == 'add':
            if not os.path.isdir(args.source_dir):
                print(f"Error: No se encontró el directorio {args.source_dir}")
                return 1
            
            result = class_service.add_class(args.source_dir, args.name)
            
            if result['success']:
                print(f"\n✅ Clase '{result['class_name']}' añadida correctamente")
                print(f"Se copiaron {result['copied_files']} imágenes a {result['destination']}")
                print("\nPara reentrenar el modelo con la nueva clase:")
                print("1. Ejecute: python scripts/train.py")
                print("2. El modelo se actualizará automáticamente para incluir la nueva clase")
                return 0
            else:
                print(f"\n❌ Error al añadir clase: {result['error']}")
                return 1
        
        # Comando: remove
        elif args.command == 'remove':
            result = class_service.remove_class(args.class_name)
            
            if result['success']:
                print(f"\n✅ {result['message']}")
                print("\nRecuerde reentrenar el modelo:")
                print("Ejecute: python scripts/train.py")
                return 0
            else:
                print(f"\n❌ Error al eliminar clase: {result['error']}")
                return 1
        
        # Comando: rename
        elif args.command == 'rename':
            result = class_service.rename_class(args.old_name, args.new_name)
            
            if result['success']:
                print(f"\n✅ {result['message']}")
                print("\nRecuerde reentrenar el modelo:")
                print("Ejecute: python scripts/train.py")
                return 0
            else:
                print(f"\n❌ Error al renombrar clase: {result['error']}")
                return 1
        
        # Comando: stats
        elif args.command == 'stats':
            stats = class_service.get_class_stats()
            
            if not stats:
                print("No hay clases disponibles")
                return 0
            
            print("\n===============================================================")
            print("   ESTADÍSTICAS DE CLASES")
            print("===============================================================")
            
            # Si se especificó una clase concreta
            if args.class_name:
                if args.class_name not in stats:
                    print(f"Error: La clase '{args.class_name}' no existe")
                    return 1
                
                info = stats[args.class_name]
                print(f"Clase: {args.class_name}")
                print(f"Total de imágenes: {info['total_images']}")
                print(f"Imágenes JPG/JPEG: {info['jpg_count']}")
                print(f"Imágenes PNG: {info['png_count']}")
                print(f"Imágenes HEIC/HEIF: {info['heic_count']}")
                if info.get('path'):
                    print(f"Ruta: {info['path']}")
                if info.get('error'):
                    print(f"Error: {info['error']}")
            else:
                # Mostrar todas las clases
                print(f"{'Clase':<30} {'Total':<10} {'JPG':<10} {'PNG':<10} {'HEIC':<10}")
                print("-" * 70)
                
                for class_name, info in stats.items():
                    print(f"{class_name:<30} {info['total_images']:<10} {info['jpg_count']:<10} {info['png_count']:<10} {info['heic_count']:<10}")
                
                print("\nTotal de clases:", len(stats))
                
                # Mostrar total de imágenes
                total_images = sum(info['total_images'] for info in stats.values())
                print(f"Total de imágenes: {total_images}")
            
            return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())