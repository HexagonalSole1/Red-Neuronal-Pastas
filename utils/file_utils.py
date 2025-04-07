# utils/file_utils.py
"""
Utilidades para manejo de archivos
"""
import os
import glob
import sys
import subprocess
from PIL import Image

def convert_heic_to_jpg(data_dir):
    """
    Convierte imágenes HEIC a JPG en un directorio y subdirectorios
    
    Args:
        data_dir: Directorio con imágenes HEIC
    """
    print("Buscando imágenes HEIC para convertir...")
    
    # Buscamos todos los archivos HEIC recursivamente
    heic_files = []
    for root, _, _ in os.walk(data_dir):
        heic_files.extend(glob.glob(os.path.join(root, "*.heic")))
        heic_files.extend(glob.glob(os.path.join(root, "*.HEIC")))
    
    if not heic_files:
        print("No se encontraron archivos HEIC para convertir.")
        return
    
    print(f"Se encontraron {len(heic_files)} archivos HEIC.")
    
    # Método 1: Intentar con pillow-heif
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        
        print("Usando pillow-heif para convertir imágenes...")
        
        for heic_file in heic_files:
            try:
                # Abrimos el archivo HEIC con PIL (ahora soporta HEIC gracias a pillow_heif)
                img = Image.open(heic_file)
                
                # Guardamos como JPG
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                img.save(jpg_file, "JPEG")
                print(f"Convertido: {heic_file} -> {jpg_file}")
                
            except Exception as e:
                print(f"Error al convertir {heic_file} con pillow-heif: {e}")
        
        return
        
    except ImportError:
        print("pillow-heif no disponible, intentando método alternativo...")
    
    # Método 2: Intentar con sips (solo MacOS)
    if sys.platform == 'darwin':  # MacOS
        print("Intentando convertir con sips (herramienta nativa de MacOS)...")
        
        for heic_file in heic_files:
            try:
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                subprocess.run(['sips', '-s', 'format', 'jpeg', heic_file, '--out', jpg_file], 
                              check=True, stdout=subprocess.PIPE)
                print(f"Convertido: {heic_file} -> {jpg_file}")
            except Exception as e:
                print(f"Error al convertir {heic_file} con sips: {e}")
        
        return
    
    # Método 3: Usar ImageMagick si está disponible
    try:
        subprocess.run(['convert', '--version'], check=True, stdout=subprocess.PIPE)
        print("Intentando convertir con ImageMagick...")
        
        for heic_file in heic_files:
            try:
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                subprocess.run(['convert', heic_file, jpg_file], check=True, stdout=subprocess.PIPE)
                print(f"Convertido: {heic_file} -> {jpg_file}")
            except Exception as e:
                print(f"Error al convertir {heic_file} con ImageMagick: {e}")
        
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ImageMagick no disponible...")
    
    print("\n¡IMPORTANTE! No se pudieron convertir los archivos HEIC a JPG.")
    print("Por favor, convierte manualmente los archivos HEIC a JPG antes de continuar.")
    print("También puedes instalar 'pillow-heif' con pip para habilitar la conversión automática.")

def ensure_dir(directory):
    """
    Asegura que un directorio exista, creándolo si es necesario
    
    Args:
        directory: Ruta del directorio
        
    Returns:
        bool: True si el directorio existe o fue creado
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return True
    return os.path.isdir(directory)

def clean_directory(directory, extensions=None):
    """
    Limpia un directorio, eliminando archivos con las extensiones especificadas
    
    Args:
        directory: Directorio a limpiar
        extensions: Lista de extensiones a eliminar (si es None, elimina todos)
        
    Returns:
        int: Número de archivos eliminados
    """
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            if extensions is None or any(item.lower().endswith(ext.lower()) for ext in extensions):
                try:
                    os.remove(item_path)
                    count += 1
                except:
                    pass
    
    return count