# core/data/preprocessing.py
"""
Funciones para preprocesamiento de imágenes
"""
import os
import sys
import subprocess
from PIL import Image
import numpy as np

def convert_heic_to_jpg(file_path):
    """
    Convierte una imagen HEIC a formato JPG
    
    Args:
        file_path: Ruta al archivo HEIC
        
    Returns:
        Ruta al archivo JPG resultante o None si falla
    """
    # Verificar que el archivo exista
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    # Verificar que sea un archivo HEIC
    if not file_path.lower().endswith(('.heic', '.heif')):
        raise ValueError(f"El archivo {file_path} no es un archivo HEIC/HEIF")
    
    # Ruta para el archivo JPG resultante
    jpg_path = os.path.splitext(file_path)[0] + ".jpg"
    
    # Intentar diferentes métodos para convertir
    try:
        # Método 1: Usar pillow_heif si está disponible
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            
            # Abrir y guardar como JPG
            img = Image.open(file_path).convert('RGB')
            img.save(jpg_path, "JPEG")
            return jpg_path
            
        except ImportError:
            # Método 2: Intentar con pyheif
            try:
                import pyheif
                
                # Leer el archivo HEIC
                heif_file = pyheif.read(file_path)
                
                # Convertir a PIL Image
                img = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data,
                    "raw", 
                    heif_file.mode, 
                    heif_file.stride,
                )
                
                # Guardar como JPG
                img.save(jpg_path, "JPEG")
                return jpg_path
                
            except ImportError:
                # Método 3: Usar herramientas del sistema
                if sys.platform == 'darwin':  # MacOS
                    try:
                        subprocess.run(
                            ['sips', '-s', 'format', 'jpeg', file_path, '--out', jpg_path],
                            check=True, 
                            stdout=subprocess.PIPE
                        )
                        return jpg_path
                    except (subprocess.SubprocessError, OSError):
                        pass
                
                # Método 4: Intentar con ImageMagick
                try:
                    subprocess.run(
                        ['convert', file_path, jpg_path],
                        check=True, 
                        stdout=subprocess.PIPE
                    )
                    return jpg_path
                except (subprocess.SubprocessError, OSError):
                    pass
    
    except Exception as e:
        print(f"Error al convertir HEIC a JPG: {e}")
    
    return None

def resize_and_normalize_image(img, target_size=(224, 224)):
    """
    Redimensiona y normaliza una imagen PIL
    
    Args:
        img: Objeto PIL Image
        target_size: Tamaño objetivo (ancho, alto)
        
    Returns:
        Array numpy normalizado [0-1]
    """
    # Asegurar que la imagen está en formato RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar
    img = img.resize(target_size)
    
    # Convertir a array y normalizar
    img_array = np.array(img) / 255.0
    
    return img_array