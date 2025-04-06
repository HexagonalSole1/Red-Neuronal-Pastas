# core/utils/file_utils.py
"""
Utilidades para manejo de archivos y directorios
"""
import os
import glob
import sys
import subprocess
from PIL import Image
import shutil

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

def convert_all_heic_in_directory(data_dir):
    """
    Convierte todas las imágenes HEIC a JPG en un directorio y sus subdirectorios
    
    Args:
        data_dir: Directorio a procesar
        
    Returns:
        int: Número de archivos convertidos
    """
    print("Buscando imágenes HEIC para convertir...")
    
    # Buscar todos los archivos HEIC en el directorio y subdirectorios
    heic_files = []
    for folder, _, _ in os.walk(data_dir):
        heic_files.extend(glob.glob(os.path.join(folder, "*.heic")))
        heic_files.extend(glob.glob(os.path.join(folder, "*.HEIC")))
        heic_files.extend(glob.glob(os.path.join(folder, "*.heif")))
        heic_files.extend(glob.glob(os.path.join(folder, "*.HEIF")))
    
    if not heic_files:
        print("No se encontraron archivos HEIC/HEIF para convertir.")
        return 0
    
    print(f"Se encontraron {len(heic_files)} archivos HEIC/HEIF.")
    
    # Método 1: Intentar con pillow-heif
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        
        print("Usando pillow-heif para convertir imágenes...")
        
        converted = 0
        for heic_file in heic_files:
            try:
                # Abrir y guardar como JPG
                img = Image.open(heic_file)
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                img.save(jpg_file, "JPEG")
                print(f"Convertido: {heic_file} -> {jpg_file}")
                converted += 1
            except Exception as e:
                print(f"Error al convertir {heic_file} con pillow-heif: {e}")
        
        return converted
        
    except ImportError:
        print("pillow-heif no disponible, intentando método alternativo...")
    
    # Método 2: Intentar con sips (solo MacOS)
    if sys.platform == 'darwin':  # MacOS
        print("Intentando convertir con sips (herramienta nativa de MacOS)...")
        
        converted = 0
        for heic_file in heic_files:
            try:
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                subprocess.run(['sips', '-s', 'format', 'jpeg', heic_file, '--out', jpg_file], 
                              check=True, stdout=subprocess.PIPE)
                print(f"Convertido: {heic_file} -> {jpg_file}")
                converted += 1
            except Exception as e:
                print(f"Error al convertir {heic_file} con sips: {e}")
        
        return converted
    
    # Método 3: Usar ImageMagick si está disponible
    try:
        subprocess.run(['convert', '--version'], check=True, stdout=subprocess.PIPE)
        print("Intentando convertir con ImageMagick...")
        
        converted = 0
        for heic_file in heic_files:
            try:
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                subprocess.run(['convert', heic_file, jpg_file], check=True, stdout=subprocess.PIPE)
                print(f"Convertido: {heic_file} -> {jpg_file}")
                converted += 1
            except Exception as e:
                print(f"Error al convertir {heic_file} con ImageMagick: {e}")
        
        return converted
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ImageMagick no disponible...")
    
    # Método 4: Intentar con pyheif
    try:
        import pyheif
        print("Intentando convertir con pyheif...")
        
        converted = 0
        for heic_file in heic_files:
            try:
                # Leer archivo HEIC
                heif_file = pyheif.read(heic_file)
                
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
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                img.save(jpg_file, "JPEG")
                print(f"Convertido: {heic_file} -> {jpg_file}")
                converted += 1
            except Exception as e:
                print(f"Error al convertir {heic_file} con pyheif: {e}")
        
        return converted
        
    except ImportError:
        print("pyheif no disponible...")
    
    print("\n⚠️ IMPORTANTE: No se pudieron convertir los archivos HEIC/HEIF a JPG.")
    print("Por favor, convierta manualmente los archivos HEIC/HEIF a JPG antes de continuar.")
    print("También puede usar herramientas como 'sips' en MacOS o aplicaciones como iPhoto, Preview, etc.")
    
    return 0

def get_file_path_with_extension(base_path, extensions):
    """
    Busca un archivo con una de las extensiones especificadas
    
    Args:
        base_path: Ruta base (sin extensión)
        extensions: Lista de extensiones a probar
        
    Returns:
        str: Ruta completa si encuentra el archivo, None en caso contrario
    """
    for ext in extensions:
        full_path = f"{base_path}{ext}"
        if os.path.exists(full_path):
            return full_path
    return None

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

def create_backup(source_dir, backup_dir=None):
    """
    Crea una copia de seguridad de un directorio
    
    Args:
        source_dir: Directorio a respaldar
        backup_dir: Directorio de respaldo (si es None, se crea uno con sufijo _backup)
        
    Returns:
        str: Ruta del directorio de respaldo
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directorio origen no encontrado: {source_dir}")
    
    # Determinar directorio de respaldo
    if backup_dir is None:
        backup_dir = f"{source_dir}_backup"
    
    # Crear directorio de respaldo si no existe
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    # Copiar contenido
    shutil.copytree(source_dir, backup_dir)
    
    print(f"Respaldo creado en: {backup_dir}")
    return backup_dir
