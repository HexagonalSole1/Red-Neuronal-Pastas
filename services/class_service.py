# services/class_service.py
"""
Servicio para gestionar clases en el conjunto de datos
"""
import os
import shutil
import glob

from config.app_config import RAW_DATA_DIR, CLASS_NAMES_PATH
from core.utils.file_utils import ensure_dir

class ClassService:
    """Servicio para gestionar clases del conjunto de datos"""
    
    def __init__(self, raw_dir=RAW_DATA_DIR, class_names_path=CLASS_NAMES_PATH):
        """
        Inicializa el servicio de gestión de clases
        
        Args:
            raw_dir: Directorio raw con las imágenes de entrenamiento
            class_names_path: Ruta al archivo con nombres de clases
        """
        self.raw_dir = raw_dir
        self.class_names_path = class_names_path
        ensure_dir(raw_dir)
    
    def get_available_classes(self):
        """
        Obtiene la lista de clases disponibles
        
        Returns:
            Lista de nombres de clases
        """
        # Verificar si el archivo de clases existe
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # Si no existe, buscar directorios en raw_dir
        if os.path.exists(self.raw_dir):
            return [d for d in os.listdir(self.raw_dir) 
                   if os.path.isdir(os.path.join(self.raw_dir, d))]
        
        return []
    
    def get_class_stats(self):
        """
        Obtiene estadísticas de las clases existentes
        
        Returns:
            Diccionario con estadísticas por clase
        """
        stats = {}
        
        # Obtener las clases disponibles
        classes = self.get_available_classes()
        
        for class_name in classes:
            class_dir = os.path.join(self.raw_dir, class_name)
            
            if os.path.isdir(class_dir):
                # Contar imágenes por tipo
                jpg_count = len(glob.glob(os.path.join(class_dir, "*.jpg"))) + \
                           len(glob.glob(os.path.join(class_dir, "*.jpeg"))) + \
                           len(glob.glob(os.path.join(class_dir, "*.JPG"))) + \
                           len(glob.glob(os.path.join(class_dir, "*.JPEG")))
                
                png_count = len(glob.glob(os.path.join(class_dir, "*.png"))) + \
                           len(glob.glob(os.path.join(class_dir, "*.PNG")))
                
                heic_count = len(glob.glob(os.path.join(class_dir, "*.heic"))) + \
                            len(glob.glob(os.path.join(class_dir, "*.HEIC"))) + \
                            len(glob.glob(os.path.join(class_dir, "*.heif"))) + \
                            len(glob.glob(os.path.join(class_dir, "*.HEIF")))
                
                total_images = jpg_count + png_count + heic_count
                
                # Guardar estadísticas
                stats[class_name] = {
                    'total_images': total_images,
                    'jpg_count': jpg_count,
                    'png_count': png_count,
                    'heic_count': heic_count,
                    'path': class_dir
                }
            else:
                # La clase está en el archivo pero no tiene directorio
                stats[class_name] = {
                    'total_images': 0,
                    'jpg_count': 0,
                    'png_count': 0,
                    'heic_count': 0,
                    'path': None,
                    'error': 'Directorio no encontrado'
                }
        
        return stats
    
    def add_class(self, source_dir, class_name=None):
        """
        Añade una nueva clase al conjunto de datos
        
        Args:
            source_dir: Directorio con imágenes de la nueva clase
            class_name: Nombre de la clase (si es None, se usa el nombre del directorio)
            
        Returns:
            Diccionario con información de la operación
        """
        # Verificar que el directorio origen exista
        if not os.path.isdir(source_dir):
            return {'success': False, 'error': f"El directorio {source_dir} no existe"}
        
        # Determinar el nombre de la clase
        if class_name is None:
            class_name = os.path.basename(source_dir)
        
        # Verificar que el nombre de clase sea válido
        if not class_name or class_name.startswith('.'):
            return {'success': False, 'error': "Nombre de clase inválido"}
        
        # Crear directorio destino
        dest_dir = os.path.join(self.raw_dir, class_name)
        
        # Verificar si la clase ya existe
        if os.path.exists(dest_dir):
            return {'success': False, 'error': f"La clase '{class_name}' ya existe"}
        
        try:
            # Crear directorio destino
            ensure_dir(dest_dir)
            
            # Contar imágenes en origen
            image_files = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', 
                      '*.heic', '*.HEIC', '*.heif', '*.HEIF']:
                image_files.extend(glob.glob(os.path.join(source_dir, ext)))
            
            if not image_files:
                # Eliminar directorio si no hay imágenes
                os.rmdir(dest_dir)
                return {'success': False, 'error': "No se encontraron imágenes en el directorio origen"}
            
            # Copiar todas las imágenes
            copied_files = 0
            for img_file in image_files:
                shutil.copy2(img_file, dest_dir)
                copied_files += 1
            
            return {
                'success': True, 
                'class_name': class_name,
                'copied_files': copied_files,
                'destination': dest_dir
            }
            
        except Exception as e:
            # Intentar limpiar en caso de error
            if os.path.exists(dest_dir):
                try:
                    shutil.rmtree(dest_dir)
                except:
                    pass
            
            return {'success': False, 'error': str(e)}
    
    def remove_class(self, class_name):
        """
        Elimina una clase del conjunto de datos
        
        Args:
            class_name: Nombre de la clase a eliminar
            
        Returns:
            Diccionario con información de la operación
        """
        # Verificar que el nombre de clase sea válido
        if not class_name:
            return {'success': False, 'error': "Nombre de clase inválido"}
        
        # Verificar que la clase exista
        class_dir = os.path.join(self.raw_dir, class_name)
        if not os.path.isdir(class_dir):
            return {'success': False, 'error': f"La clase '{class_name}' no existe"}
        
        try:
            # Eliminar el directorio
            shutil.rmtree(class_dir)
            
            # Actualizar el archivo de clases si existe
            if os.path.exists(self.class_names_path):
                with open(self.class_names_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                
                # Eliminar la clase del archivo
                if class_name in classes:
                    classes.remove(class_name)
                    
                    # Guardar el archivo actualizado
                    with open(self.class_names_path, 'w') as f:
                        for cls in classes:
                            f.write(f"{cls}\n")
            
            return {
                'success': True,
                'class_name': class_name,
                'message': f"Clase '{class_name}' eliminada correctamente"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def rename_class(self, old_name, new_name):
        """
        Renombra una clase existente
        
        Args:
            old_name: Nombre actual de la clase
            new_name: Nuevo nombre para la clase
            
        Returns:
            Diccionario con información de la operación
        """
        # Verificar que los nombres sean válidos
        if not old_name or not new_name:
            return {'success': False, 'error': "Nombres de clase inválidos"}
        
        # Verificar que la clase origen exista
        old_dir = os.path.join(self.raw_dir, old_name)
        if not os.path.isdir(old_dir):
            return {'success': False, 'error': f"La clase '{old_name}' no existe"}
        
        # Verificar que la clase destino no exista
        new_dir = os.path.join(self.raw_dir, new_name)
        if os.path.exists(new_dir):
            return {'success': False, 'error': f"La clase '{new_name}' ya existe"}
        
        try:
            # Renombrar el directorio
            shutil.move(old_dir, new_dir)
            
            # Actualizar el archivo de clases si existe
            if os.path.exists(self.class_names_path):
                with open(self.class_names_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                
                # Reemplazar la clase en el archivo
                if old_name in classes:
                    classes[classes.index(old_name)] = new_name
                    
                    # Guardar el archivo actualizado
                    with open(self.class_names_path, 'w') as f:
                        for cls in classes:
                            f.write(f"{cls}\n")
            
            return {
                'success': True,
                'old_name': old_name,
                'new_name': new_name,
                'message': f"Clase renombrada de '{old_name}' a '{new_name}'"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}