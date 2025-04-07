# services/training_service.py
"""
Servicio para el entrenamiento de modelos de clasificación de imágenes
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from PIL import Image

class TrainingService:
    """Servicio para entrenar y evaluar modelos de clasificación"""
    
    def __init__(self, data_dir='data/raw'):
        """
        Inicializa el servicio de entrenamiento
        
        Args:
            data_dir: Directorio con los datos de entrenamiento organizados por carpetas
        """
        self.data_dir = data_dir
        self.img_height = 224
        self.img_width = 224
        self.min_samples_per_class = 5
        
        # Parámetros de entrenamiento
        self.batch_size = 32
        self.epochs = 20
        self.k_folds = 5
        self.learning_rate = 0.001
        
        # Configuraciones adicionales
        self.model_architecture = 'MobileNetV2'  # Opciones: MobileNetV2, ResNet50, EfficientNetB0
        self.use_data_augmentation = True
        self.early_stopping = True
        self.early_stopping_patience = 5
        
        # Crear directorios necesarios
        os.makedirs('models', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        print(f"📋 Servicio de entrenamiento inicializado:")
        print(f"   - Directorio de datos: {self.data_dir}")
        print(f"   - Dimensiones de imágenes: {self.img_height}x{self.img_width}")
        print(f"   - Arquitectura base: {self.model_architecture}")
        print(f"   - Aumento de datos: {'Activado' if self.use_data_augmentation else 'Desactivado'}")
        print(f"   - Early stopping: {'Activado' if self.early_stopping else 'Desactivado'}")
    
    def prepare_dataset(self, test_split=0.2):
        """
        Prepara el conjunto de datos para entrenamiento
        
        Args:
            test_split: Proporción para conjunto de prueba
            
        Returns:
            X: Datos de imágenes
            y: Etiquetas codificadas
            class_names: Nombres de las clases
        """
        X = []  # Datos de imágenes
        y = []  # Etiquetas
        class_names = []  # Nombres de clases
        valid_class_indices = []  # Índices de clases válidas
        
        print("\n=== Preparando conjunto de datos ===")
        
        # Verificar que exista el directorio
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"No se encontró el directorio {self.data_dir}")
        
        # Obtener las clases (carpetas)
        folders = [f for f in sorted(os.listdir(self.data_dir)) 
                 if os.path.isdir(os.path.join(self.data_dir, f))]
        
        if not folders:
            raise ValueError(f"No se encontraron carpetas de clases en {self.data_dir}")
        
        # Crear directorios para división train/test
        train_base_dir = os.path.join("data", "entrenamiento")
        test_base_dir = os.path.join("data", "prueba")
        os.makedirs(train_base_dir, exist_ok=True)
        os.makedirs(test_base_dir, exist_ok=True)
        
        # Procesar cada clase
        for idx, folder in enumerate(folders):
            folder_path = os.path.join(self.data_dir, folder)
            
            print(f"\nProcesando clase: {folder} (índice {idx})")
            
            # Obtener todas las imágenes
            image_files = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            # Verificar si hay suficientes imágenes
            if len(image_files) < self.min_samples_per_class:
                print(f"⚠️ Advertencia: La clase '{folder}' solo tiene {len(image_files)} imágenes, "
                      f"se requieren al menos {self.min_samples_per_class}. Esta clase será ignorada.")
                continue
            
            print(f"Encontradas {len(image_files)} imágenes")
            
            # Procesar cada imagen
            class_X = []  # Imágenes de esta clase
            class_y = []  # Etiquetas de esta clase
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((self.img_width, self.img_height))
                    img_array = np.array(img) / 255.0  # Normalización
                    
                    class_X.append(img_array)
                    class_y.append(idx)
                    
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
            
            # Solo añadir la clase si se procesaron suficientes imágenes
            if len(class_X) >= self.min_samples_per_class:
                X.extend(class_X)
                y.extend(class_y)
                class_names.append(folder)
                valid_class_indices.append(idx)
                print(f"✓ Clase '{folder}' añadida con {len(class_X)} imágenes")
            else:
                print(f"⚠️ Advertencia: No se pudieron procesar suficientes imágenes para '{folder}', "
                      f"se requieren al menos {self.min_samples_per_class}. Esta clase será ignorada.")
        
        # Verificar si hay suficientes clases
        if len(class_names) == 0:
            raise ValueError("No se encontraron clases con suficientes imágenes para entrenar")
        
        # Ajustar índices si algunas clases fueron ignoradas
        if len(valid_class_indices) < len(folders):
            # Crear un mapa de índice original a nuevo índice
            idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_class_indices)}
            # Remapear las etiquetas
            y = [idx_map[label] for label in y]
        
        # Convertir a arrays de numpy
        X = np.array(X)
        y = np.array(y)
        
        # Codificar etiquetas en one-hot
        y_encoded = to_categorical(y, num_classes=len(class_names))
        
        print(f"\n✅ Dataset preparado: {X.shape[0]} imágenes en {len(class_names)} clases")
        print(f"   Dimensiones de las imágenes: {self.img_height}x{self.img_width}")
        
        # Visualizar algunos ejemplos
        self.visualize_examples(X, y, class_names)
        
        return X, y_encoded, class_names
    
    def visualize_examples(self, X, y, class_names, samples_per_class=3):
        """
        Visualiza ejemplos de cada clase
        
        Args:
            X: Datos de imágenes
            y: Etiquetas (no codificadas)
            class_names: Nombres de las clases
            samples_per_class: Número de ejemplos por clase
        """
        os.makedirs('output', exist_ok=True)
        
        num_classes = len(class_names)
        plt.figure(figsize=(15, 2 * num_classes))
        
        for class_idx in range(num_classes):
            # Obtenemos imágenes de esta clase
            indices = np.where(y == class_idx)[0]
            
            # Si hay suficientes imágenes, mostramos samples_per_class ejemplos
            if len(indices) >= samples_per_class:
                sample_indices = indices[:samples_per_class]
                
                for i, sample_idx in enumerate(sample_indices):
                    plt.subplot(num_classes, samples_per_class, class_idx * samples_per_class + i + 1)
                    plt.imshow(X[sample_idx])
                    plt.title(class_names[class_idx])
                    plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('output/class_examples.png')
        plt.close()
        print("✓ Ejemplos de clases guardados en 'output/class_examples.png'")
    
    def create_model(self, num_classes):
        """
        Crea un modelo de red neuronal convolucional para clasificación de imágenes
        
        Args:
            num_classes: Número de clases a clasificar
        
        Returns:
            modelo: Modelo de red neuronal compilado
        """
        # Seleccionar la arquitectura base
        if self.model_architecture == 'MobileNetV2':
            print("🏗️ Creando modelo con MobileNetV2 como base...")
            base_model = applications.MobileNetV2(
                input_shape=(self.img_height, self.img_width, 3),
                include_top=False,
                weights='imagenet'
            )
        elif self.model_architecture == 'ResNet50':
            print("🏗️ Creando modelo con ResNet50 como base...")
            base_model = applications.ResNet50(
                input_shape=(self.img_height, self.img_width, 3),
                include_top=False,
                weights='imagenet'
            )
        elif self.model_architecture == 'EfficientNetB0':
            print("🏗️ Creando modelo con EfficientNetB0 como base...")
            base_model = applications.EfficientNetB0(
                input_shape=(self.img_height, self.img_width, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            print(f"⚠️ Arquitectura {self.model_architecture} no reconocida. Usando MobileNetV2 por defecto...")
            base_model = applications.MobileNetV2(
                input_shape=(self.img_height, self.img_width, 3),
                include_top=False,
                weights='imagenet'
            )
        
        # Congelamos las capas base para fine-tuning
        base_model.trainable = False
        
        # Construir modelo completo
        modelo = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compilamos el modelo
        modelo.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Imprimir resumen
        modelo.summary()
        
        return modelo
    
    def train_with_cross_validation(self, X, y, num_classes, k_folds=None):
        """
        Entrena el modelo utilizando validación cruzada
        
        Args:
            X: Datos de imágenes
            y: Etiquetas
            num_classes: Número de clases
            k_folds: Número de particiones para validación cruzada
        
        Returns:
            histories: Historiales de entrenamiento
            val_accuracies: Precisiones de validación
            best_model: Mejor modelo entrenado
        """
        if k_folds is None:
            k_folds = self.k_folds
            
        # Ajustamos k_folds si hay pocas muestras
        sample_count = X.shape[0]
        if sample_count < k_folds * 2:
            new_k_folds = max(2, sample_count // 2)
            print(f"⚠️ Advertencia: Demasiados folds ({k_folds}) para {sample_count} muestras.")
            print(f"Ajustando a {new_k_folds} folds para evitar errores.")
            k_folds = new_k_folds
        
        # Definimos la validación cruzada
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_no = 1
        histories = []
        val_accuracies = []
        best_accuracy = 0
        best_model = None

        # Configurar callbacks
        callbacks = []
        
        # Early stopping para evitar sobreajuste
        if self.early_stopping:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Reducer para ajustar el learning rate automáticamente
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        print(f"\n🔄 Iniciando entrenamiento con validación cruzada ({k_folds} folds)...")
        
        for train_idx, val_idx in kfold.split(X):
            print(f'\n=== Entrenando fold {fold_no}/{k_folds} ===')
            
            # Dividimos los datos
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"   Muestras de entrenamiento: {X_train.shape[0]}")
            print(f"   Muestras de validación: {X_val.shape[0]}")
            
            # Creamos y entrenamos el modelo
            model = self.create_model(num_classes)
            
            # Configurar aumento de datos
            train_data = None
            
            if self.use_data_augmentation:
                print("   Aplicando aumento de datos en el entrenamiento...")
                data_augmentation = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    zoom_range=0.2,
                    shear_range=0.1,
                    fill_mode='nearest'
                )
                train_data = data_augmentation.flow(
                    X_train, y_train, 
                    batch_size=self.batch_size
                )
            else:
                # Sin aumento de datos
                train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)
            
            # Entrenamos el modelo
            print(f"   Iniciando entrenamiento (máximo {self.epochs} épocas)...")
            history = model.fit(
                train_data,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluamos el modelo
            print("   Evaluando modelo en datos de validación...")
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f'   Fold {fold_no} completado: Precisión de validación = {val_accuracy:.4f}')
            
            histories.append(history)
            val_accuracies.append(val_accuracy)
            
            # Guardamos el mejor modelo
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                print(f"   ⭐ Nuevo mejor modelo encontrado (precisión: {val_accuracy:.4f})")
            
            fold_no += 1
        
        print(f"\n✅ Validación cruzada completada:")
        print(f"   Mejor precisión: {best_accuracy:.4f}")
        print(f"   Precisión promedio: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
        
        return histories, val_accuracies, best_model
    
    def plot_training_history(self, histories, k_folds=None):
        """
        Grafica el historial de entrenamiento
        
        Args:
            histories: Lista de historiales de entrenamiento
            k_folds: Número de particiones
        """
        if k_folds is None:
            k_folds = self.k_folds
            
        plt.figure(figsize=(12, 8))
        
        for i in range(min(k_folds, len(histories))):
            plt.subplot(2, 3, i+1)
            plt.plot(histories[i].history['accuracy'], label='train')
            plt.plot(histories[i].history['val_accuracy'], label='validation')
            plt.title(f'Fold {i+1}')
            plt.xlabel('Épocas')
            plt.ylabel('Precisión')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('output/cross_validation_results.png')
        plt.close()
        print("✓ Gráficas de entrenamiento guardadas en 'output/cross_validation_results.png'")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Genera y guarda la matriz de confusión
        
        Args:
            y_true: Etiquetas reales
            y_pred: Predicciones
            class_names: Nombres de las clases
            
        Returns:
            cm: Matriz de confusión calculada
        """
        # Calculamos la matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Visualizamos y guardamos la matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title('Matriz de Confusión')
        plt.tight_layout()
        plt.savefig('output/confusion_matrix.png')
        plt.close()
        print("✓ Matriz de confusión guardada en 'output/confusion_matrix.png'")
        
        # También guardamos un reporte de clasificación
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv('output/classification_report.csv')
        print("✓ Reporte de clasificación guardado en 'output/classification_report.csv'")
        
        return cm
    
    def train_model(self, epochs=None, batch_size=None, k_folds=None, 
                    learning_rate=None, save_model_path=None):
        """
        Entrena un nuevo modelo con los datos disponibles
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño de lote
            k_folds: Número de folds para validación cruzada
            learning_rate: Tasa de aprendizaje
            save_model_path: Ruta para guardar el modelo (por defecto, 'models/best_model.h5')
            
        Returns:
            Resultados del entrenamiento
        """
        # Actualizar parámetros si se proporcionan
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if k_folds is not None:
            self.k_folds = k_folds
        if learning_rate is not None:
            self.learning_rate = learning_rate
            
        # Establecer ruta de guardado por defecto si no se especifica
        if save_model_path is None:
            save_model_path = os.path.join('models', 'best_model.h5')
        
        # Preparar el dataset
        print("🔄 Preparando conjunto de datos...")
        X, y, class_names = self.prepare_dataset()
        num_classes = len(class_names)
        
        # Entrenar con validación cruzada
        print("\n🚀 Iniciando entrenamiento con validación cruzada...")
        histories, val_accuracies, best_model = self.train_with_cross_validation(
            X, y, num_classes, k_folds=self.k_folds
        )
        
        # Graficar resultados de validación cruzada
        print("\n📊 Generando gráficas y evaluaciones...")
        self.plot_training_history(histories, k_folds=self.k_folds)
        
        # Guardar el mejor modelo
        best_model.save(save_model_path)
        print(f"✓ Mejor modelo guardado en '{save_model_path}'")
        
        # Evaluar en todo el conjunto de datos
        y_pred_probs = best_model.predict(X)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y, axis=1)
        
        # Generar y guardar la matriz de confusión
        self.plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Guardar los nombres de las clases
        class_names_path = os.path.join('models', 'class_names.txt')
        with open(class_names_path, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        print(f"✓ Nombres de clases guardados en '{class_names_path}'")
        
        # Generar archivo de resumen
        accuracy_mean = float(np.mean(val_accuracies))
        accuracy_std = float(np.std(val_accuracies))
        
        with open(os.path.join('output', 'model_summary.txt'), 'w') as f:
            f.write(f"Número de clases: {num_classes}\n")
            f.write(f"Clases: {', '.join(class_names)}\n")
            f.write(f"Tamaño de imagen: {self.img_height}x{self.img_width}\n")
            f.write(f"Épocas: {self.epochs}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Learning rate: {self.learning_rate}\n")
            f.write(f"Número de folds: {self.k_folds}\n")
            f.write(f"Precisión promedio: {accuracy_mean:.4f}\n")
            f.write(f"Desviación estándar: {accuracy_std:.4f}\n")
        print("✓ Resumen del modelo guardado en 'output/model_summary.txt'")
        
        # Preparar resultados para devolver
        results = {
            'accuracy': accuracy_mean,
            'std_accuracy': accuracy_std,
            'num_classes': num_classes,
            'classes': class_names,
            'model_path': save_model_path,
            'class_names_path': class_names_path
        }
        
        print(f"\n✅ Entrenamiento completado con éxito.")
        print(f"   Precisión promedio: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
        
        return results