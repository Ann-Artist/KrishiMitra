import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LeafDiseaseDetector:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.class_names = []
        
    def create_model(self, num_classes):
        """Create CNN model for leaf disease detection"""
        model = models.Sequential([
            # Data augmentation layers
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Global average pooling instead of flatten to reduce parameters
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def load_dataset_from_directory(self, dataset_path):
        """Load dataset from directory structure"""
        print(f"Loading dataset from: {dataset_path}")
        
        # Get class names from subdirectory names
        self.class_names = sorted(os.listdir(dataset_path))
        print(f"Found classes: {self.class_names}")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            print(f"Loading images for class: {class_name}")
            class_image_count = 0
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    
                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.input_shape[:2])
                            img = img.astype(np.float32) / 255.0
                            
                            images.append(img)
                            labels.append(class_idx)
                            class_image_count += 1
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
            
            print(f"  Loaded {class_image_count} images for {class_name}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Total dataset size: {len(images)} images")
        return images, labels
    
    def train_model(self, dataset_path, validation_split=0.2, epochs=50, batch_size=32):
        """Train the leaf disease detection model"""
        
        # Load dataset
        X, y = self.load_dataset_from_directory(dataset_path)
        
        if len(X) == 0:
            raise ValueError("No images found in dataset directory")
        
        # Convert labels to categorical
        num_classes = len(self.class_names)
        y_categorical = tf.keras.utils.to_categorical(y, num_classes)
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, 
            stratify=y, random_state=42
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        
        # Create model
        self.model = self.create_model(num_classes)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Display model summary
        self.model.summary()
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'ml_models/leaf_disease/best_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save class names
        self.save_class_names()
        
        return history
    
    def save_class_names(self):
        """Save class names to labels.txt"""
        labels_path = "ml_models/leaf_disease/labels.txt"
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        
        with open(labels_path, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        print(f"Class names saved to: {labels_path}")
    
    def convert_to_tflite(self, output_path="ml_models/leaf_disease/disease_model.tflite"):
        """Convert trained model to TensorFlow Lite format"""
        if self.model is None:
            print("No model to convert. Train a model first.")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimize the model
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert to TFLite
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted and saved to: {output_path}")
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"Model size: {model_size:.2f} MB")
    
    def predict_disease(self, image_path):
        """Predict disease from image"""
        if self.model is None:
            print("No model loaded. Train a model first.")
            return None
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape[:2])
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        result = {
            'disease': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
        
        return result

def download_sample_dataset():
    """Instructions for downloading dataset"""
    print("📁 Dataset Setup Instructions:")
    print("=" * 50)
    print("1. Download one of these datasets:")
    print("   • PlantVillage Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("   • New Plant Diseases Dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("   • Rice Leaf Diseases: https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases")
    print()
    print("2. Extract to: ML/data/training_data/diseases/")
    print("3. Organize into subfolders by disease class:")
    print("   ML/data/training_data/diseases/")
    print("   ├── healthy/")
    print("   ├── bacterial_blight/")
    print("   ├── brown_spot/")
    print("   └── [other_diseases]/")
    print()
    print("4. Each subfolder should contain images of that specific class")

# Example usage
if __name__ == "__main__":
    detector = LeafDiseaseDetector()
    
    # Show dataset setup instructions
    download_sample_dataset()
    
    # Dataset path based on your project structure
    dataset_path = "ML/data/training_data/diseases"
    
    # Check if dataset exists
    if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
        print(f"\n✅ Dataset found at: {dataset_path}")
        
        # Train model
        print("\n🚀 Starting model training...")
        history = detector.train_model(dataset_path)
        
        # Convert to TFLite
        print("\n📱 Converting to TFLite...")
        detector.convert_to_tflite("ML/ml_models/leaf_disease/disease_model.tflite")
        
        print("\n🎉 Leaf Disease Detection Model Setup Complete!")
        
        # Test prediction (if you have a test image)
        # result = detector.predict_disease("path/to/test/image.jpg")
        # print(f"Prediction: {result}")
        
    else:
        print(f"\n❌ Dataset not found at: {dataset_path}")
        print("Please download and organize your dataset first!")