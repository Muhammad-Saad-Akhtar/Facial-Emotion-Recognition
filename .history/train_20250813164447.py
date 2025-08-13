import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
IMG_SIZE = 48
NUM_CHANNELS = 1  # Grayscale images
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Emotion labels
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

class EmotionDataset(Dataset):
    """Custom Dataset for loading emotion recognition data."""
    def __init__(self, data_path, transform=None, train=True):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load data
        folder_type = 'train' if train else 'test'
        base_path = os.path.join(data_path, folder_type)
        
        # Get total number of images for progress bar
        total_images = sum(len(os.listdir(os.path.join(base_path, emotion))) 
                          for emotion in EMOTIONS)
        
        print(f"\nLoading {folder_type} data...")
        pbar = tqdm(total=total_images, desc=f'Processing {folder_type} images')
        
        for emotion_idx, emotion in enumerate(EMOTIONS):
            emotion_path = os.path.join(base_path, emotion)
            for img_file in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_file)
                try:
                    # Read image in grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    
                    self.images.append(img)
                    self.labels.append(emotion_idx)
                    pbar.update(1)
                except Exception as e:
                    print(f"\nError loading image {img_path}: {str(e)}")
        
        pbar.close()
        
        # Convert to numpy arrays
        self.images = np.array(self.images, dtype=np.float32) / 255.0
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor if not already
        image = torch.FloatTensor(image)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def build_model():
    """Create the CNN model architecture."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Flatten and Dense Layers
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot the training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Data paths
    data_dir = r"C:\Users\HP\Desktop\Others\Data\Facial_Emotions"
    
    # Create weights directory
    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)
    
    # Load training data
    train_images, train_labels = load_data(data_dir, 'train')
    
    # Convert labels to categorical
    train_labels = to_categorical(train_labels, NUM_CLASSES)
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_labels, 
        test_size=0.2, 
        random_state=42
    )
    
    print("\nDataset Summary:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build and compile model
    print("\nBuilding and compiling model...")
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        # Model checkpoint to save best weights
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(weights_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when learning stagnates
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    final_model_path = os.path.join(weights_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved as '{final_model_path}'")
    
    # Load and evaluate on test data
    print("\nEvaluating on test data...")
    test_images, test_labels = load_data(data_dir, 'test')
    test_labels = to_categorical(test_labels, NUM_CLASSES)
    
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()

if __name__ == "__main__":
    main()
