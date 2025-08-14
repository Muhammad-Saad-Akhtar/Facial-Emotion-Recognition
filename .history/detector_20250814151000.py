import torch
import torch.nn as nn
import cv2
import numpy as np
from tkinter import Tk, filedialog
import os
from train import EmotionCNN, IMG_SIZE, NUM_CHANNELS, NUM_CLASSES

# Emotion labels
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def load_model():
    """Load the trained model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create and load model
    model = EmotionCNN().to(device)
    model_path = r"C:\Users\HP\Desktop\Others\Facial-Emotion-Recognition\weights\final_model.pth"
    
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Preprocess the image for model input."""
    try:
        # Read and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read the image")
            
        # Resize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_emotion(model, image_tensor, device):
    """Predict emotion from image tensor."""
    try:
        with torch.no_grad():
            # Move image to device and get prediction
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            
            # Get prediction probabilities for all emotions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get predictions for all emotions
            predictions = []
            for idx, prob in enumerate(probabilities):
                emotion = EMOTIONS[idx]
                probability = prob.item() * 100
                predictions.append((emotion, probability))
            
            # Sort by probability in descending order
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def select_image():
    """Open a file dialog to select an image."""
    root = Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    
    return file_path if file_path else None

def display_result(image_path, predictions):
    """Display the image with predictions."""
    if predictions is None:
        return
    
    # Read and resize image for display
    img = cv2.imread(image_path)
    display_img = cv2.resize(img, (400, 400))
    
    # Create a black background for text
    text_bg = np.zeros((280, 400, 3), dtype=np.uint8)  # Increased height for all emotions
    
    # Add predictions text
    for i, (emotion, probability) in enumerate(predictions):
        text = f"{emotion}: {probability:.2f}%"
        position = (10, 30 + i * 35)  # Reduced spacing between lines
        cv2.putText(text_bg, text, position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine image and text
    final_img = np.vstack([display_img, text_bg])
    
    # Display
    cv2.imshow("Emotion Detection", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load model
    model, device = load_model()
    if model is None:
        return
    
    while True:
        # Select image
        print("\nSelect an image file (press Cancel to exit)...")
        image_path = select_image()
        
        if image_path is None or not image_path:
            print("No image selected. Exiting...")
            break
        
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            continue
        
        # Predict emotion
        predictions = predict_emotion(model, image_tensor, device)
        if predictions is None:
            continue
        
        # Print results
        print("\nPredictions:")
        for emotion, probability in predictions:
            print(f"{emotion}: {probability:.2f}%")
        
        # Display result and wait for window to close
        display_result(image_path, predictions)
        
        # After window is closed, loop continues automatically to select new image
    
    print("Thank you for using the Emotion Detector!")

if __name__ == "__main__":
    main()
