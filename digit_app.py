import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import os


# Neural Network definition - identical to your training code
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Flatten the 28x28 image to a 784-dimensional vector
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
           #input
            nn.Flatten(),
            #layer1
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            #layer 2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            #layer 3
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            #layer 4
            nn.Linear(128, 60),
            nn.ReLU(),
            nn.Dropout(0.05),
            #layer 5
            nn.Linear(60,10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_stack(x)

class DrawingApp:
    def __init__(self, root):
        # Setup the window
        self.root = root
        self.root.title("MNIST Digit Recognizer")
        self.root.geometry("400x450")
        
        # Create a label with instructions
        self.instruction = tk.Label(root, text="Draw a digit (0-9)", font=("Arial", 14))
        self.instruction.pack(pady=10)
        
        # Create canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg="black", bd=3, relief=tk.SUNKEN)
        self.canvas.pack(pady=10)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # Create buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)
        
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas, width=10, height=2)
        self.clear_button.grid(row=0, column=0, padx=10)
        
        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict, width=10, height=2)
        self.predict_button.grid(row=0, column=1, padx=10)
        
        # Create result display
        self.result = tk.Label(root, text="", font=("Arial", 18, "bold"))
        self.result.pack(pady=10)
        
        # Path to the model file - modify this to match your model's location
        model_path = '/Users/kaimorales/model.pth'
        
        # Check if model file exists and print helpful message
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            print(f"Current working directory: {os.getcwd()}")
            print("Please make sure your model.pth file is in this directory")
            print("Or update the model_path variable to point to your model file")
            self.result.config(text="⚠️ Model file not found")
        else:
            try:
                # Create model instance
                self.model = NeuralNetwork()
                # Load the weights
                self.model.load_state_dict(torch.load(model_path))
                # Set to evaluation mode
                self.model.eval()
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Make sure the model file was saved correctly")
                self.result.config(text="⚠️ Error loading model")
        
        # Drawing variables
        self.prev_x = None
        self.prev_y = None
    
    def start_draw(self, event):
        self.prev_x = event.x
        self.prev_y = event.y
    
    def draw(self, event):
        x, y = event.x, event.y
        if self.prev_x and self.prev_y:
            # Draw a white line from previous position to current position
            self.canvas.create_line(
                self.prev_x, self.prev_y, x, y,
                width=15, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE
            )
        self.prev_x = x
        self.prev_y = y
    
    def clear_canvas(self):
        # Delete everything on the canvas
        self.canvas.delete("all")
        # Reset result display
        self.result.config(text="")
        # Reset drawing variables
        self.prev_x = None
        self.prev_y = None
    
    def predict(self):
        # Make sure the model was loaded successfully
        if not hasattr(self, 'model'):
            self.result.config(text="No model loaded")
            return
        
        # Create a PIL image from the canvas
        image = Image.new("L", (280, 280), "black")
        draw = ImageDraw.Draw(image)
        
        # Draw all lines from the canvas onto the PIL image
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 4:  # Line has 4 coordinates: x1, y1, x2, y2
                draw.line(coords, fill="white", width=15)
        
        # Resize to MNIST format (28x28 pixels)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to 0-1 range
        img_array = np.array(image) / 255.0
        
        # Convert to PyTorch tensor with correct dimensions [batch, channel, height, width]
        tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        try:
            # Make prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted].item() * 100
            
            # Update result label
            self.result.config(text=f"Prediction: {predicted} ({confidence:.1f}%)")
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.result.config(text="Error making prediction")

# Main function
def main():
    # Create the main window
    root = tk.Tk()
    # Create app instance
    app = DrawingApp(root)
    # Start the app
    root.mainloop()

if __name__ == "__main__":
    main()