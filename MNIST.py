import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# 1. Prepare Data
def prepare_data():
    # Simple data transformation
    transform = transforms.ToTensor()
    
    # Download MNIST dataset
    trainset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    testset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=64, 
        shuffle=True
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=64, 
        shuffle=False
    )
    
    return trainloader, testloader

# 2. Simple Neural Network
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

# 3. Training Function
def train_model(model, trainloader, epochs=10):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    
    print('Training Finished')

# 4. Evaluation Function
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy:.2f}%')
    return accuracy  # Return the accuracy value

# 5. Visualization Function
def show_predictions(model, testloader):
    model.eval()
    
    # Grab a batch of test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Plot images
    plt.figure(figsize=(10, 4))
    for i in range(10):  # Show first 10 images
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Pred: {predicted[i]}, True: {labels[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')  # Save the figure first
    plt.show()  # Then show it interactively

# 6. Save Model Function - Separated for clarity
def save_model(model, filename='model.pth', directory='.'):
 #save function made by calude 3.7
    try:
        # Make sure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Full path to save the model
        filepath = os.path.join(directory, filename)
        
        # Save the model
        torch.save(model.state_dict(), filepath)
        
        # Verify if the file was created
        if os.path.exists(filepath):
            print(f"Model successfully saved to {filepath}")
            print(f"File size: {os.path.getsize(filepath) / 1024:.2f} KB")
        else:
            print(f"Error: File {filepath} was not created")
            
    except Exception as e:
        print(f"Error saving model: {str(e)}")

# 7. Main Function
def main():
    try:
        # Prepare data
        print("Preparing data...")
        trainloader, testloader = prepare_data()
        
        # Create model
        print("Creating model...")
        model = NeuralNetwork()
        
        # Train model
        print("Training model...")
        train_model(model, trainloader)
        
        # Evaluate model
        print("Evaluating model...")
        accuracy = evaluate_model(model, testloader)
        
        # Save model - now we save right after evaluation
        print("Saving model...")
        save_model(model)
        
        # After saving, we can visualize predictions
        print("Visualizing predictions...")
        show_predictions(model, testloader)
        
        print(f"Complete! Model achieved {accuracy:.2f}% accuracy and was saved to model.pth")
        
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        # Even if there's an error, try to save the model
        print("Attempting to save model despite error...")
        try:
            save_model(model, 'emergency_save_model.pth')
        except Exception as save_error:
            print(f"Could not perform emergency save: {str(save_error)}")

# Run the script
if __name__ == '__main__':
    main()