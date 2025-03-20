# MNIST Digit Recognition Neural Network

## Video Demo
![Demo](https://github.com/kaimorales/MNIST-Predictor/blob/main/demo.gif)
## Project Overview
A neural network designed to recognize handwritten digits using the MNIST dataset, achieving 98.20% accuracy.

## Project Highlights
- **Accuracy**: 98.20%
- **Dataset**: MNIST Handwritten Digits
- **Framework**: PyTorch
- **Model Type**: Multi-layer Neural Network

## Neural Network Architecture
```
Input Layer (784 neurons)
↓
Hidden Layer 1: 512 neurons (ReLU, Dropout 0.3)
↓
Hidden Layer 2: 256 neurons (ReLU, Dropout 0.2)
↓
Hidden Layer 3: 128 neurons (ReLU, Dropout 0.1)
↓
Hidden Layer 4: 60 neurons (ReLU, Dropout 0.05)
↓
Output Layer: 10 neurons (Digit Classification)
```

## Key Technologies
- Python
- PyTorch
- NumPy
- Matplotlib

## Performance Metrics
- **Training Epochs**: 10
- **Batch Size**: 64
- **Optimizer**: Adam
- **Learning Rate**: 0.001

## Key Learnings
1. Neural network layer design
2. Dropout as a regularization technique
3. Importance of layer sizes and activations
4. PyTorch training workflows

## Challenges Overcome
- Preventing overfitting
- Balancing model complexity
- Achieving high accuracy with limited computational resources

## Future Improvements
- Experiment with batch normalization
- Try more advanced architectures
- Implement data augmentation

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

### Installation
```bash
pip install torch torchvision numpy matplotlib
```

### Running the Project
```bash
python mnist_classifier.py
```

## Project Structure
```
mnist-project/
│
├── mnist_classifier.py    # Main training script
├── models/                # Saved model weights
└── README.md              # Project documentation
```

## License
MIT License

## Author
Kai Morales
