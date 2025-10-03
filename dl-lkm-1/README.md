# Deep Learning Fundamentals - LKM 2

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)

> **LKM 2: Dasar Deep Learning** - Implementasi komprehensif konsep fundamental deep learning dengan visualisasi interaktif dan analisis mendalam.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Results](#results)
- [Visualizations](#visualizations)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Project ini merupakan implementasi lengkap dari **Lembar Kerja Mahasiswa (LKM) 2** tentang dasar-dasar Deep Learning. Mencakup 4 eksperimen utama mulai dari neuron tunggal hingga klasifikasi MNIST dengan analisis mendalam tentang fungsi aktivasi, training dynamics, dan linear separability.

### Learning Objectives
- ✅ Memahami komponen fundamental neuron buatan
- ✅ Menganalisis peran bias dalam decision making
- ✅ Membandingkan karakteristik fungsi aktivasi
- ✅ Mengimplementasikan neural network untuk logic gates
- ✅ Membangun classifier untuk real-world data (MNIST)

## 🚀 Key Features

### 🧠 Core Implementations
- **Single Neuron**: Implementasi from-scratch dengan NumPy
- **Activation Functions**: 6 fungsi aktivasi dengan analisis komparatif
- **OR Gate Network**: PyTorch neural network untuk logic gates
- **MNIST Binary Classifier**: Single neuron untuk digit classification

### 📊 Advanced Analytics
- **Decision Boundary Visualization**: 2D dan 3D plotting
- **Training Dynamics**: Learning curve analysis
- **Performance Metrics**: Comprehensive evaluation
- **Gradient Analysis**: Vanishing gradient investigation

### 🎨 Rich Visualizations
- **Interactive Plots**: 25+ high-quality visualizations
- **Architecture Diagrams**: Neural network structure visualization
- **Mathematical Insights**: Function derivatives dan gradients
- **Results Summary**: Comprehensive analysis dashboard

## 📁 Project Structure

```
dl-lkm-1/
├── 📓 notebooks/                          # Jupyter notebooks
│   ├── 01_neuron_tunggal.ipynb           # Single neuron implementation
│   ├── 02_perbandingan_aktivasi.ipynb    # Activation function comparison
│   ├── 03_neural_network_sederhana.ipynb # OR gate neural network
│   └── 04_mlp_mnist.ipynb                # MNIST binary classification
├── 🔧 src/                               # Source code modules
│   └── visualizations.py                 # Comprehensive visualization generator
├── 📚 docs/                              # Documentation
│   ├── jawaban_pertanyaan_lkm.md         # LKM Q&A answers
│   └── hasil_eksperimen.md               # Detailed experiment results
├── 📊 results/                           # Experiment outputs
│   ├── or_gate_results.json              # OR gate training results
│   └── mlp_mnist_results.json            # MNIST classification results  
├── 🎨 assets/                            # Generated visualizations
│   ├── activation_functions_comprehensive.png
│   ├── neuron_architecture_visualization.png
│   ├── learning_curves_optimization.png
│   ├── mnist_analysis_results.png
│   └── comprehensive_summary.png
├── 📦 data/                              # Datasets
│   └── mnist/                            # MNIST dataset cache
├── ⚙️ requirements.txt                   # Python dependencies
├── 🚫 .gitignore                         # Git ignore rules
└── 📖 README.md                          # This file
```

## 🛠 Installation

### Prerequisites
- Python 3.8+ 
- pip package manager
- Jupyter Notebook

### Setup Steps

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd dl-lkm-1
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

### Dependencies
```txt
torch>=1.9.0              # Deep learning framework
torchvision>=0.10.0        # Computer vision datasets
numpy>=1.21.0              # Numerical computing
matplotlib>=3.4.0          # Plotting library
seaborn>=0.11.0            # Statistical visualization
pandas>=1.3.0              # Data manipulation
scikit-learn>=0.24.0       # Machine learning utilities
jupyter>=1.0.0             # Interactive notebooks
tqdm>=4.61.0               # Progress bars
```

## ⚡ Quick Start

### 1. Generate All Visualizations
```bash
python3 src/visualizations.py
```
Output: 5 comprehensive plots in `assets/` folder

### 2. Run Individual Experiments

#### Experiment 1: Single Neuron
```bash
jupyter notebook notebooks/01_neuron_tunggal.ipynb
```
- Implements basic neuron with bias analysis
- Visualizes decision boundaries
- Compares with/without bias scenarios

#### Experiment 2: Activation Functions
```bash
jupyter notebook notebooks/02_perbandingan_aktivasi.ipynb
```
- Compares 6 activation functions
- Analyzes gradients and saturation
- Performance recommendations

#### Experiment 3: OR Gate Network
```bash
jupyter notebook notebooks/03_neural_network_sederhana.ipynb
```
- PyTorch neural network implementation
- Training loop with BCELoss
- Learning rate optimization

#### Experiment 4: MNIST Classification
```bash
jupyter notebook notebooks/04_mlp_mnist.ipynb
```
- Binary digit classification (0 vs others)
- Performance comparison across activations
- Real-world data analysis

### 3. View Complete Results
```bash
# Open comprehensive documentation
open docs/hasil_eksperimen.md
```

## 🧪 Experiments

### Experiment 1: Single Neuron Analysis
**Objective**: Understand fundamental neuron components and bias effects

**Key Results**:
- ✅ Bias shifts decision boundary (non-origin)
- ✅ Sigmoid provides smooth probabilistic output
- ✅ Single neuron works for linearly separable problems

**Files**: `notebooks/01_neuron_tunggal.ipynb`

### Experiment 2: Activation Function Comparison
**Objective**: Compare characteristics of different activation functions

**Functions Tested**:
- Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish

**Key Findings**:
- ✅ **ReLU**: Best for hidden layers (no vanishing gradient)
- ✅ **Sigmoid**: Best for binary classification output
- ✅ **Tanh**: Good when zero-centered needed
- ❌ **Sigmoid/Tanh**: Vanishing gradient in deep networks

**Files**: `notebooks/02_perbandingan_aktivasi.ipynb`

### Experiment 3: OR Gate Neural Network
**Objective**: Implement neural network for logic gate learning

**Architecture**: 2 inputs → 1 neuron → 1 output

**Results**:
- ✅ **100% accuracy** achieved
- ✅ **~300 epochs** convergence
- ✅ **Learning rate 0.1** optimal
- ✅ **Linearly separable** problem solved perfectly

**Files**: `notebooks/03_neural_network_sederhana.ipynb`

### Experiment 4: MNIST Binary Classification
**Objective**: Real-world data classification with single neuron

**Setup**: 784 inputs → 1 neuron → binary output (digit 0 vs others)

**Results by Activation**:
| Activation | Test Accuracy | Convergence | Best Use |
|------------|---------------|-------------|----------|
| **Sigmoid** | **97.3%** | 4 epochs | Binary output |
| Tanh | 95.8% | 5 epochs | Zero-centered |
| ReLU | 94.2% | 5 epochs | Hidden layers |

**Files**: `notebooks/04_mlp_mnist.ipynb`

## 📊 Results

### Performance Summary

| Experiment | Problem Type | Accuracy | Key Insight |
|------------|--------------|----------|-------------|
| Single Neuron | Toy (bias analysis) | Analytical | Bias crucial for flexibility |
| Activation Comparison | Theoretical | Comparative | ReLU best for deep networks |
| OR Gate | Logic gate | 100% | Linear separability = perfect |
| MNIST Binary | Real data | 97.3% | Single neuron surprisingly good |

### Linear Separability Analysis
- ✅ **OR/AND/NAND/NOR**: Single neuron perfect
- ❌ **XOR/XNOR**: Need multi-layer network
- ✅ **MNIST Binary**: High performance possible

### Training Dynamics
- **Learning Rate**: 0.1 optimal for most experiments
- **Convergence**: 300-500 epochs typical
- **Overfitting**: Minimal with single neuron
- **Gradient Flow**: Good for shallow networks

## 🎨 Visualizations

### Generated Plots (5 comprehensive visualizations)

1. **`activation_functions_comprehensive.png`**
   - 6 activation functions comparison
   - Function plots, derivatives, characteristics
   - Decision boundary analysis

2. **`neuron_architecture_visualization.png`**
   - Single neuron structure diagram
   - Forward pass visualization
   - Mathematical formulation

3. **`learning_curves_optimization.png`**
   - Training dynamics across experiments
   - Loss curves and convergence analysis
   - Learning rate comparison

4. **`mnist_analysis_results.png`**
   - MNIST classification results
   - Confusion matrices and metrics
   - Activation function comparison

5. **`comprehensive_summary.png`**
   - Project overview dashboard
   - Key results and insights
   - Performance summary

### Interactive Features
- **Jupyter Widgets**: Parameter adjustment sliders
- **Plotly Integration**: 3D interactive plots
- **Real-time Updates**: Dynamic visualization

## 📚 Documentation

### Core Documents
- **[LKM Q&A Answers](docs/jawaban_pertanyaan_lkm.md)**: Detailed answers to all LKM questions
- **[Experiment Results](docs/hasil_eksperimen.md)**: Comprehensive analysis and findings
- **[README](README.md)**: Project overview and usage guide

### Code Documentation
- **Docstrings**: All functions documented
- **Type Hints**: Function signatures with types
- **Comments**: Inline explanations for complex logic
- **Examples**: Usage examples in notebooks

### Mathematical Background
- **Linear Algebra**: Vector operations and matrix multiplication
- **Calculus**: Derivative computation and chain rule
- **Probability**: Sigmoid interpretation and cross-entropy
- **Optimization**: Gradient descent and learning dynamics

## 🔬 Advanced Topics

### Research Extensions
1. **Multi-layer Networks**: Solve XOR problem
2. **Regularization**: Add dropout and weight decay
3. **Advanced Optimizers**: Adam, RMSprop comparison
4. **Convolutional Networks**: Full MNIST classification

### Implementation Details
- **Numerical Stability**: Safe exponential computation
- **Memory Efficiency**: Batch processing optimization
- **Reproducibility**: Fixed random seeds
- **Error Handling**: Robust input validation

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards
- **PEP 8**: Python style guide compliance
- **Type Hints**: Add type annotations
- **Documentation**: Update docstrings
- **Testing**: Add unit tests for new features

### Issues and Feedback
- 🐛 **Bug Reports**: Use GitHub Issues
- 💡 **Feature Requests**: Discussion encouraged
- 📖 **Documentation**: Improvements welcome
- 🔧 **Code Review**: Constructive feedback appreciated

## 📈 Performance Metrics

### Computational Requirements
- **Runtime**: ~2 hours total (all experiments)
- **Memory**: 4GB RAM recommended
- **Storage**: 2GB for datasets and results
- **GPU**: Optional (CPU sufficient)

### Scalability Analysis
- **Single Neuron**: O(n) time complexity
- **Training**: O(epochs × samples) 
- **Visualization**: O(resolution²) for 2D plots
- **Memory**: O(features × samples) for data

## 🎓 Educational Value

### Learning Outcomes
After completing this project, you will understand:

1. **Neural Network Fundamentals**
   - Neuron structure and computation
   - Weight, bias, and activation functions
   - Forward pass mathematics

2. **Training Dynamics**
   - Gradient descent optimization
   - Loss function design
   - Learning rate effects

3. **Activation Functions**
   - Mathematical properties
   - Gradient behavior
   - Use case recommendations

4. **Problem Analysis**
   - Linear separability assessment
   - Data preprocessing importance
   - Performance evaluation metrics

### Prerequisites Knowledge
- **Python Programming**: Functions, classes, NumPy
- **Mathematics**: Linear algebra, calculus basics
- **Machine Learning**: Supervised learning concepts
- **Statistics**: Probability and distributions

### Next Steps
- **Deep Networks**: Multi-layer implementations
- **CNN**: Convolutional neural networks
- **RNN**: Recurrent neural networks
- **Advanced Topics**: Attention, transformers

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Deep Learning LKM

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- **PyTorch Team**: Excellent deep learning framework
- **Jupyter Project**: Interactive notebook environment  
- **Matplotlib/Seaborn**: Beautiful visualization libraries
- **NumPy**: Fundamental array computing
- **MNIST Database**: Classic machine learning dataset
- **Open Source Community**: Continuous inspiration

## 📞 Contact

- **Project Lead**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername]
- **LinkedIn**: [your-linkedin-profile]

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Built with ❤️ for deep learning education

[🔝 Back to Top](#deep-learning-fundamentals---lkm-2)

</div>