# Car Price Classification - Assignment 3
**Student ID: st126010 - Htut Ko Ko**

This project implements a multinomial logistic regression classifier for car price prediction, converting the regression problem into a 4-class classification task. The project includes MLflow experiment tracking, model deployment, and a modern responsive Dash web application with CI/CD pipeline.

## 📋 Project Overview

### Objectives
1. **Classification Implementation**: Convert car price prediction from regression to 4-class classification
2. **Custom Metrics**: Implement accuracy, precision, recall, and F1-score from scratch
3. **Ridge Regularization**: Add L2 penalty option to logistic regression
4. **MLflow Integration**: Log experiments to remote MLflow server
5. **Model Deployment**: Deploy best model using MLflow Model Registry
6. **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
7. **Modern Web App**: Responsive dashboard with model comparisons across all assignments

### Price Classes
- **Class 0**: Low (₹0 - ₹25 Lakhs)
- **Class 1**: Medium (₹25 - ₹50 Lakhs)  
- **Class 2**: High (₹50 Lakhs - ₹1 Crore)
- **Class 3**: Premium (Above ₹1 Crore)

## 🏗️ Project Structure

```
A3/
├── app/
│   └── app.py                    # Modern responsive Dash web application
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions CI/CD pipeline
├── LogisticRegression.py        # Custom logistic regression implementation
├── A3_car_price_classification.ipynb  # Main notebook with experiments
├── Predicting_Car_Price1.ipynb # A1 assignment notebook
├── Predict_Car_Price2.ipynb    # A2 assignment notebook
├── test_model.py               # Unit tests for the model
├── Cars.csv                    # Dataset
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── model_artifacts.pkl         # Trained model and preprocessing components
├── MLFLOW_experiment.png       # MLflow experiment screenshot
├── MLFLOW_scores.png          # MLflow scores screenshot
├── Comparison.png             # Model comparison visualization
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker (for containerization)
- Git (for version control)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd A3
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### 1. Training the Model

Run the Jupyter notebook to train the model and log experiments:

```bash
jupyter notebook A3_car_price_classification.ipynb
```

### 2. Running Unit Tests

Execute the unit tests to verify model functionality:

```bash
python -m pytest test_model.py -v
```

### 3. Running the Web Application

Start the modern Dash web application:

```bash
cd app
python app.py
```

Access the application at `http://localhost:8050`

**Features:**
- 📊 **Model Comparison**: Compare performance across A1, A2, and A3 assignments
- 🔮 **Price Prediction**: Interactive car price class prediction
- 📈 **Data Analytics**: Visualizations and insights from the dataset

### 4. Docker Deployment

Build and run the Docker container:

```bash
# Build the image
docker build -t car-price-classifier .

# Run the container
docker run -p 8050:8050 car-price-classifier
```

## 🔬 Model Implementation

### Custom Logistic Regression Features

The `LogisticRegression` class implements:

- **Multinomial Classification**: One-vs-rest approach for 4-class problem
- **Ridge Regularization**: Optional L2 penalty with configurable lambda
- **Weight Initialization**: Zeros or Xavier initialization methods
- **Custom Metrics**: From-scratch implementation of classification metrics

### Classification Metrics

All metrics implemented from scratch:

- **Accuracy**: `correct_predictions / total_predictions`
- **Precision**: `TP / (TP + FP)` per class
- **Recall**: `TP / (TP + FN)` per class  
- **F1-Score**: `2 * precision * recall / (precision + recall)` per class
- **Macro Averaging**: Simple average across classes
- **Weighted Averaging**: Class-frequency weighted average

## 📈 Assignment Evolution & Results

### Assignment Progression
| Assignment | Model Type | Problem Type | Best Score | Key Features |
|------------|------------|--------------|------------|--------------|
| **A1** | Linear Regression | Regression | R² = 0.6040 | Basic implementation + proper pipeline |
| **A2** | Enhanced Linear Regression | Regression | R² = 0.8472 | Polynomial features + Lasso + proper pipeline |
| **A3** | Logistic Classification | Classification | Accuracy = 70.48% | Custom metrics + Ridge penalty + MLflow + CI/CD + proper pipeline |

### A3 Best Model Configuration
- **Penalty**: None (no regularization needed)
- **Initialization**: Zeros
- **Learning Rate**: 0.01
- **Accuracy**: 70.48%
- **Macro F1**: Improved performance

### Key Findings
1. Zeros initialization outperformed Xavier initialization
2. Higher learning rates (0.01) achieved better convergence
3. Ridge regularization didn't improve performance significantly
4. Proper feature engineering was key to performance improvement

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **Testing Phase**:
   - Sets up Python environment
   - Installs dependencies
   - Runs unit tests
   - Validates Dash app imports

2. **Deployment Phase** (on main branch):
   - Builds Docker image
   - Tests container functionality
   - Provides deployment confirmation

## 🌐 MLflow Integration

### Experiment Tracking
- **Server**: `http://mlflow.ml.brain.cs.ait.ac.th/`
- **Experiment Name**: `st126010-a3`
- **Model Registry**: `st126010-a3-model`

### Logged Metrics
- Accuracy, Precision, Recall, F1-score
- Macro and weighted averages
- Model hyperparameters
- Training artifacts

## 🧪 Testing

### Unit Tests Coverage
- Input validation and format checking
- Output shape verification  
- Model consistency testing
- Error handling for invalid inputs

Run tests with:
```bash
python -m pytest test_model.py -v --tb=short
```

## 📱 Modern Web Application Features

The enhanced Dash app provides:

### 📊 Model Comparison Tab
- Performance comparison across all three assignments
- Interactive charts showing model evolution
- Detailed comparison table with key metrics

### 🔮 Price Prediction Tab
- Interactive car feature input form
- Real-time price class prediction
- Modern, responsive design with visual feedback

### 📈 Data Analytics Tab
- Price distribution visualization
- Feature correlation matrix
- Price trends analysis by year

### Design Features
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Clean, professional interface with gradients and shadows
- **Interactive Elements**: Hover effects, smooth transitions
- **Student Branding**: Displays "st126010 - Htut Ko Ko" prominently

## 🐳 Docker Configuration

The Dockerfile:
- Uses Python 3.10 slim base image
- Installs system dependencies (gcc for compilation)
- Copies requirements and installs Python packages
- Exposes port 8050 for the web application
- Sets environment variables for production

## 📄 License

This project is part of an academic assignment for machine learning coursework.

## 🙋‍♂️ Support

For questions about this implementation:
- Check the Jupyter notebooks for detailed explanations
- Review the MLflow experiments for performance comparisons
- Run unit tests to verify functionality

---

**Student**: st126010 - Htut Ko Ko  
**Course**: Machine Learning  
**Assignment**: A3 - Car Price Classification  

This project demonstrates the complete machine learning pipeline from data preprocessing and model training to deployment and monitoring, following best practices for reproducible ML workflows.
