# Iris Species Classification

A machine learning project that classifies iris flowers into three species using logistic regression. This project demonstrates the complete data science workflow from data exploration to model evaluation and prediction.

## 📊 Dataset

The project uses the famous **Iris dataset** which contains measurements of iris flowers from three different species:
- **Iris-setosa**
- **Iris-versicolor** 
- **Iris-virginica**

### Features
- `sepal_length`: Length of the sepal in cm
- `sepal_width`: Width of the sepal in cm
- `petal_length`: Length of the petal in cm
- `petal_width`: Width of the petal in cm

### Dataset Statistics
- **Total samples**: 150
- **Features**: 4 numerical features
- **Target classes**: 3 species (50 samples each)
- **No missing values**: Clean dataset ready for analysis

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.x**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
  - `LogisticRegression`: Classification model
  - `train_test_split`: Data splitting
  - `accuracy_score`: Model evaluation
  - `classification_report`: Detailed performance metrics
  - `confusion_matrix`: Classification performance visualization

### Visualization Libraries
- **Matplotlib**: Basic plotting and visualization
- **Seaborn**: Statistical data visualization
  - Scatter plots for data exploration
  - Heatmaps for confusion matrix visualization

### Development Environment
- **Jupyter Notebook**: Interactive development and analysis

## 🚀 Project Structure

```
iris-classification/
├── iris-classification.ipynb    # Main Jupyter notebook
├── iris-classification.csv      # Dataset
└── README.md                    # Project documentation
```

## 📈 Model Performance

The logistic regression model achieved excellent performance:

- **Accuracy**: 100.00% on test set
- **Precision**: 1.00 for all classes
- **Recall**: 1.00 for all classes
- **F1-Score**: 1.00 for all classes

### Confusion Matrix
```
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

Perfect classification with no misclassifications on the test set!

## 🔬 Methodology

### 1. Data Exploration
- Loaded and examined the dataset structure
- Generated summary statistics using `describe()`
- Checked for missing values (none found)
- Created scatter plots to visualize feature relationships

### 2. Data Preprocessing
- Separated features (X) and target variable (y)
- Split data into training (80%) and testing (20%) sets
- Used `random_state=42` for reproducible results

### 3. Model Training
- Implemented Logistic Regression classifier
- Set `max_iter=200` for convergence
- Trained on 120 samples, tested on 30 samples

### 4. Model Evaluation
- Calculated accuracy score
- Generated classification report with precision, recall, and F1-score
- Created confusion matrix visualization
- Performed manual prediction testing

### 5. Prediction Capability
- Implemented manual input prediction system
- Tested with custom feature values
- Provided confidence scores for predictions

## 💻 Usage

### Running the Notebook
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook iris-classification.ipynb
   ```

3. Run all cells sequentially to reproduce the analysis

### Making Predictions
You can modify the manual prediction section in the notebook to test with your own iris measurements:

```python
# Example prediction
sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

# The model will predict the species
```

## 📊 Key Insights

1. **Perfect Classification**: The model achieves 100% accuracy, indicating the iris species are highly separable based on the four measurements.

2. **Feature Importance**: All four features (sepal and petal dimensions) contribute to accurate classification.

3. **Data Quality**: The dataset is clean with no missing values, making it ideal for machine learning.

4. **Model Reliability**: The logistic regression model is simple yet highly effective for this classification task.

## 🎯 Applications

This project demonstrates:
- **Data Science Workflow**: Complete end-to-end analysis
- **Classification Techniques**: Logistic regression implementation
- **Model Evaluation**: Comprehensive performance assessment
- **Visualization**: Data exploration and results presentation
- **Prediction System**: Interactive prediction capability

## 📚 Learning Outcomes

- Data loading and exploration with Pandas
- Data visualization with Matplotlib and Seaborn
- Machine learning model implementation with Scikit-learn
- Model evaluation and performance metrics
- Confusion matrix interpretation
- Interactive prediction system development

## 🔮 Future Enhancements

- Implement other classification algorithms (Random Forest, SVM, Neural Networks)
- Add cross-validation for more robust evaluation
- Create a web interface for predictions
- Implement feature importance analysis
- Add data augmentation techniques

---

*This project is part of a comprehensive data science learning journey, demonstrating practical application of machine learning concepts on a classic dataset.*
