Iris Flower Classification using Machine Learning

# Iris Classification 🌸

This project classifies iris flowers into Setosa, Versicolor, and Virginica using various machine learning models (KNN, SVM, Logistic Regression, etc.).

📁 Files
- `iris.ipynb`: Main Jupyter Notebook with code, visualizations, and model training.
- `Iris.csv`: Dataset from UCI Repository.

📊 Tools & Libraries
- Python, Pandas, Seaborn, scikit-learn, Matplotlib

📈 Models Used
- KNN, Decision Tree, Random Forest, SVM, Logistic Regression

📌 Features Used
- SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm


DETAILS:

This project is part of the OIBSIP Internship. It involves applying machine learning techniques to classify iris flowers into three species — Setosa, Versicolor, and Virginica — based on their sepal and petal measurements.

---

## 📂 Dataset Used

- Dataset Name: `Iris.csv`
- Source: UCI Machine Learning Repository
- Columns:
  - `SepalLengthCm`
  - `SepalWidthCm`
  - `PetalLengthCm`
  - `PetalWidthCm`
  - `Species`

---

## 🛠 Tools & Technologies

- Python
- Jupyter Notebook
- NumPy, Pandas
- Seaborn, Matplotlib
- scikit-learn (sklearn)

---

## 📊 Project Flow

### 1️⃣ Data Exploration
- Loaded dataset using `pandas`
- Viewed basic structure using `.head()`, `.info()`, `.describe()`
- Checked missing values and class distribution

### 2️⃣ Data Visualization
- Countplot for species
- Scatter plots (Sepal & Petal)
- Pairplots and Histograms
- Correlation matrix and boxplots

### 3️⃣ Preprocessing
- Standardized the features using `StandardScaler`
- Train-test split (80:20)

### 4️⃣ Model Training & Testing
Trained and compared multiple models:
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Logistic Regression

### 5️⃣ Model Evaluation
- Accuracy Score
- Classification Report
- Confusion Matrix (visualized)

### 6️⃣ Hyperparameter Tuning
- GridSearchCV applied for optimal `n_neighbors` in KNN

### 7️⃣ PCA Visualization
- Applied Principal Component Analysis (PCA)
- Plotted 2D visualization of reduced features

---

## 📌 Sample Code Snippet

``python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


✅ Output Example
Model Accuracy: 97%+

Confusion Matrix:


<img width="451" height="541" alt="image" src="https://github.com/user-attachments/assets/cfae09f4-7c0f-4659-a504-c04fc49108c9" />

Sepal Width vs Length:

<img width="579" height="459" alt="Sepal_WL" src="https://github.com/user-attachments/assets/4f2dd115-855d-4ce3-a8d3-4e7858666ad8" />

Feature Importance

<img width="520" height="311" alt="Feature Importance" src="https://github.com/user-attachments/assets/e304648d-1b9f-4dc6-bbad-fe061666e9b2" />

PCA Iris Dataset

<img width="573" height="455" alt="PCA_Iris_Dataset" src="https://github.com/user-attachments/assets/58bc4885-58ed-424e-b97f-eff8d6cbb82a" />


🙌 Conclusion
This project demonstrates:

Data preprocessing

Multiple ML model comparison

Hyperparameter tuning

Interpretation of model results


## 🔗 Reference

* [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
* [scikit-learn documentation](https://scikit-learn.org/)


