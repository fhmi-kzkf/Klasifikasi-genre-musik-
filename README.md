# 🎵 Music Genre Classification Dashboard

A Streamlit web application for classifying music genres based on audio features. This application allows you to upload a dataset, compare multiple machine learning algorithms, and predict music genres with high accuracy.

## 🚀 Features

- **Multiple ML Models**: Compare performance of 8 different machine learning algorithms
- **Feature Selection**: Automatically select the most relevant features for classification
- **Cross-Validation**: Evaluate model performance with cross-validation
- **Ensemble Method**: Voting classifier that combines multiple models for better performance
- **Detailed Evaluation**: Confusion matrices, classification reports, and accuracy metrics
- **Prediction Pipeline**: Upload new data for genre prediction
- **Export Results**: Download predictions as CSV file

## 📊 Supported Models

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Naive Bayes
6. Gradient Boosting
7. Neural Network (MLP)
8. Ensemble (Voting Classifier)

## 📁 Dataset Format

The application expects a CSV file with the following structure:
- Multiple feature columns (numeric values)
- Last column should contain the genre labels

Example:
```
feature1,feature2,feature3,...,genre
0.123,0.456,0.789,...,rock
0.234,0.567,0.890,...,pop
...
```

## 🛠️ Installation

1. Clone or download this repository
2. Install the required packages:

```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
```

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open your browser to the URL provided (typically http://localhost:8501)

## 🎯 Usage

1. **Upload Dataset**: Click "Browse files" to upload your CSV dataset
2. **Feature Selection**: Adjust the number of features to use (5-50)
3. **Model Training**: The app will automatically train all models
4. **View Results**: See accuracy scores and confusion matrices for each model
5. **Make Predictions**: Upload a new CSV file to predict genres
6. **Download Results**: Get your predictions as a CSV file

## 📈 Expected Performance

With the enhanced feature selection and ensemble methods, the application should achieve accuracy above 80% for most music genre classification tasks. The app will display a warning if accuracy falls below this threshold with suggestions for improvement.

## 🧠 Technical Details

- **Feature Selection**: Uses SelectKBest with f_classif scoring
- **Data Preprocessing**: StandardScaler for feature normalization
- **Evaluation**: Cross-validation (optional) and train/test split
- **Label Encoding**: Automatically handles categorical labels

## 📝 Notes

- The application assumes the last column in your dataset contains the genre labels
- Only numeric features are used for training
- For best results, ensure your dataset is balanced across genres
- The ensemble method typically provides the best performance by combining multiple models

## 🤝 Contributing

Feel free to fork this project and submit pull requests with improvements. Suggestions for enhancements include:
- Additional feature selection methods
- More advanced ML models
- Improved visualization options
- Additional evaluation metrics

## 📄 License

This project is open source and available under the MIT License.