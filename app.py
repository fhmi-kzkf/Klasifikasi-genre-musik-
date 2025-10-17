import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

st.title("🎵 Music Genre Classification Dashboard")
st.write("Upload your dataset and compare multiple ML algorithms for genre classification.")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Dataset")
    st.dataframe(df.head())
    
    # Show dataset info
    st.subheader("Dataset Information")
    st.write(f"Dataset shape: {df.shape}")
    st.write(f"Number of genres: {df.iloc[:, -1].nunique()}")
    st.write("Class distribution:")
    st.write(df.iloc[:, -1].value_counts())

    # Select target column (assuming last column is the label)
    target_col = df.columns[-1]
    st.write(f"Target column selected: {target_col}")

    # Remove target from features
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Show feature information
    st.write(f"Number of features: {X.shape[1]}")

    # Encode labels if categorical
    label_encoder_used = False
    le = None
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_encoder_used = True

    # Feature selection
    st.subheader("Feature Selection")
    k_features = st.slider("Select number of best features", 5, min(50, X.shape[1]), min(20, X.shape[1]))
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Show selected features
    if st.checkbox("Show selected features"):
        st.write(selected_features)

    # Split data
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(max_iter=500, random_state=42)
    }

    # Add ensemble method
    ensemble = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ], voting='soft')
    models["Ensemble (Voting)"] = ensemble

    results = {}
    cv_results = {}
    
    st.subheader("Model Training & Evaluation")

    # Cross-validation option
    use_cv = st.checkbox("Use Cross-Validation", value=True)
    cv_folds = 5
    if use_cv:
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)

    for name, model in models.items():
        with st.spinner(f'Training {name}...'):
            if use_cv:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                cv_results[name] = cv_scores.mean()
            
            # Train on full training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            
            # Display results
            if use_cv:
                st.write(f"**{name}** - Accuracy: {acc:.3f} (CV: {cv_results[name]:.3f})")
            else:
                st.write(f"**{name}** - Accuracy: {acc:.3f}")

            # Show detailed classification report for best model
            best_model_name_key = max(results.keys(), key=lambda k: results[k])
            if name == best_model_name_key:
                st.write("Classification Report:")
                if label_encoder_used and le is not None:
                    target_names = le.classes_
                    st.text(classification_report(y_test, y_pred, target_names=target_names))
                else:
                    st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix - {name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    # Comparison plot
    st.subheader("Model Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = list(results.values())
    
    bars = ax.bar(model_names, accuracies)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparison of Models")
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Show best model
    best_model_name_key = max(results.keys(), key=lambda k: results[k])
    best_accuracy = results[best_model_name_key]
    st.subheader("Best Performing Model")
    st.success(f"**{best_model_name_key}** with accuracy: {best_accuracy:.3f}")
    
    # Warning if accuracy is below 80%
    if best_accuracy < 0.8:
        st.warning(f"Accuracy is below 80% ({best_accuracy:.1%}). Consider:")
        st.markdown("""
        - Collecting more data
        - Engineering additional features
        - Trying different preprocessing techniques
        - Using more advanced models
        """)
    else:
        st.success(f"🎉 Great! Accuracy is above 80% ({best_accuracy:.1%})")

    # Prediction on new dataset
    st.subheader("Upload New Dataset for Prediction")
    new_file = st.file_uploader("Upload new CSV for prediction", type=["csv"], key="new")

    if new_file is not None:
        new_df = pd.read_csv(new_file)
        st.write("Preview new dataset:")
        st.dataframe(new_df.head())

        # Ensure only numeric columns are used
        new_df_features = new_df.select_dtypes(include=[np.number])
        
        # Apply feature selection
        new_df_selected = new_df_features.reindex(columns=selected_features, fill_value=0)
        new_scaled = scaler.transform(new_df_selected)

        best_model = models[best_model_name_key]
        preds = best_model.predict(new_scaled)

        if label_encoder_used and le is not None:
            preds = le.inverse_transform(preds)

        new_df['Predicted Genre'] = preds
        st.write("Predictions:")
        st.dataframe(new_df)

        # Download predictions
        csv = new_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")