# 📱 Mobile Usage Pattern Prediction and User Segmentation
## 🧠 Project Overview

This project analyzes mobile usage behavior to understand user patterns and predict their primary phone usage (e.g., communication, entertainment, productivity, etc.) using machine learning. It also performs clustering-based segmentation to group users with similar behavior patterns.

A Streamlit web app is developed to:

- Explore interactive EDA visualizations

- Perform Primary Use Prediction through user input

- Display Clustering and Segmentation Results

## 🧩 Key Objectives

1. Data Understanding & Preparation

    - Clean and preprocess raw mobile usage data.

    - Handle missing values, scale features, and encode categorical variables.

2. Exploratory Data Analysis (EDA)

   - Identify key trends in screen time, app categories, battery usage, etc.

   - Visualize relationships among features using interactive charts.

3. Machine Learning Model – Primary Use Prediction

   - Predict the primary usage category using a Decision Tree Classifier.

   - Save the trained model as decision_tree_model.pkl for real-time predictions.

4. Unsupervised Learning – Clustering

   * Perform segmentation using:

        * K-Means

        * Hierarchical Clustering

        * DBSCAN

        * Gaussian Mixture Model (GMM)

        * Spectral Clustering

   * Compare results and interpret user segments.

5. Streamlit Application

   * Visual EDA Dashboard

   * Primary Use Prediction via user inputs

   * Clustering visualization and user segmentation insights

## 🧮 Data Preparation & Feature Engineering

- Removed missing and duplicate records

- Encoded categorical variables using LabelEncoder
  
- Scaled numerical features using StandardScaler

- Feature selection based on correlation and domain relevance

- Split dataset into training and testing sets for model building


## 📊 Exploratory Data Analysis (EDA)

The EDA module in the Streamlit app allows users to:

- Select variables for comparison

- View distributions, boxplots, correlations, and pairplots

- Analyze top features contributing to mobile usage behavior

Example Visuals:

- Screen time vs. battery drain

- Most frequently used app category

- Correlation heatmap of features

- Cluster visualizations using PCA-reduced dimensions

## 🤖 Model Training – Primary Use Classification

- Model Used: Decision Tree Classifier

- Target Variable: Primary_Use

- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

## 🧭 Clustering & Segmentation

Performed unsupervised learning using:

   1. KMeans → Clear segmentation of usage patterns

   2. Hierarchical Clustering → Dendrogram for relationship visualization

  3. DBSCAN → Density-based identification of core and noise points

  4. GMM → Probabilistic cluster assignments

  5. Spectral Clustering → Community-based segmentation for non-linear patterns

Each method provides different insights into user groups (e.g., heavy gamers, casual users, productivity-focused users).


## 💻 Streamlit App Features
### 🧩 1. Dashboard – EDA Visualizations

- Dropdowns to select features dynamically

- Interactive visualizations using Plotly

- Displays distribution plots, heatmaps, and scatter plots

### 🔮 2. Primary Use Prediction

- User-friendly form to input values for features used in training

- Predicts Primary Use Category using pre-trained Decision Tree model

- Displays prediction with a confidence score

### 🧱 3. Clustering & Segmentation

- Allows user to select clustering algorithm

- Displays PCA-reduced 2D cluster plots

- Provides interpretation of each cluster

## 📈 Insights and Findings

- Users can be segmented into distinct behavioral groups:

    - Cluster 1: High screen time & entertainment-heavy users

    - Cluster 2: Low battery consumption, moderate app usage (balanced users)

    - Cluster 3: Communication-centric users with frequent app switching

- Model achieved high accuracy in predicting primary usage behavior.

- EDA revealed strong correlations between screen time, app usage, and data consumption.

## 🔮 Future Enhancements

- Integrate real-time mobile usage tracking API for live predictions

- Deploy the Streamlit app using Streamlit Cloud or AWS EC2

- Improve model performance using Ensemble Models (Random Forest, XGBoost)

- Add user feedback analytics to refine segmentation

## 🏁 Conclusion

This project provides a complete end-to-end pipeline for analyzing, predicting, and segmenting mobile user behavior using data science and machine learning.
It serves as an analytical tool for understanding digital habits, app usage trends, and user engagement patterns.
