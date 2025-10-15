# ML-Internship-Portfolio
# Project 1: California Housing Price Prediction (End-to-End ML Workflow)

This project serves as a comprehensive, hands-on implementation of an end-to-end Machine Learning workflow, based on Chapter 2 of Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. The goal is to predict the median house value in California districts, given a number of features from the 1990 census data.

This project is not just about building a model; it's about internalizing the complete process of a professional ML engineer: from data exploration and preparation to model training, evaluation, and fine-tuning.

## 🎯 Project Objective

To build a regression model that accurately predicts the median housing price in California districts and to master the entire Machine Learning project lifecycle.

## 📊 Dataset

The dataset used is the California Housing dataset, which contains metrics such as population, median income, median housing age, etc., for each census block group in California.

## ⚙️ Workflow & Techniques Implemented

This project follows a structured and professional workflow, emphasizing best practices at each stage:

1.  **Data Loading & Initial Exploration (EDA):** Loaded the dataset and performed an initial analysis using `.info()`, `.describe()`, and visualizations to understand its structure, identify missing values, and observe data distributions.

2.  **Data Visualization:** Created geographical scatterplots to visualize housing prices by location and used correlation matrices (`heatmap`) to identify key predictive features, such as `median_income`.

3.  **Stratified Sampling:** Implemented a **stratified split** based on the `median_income` categories. This is a crucial step to ensure that both the training and test sets are representative of the overall data distribution, preventing sampling bias.

4.  **Advanced Feature Engineering:**
    * Created meaningful new features by combining existing ones (e.g., `rooms_per_house`, `bedrooms_ratio`).
    * Applied transformations to handle skewed distributions (e.g., **log transform** on `population`).
    * Explored techniques like **bucketizing** for multimodal distributions using Radial Basis Functions (`rbf_kernel`).

5.  **Robust Preprocessing Pipeline:**
    * Constructed a full preprocessing pipeline using Scikit-Learn's `Pipeline` and `ColumnTransformer`.
    * This pipeline automates the cleaning and preparation steps for any new data, ensuring consistency and preventing data leakage. It handles:
        * **Imputation:** Filling missing numerical values using `SimpleImputer`.
        * **Categorical Encoding:** Converting text features to numbers using `OneHotEncoder`.
        * **Feature Scaling:** Standardizing numerical features with `StandardScaler`.
        * **Custom Transformations:** Integrating custom-built transformers and `FunctionTransformer` for advanced feature engineering.

6.  **Model Training & Evaluation:**
    * Trained and evaluated multiple regression models, including **Linear Regression**, **Decision Tree**, and **Random Forest**.
    * Identified the issue of **overfitting** with the Decision Tree model by comparing training error (RMSE=0) with evaluation error.
    * Used **K-Fold Cross-Validation** (`cross_val_score`) to get a robust and reliable measure of each model's performance, avoiding dependency on a single train-validation split.

7.  **Hyperparameter Tuning:**
    * Fine-tuned the best-performing model (Random Forest) to optimize its performance.
    * Implemented both **`GridSearchCV`** and **`RandomizedSearchCV`** to efficiently search the hyperparameter space.

8.  **Final Evaluation & Analysis:**
    * Evaluated the final, tuned model on the unseen **test set** to get a definitive performance score.
    * Calculated a **95% confidence interval** for the final RMSE to understand the model's expected performance range in production.
    * Inspected **feature importances** to understand which attributes were most influential in the model's predictions.

## 🚀 Key Learnings

This project solidified my understanding of the entire ML project lifecycle. The key takeaways were:
* The critical importance of a **robust data splitting strategy** like stratified sampling.
* The significant performance gains achieved through thoughtful **feature engineering**.
* The power and efficiency of building a **reusable preprocessing pipeline** with Scikit-Learn.
* The necessity of **cross-validation** for obtaining a reliable estimate of a model's generalization error.
* The practical application of **hyperparameter tuning** to move from a good baseline model to an optimized one.

## 💾 Saving the Model

The final trained model, including its complete preprocessing pipeline, was saved using `joblib` for potential deployment.
