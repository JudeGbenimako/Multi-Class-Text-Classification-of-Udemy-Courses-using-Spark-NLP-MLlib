# Multi-Class-Text-Classification-of-Udemy-Courses-using-Spark-NLP-MLlib

**Project Title**

Multi-Class Text Classification of Udemy Courses using Spark NLP & MLlib

**Author**

**Jude Gbenimako**

NOVA Information Management School
Master’s Degree Program in Data Science and Advanced Analytics (Business Analytics)
June 2025

**Project Overview**

This project tackles the challenge of categorizing Udemy online course titles into distinct subject categories using Natural Language Processing (NLP) and machine learning (ML) techniques.
It addresses the problem of course misclassification which negatively affects user experience and discoverability on educational platforms.

**Objectives**

To preprocess and clean Udemy course titles effectively.

To extract meaningful textual features using NLP methods such as tokenization, stopword removal, bigrams, and TF-IDF.

To develop and compare ML classification models (Logistic Regression, Naive Bayes, and Random Forest) using Spark MLlib.

To evaluate model performance with accuracy, precision, recall, and F1 score metrics.

To identify the best performing model for production use.

To test model predictions on new course titles in both single and batch modes.

**Dataset**

Source: Publicly available Udemy dataset on Kaggle.

Size: 98,104 rows with 13 columns including course ID, title, and category.

Categories: Initially 13 categories, later merged to 6 broader categories (Business, Tech, Education, Health & Lifestyle, Creative Arts, Marketing) to mitigate class imbalance and improve model performance.

**Methodology**

Data Cleaning: Lowercasing, punctuation removal, whitespace trimming.

Feature Engineering: Tokenization, stopword removal, bigram creation, count vectorization, TF-IDF weighting.

Model Training: Stratified train-test split (80/20) preserving class distribution.

Models Compared: Logistic Regression, Naive Bayes, Random Forest.

Pipeline created using Spark NLP and Spark MLlib.

Label encoding applied for numeric model labels and decoded post-prediction for interpretability.

**Tools & Technologies**

Apache Spark (DataFrames, MLlib, NLP)

Databricks Lakehouse for scalable data storage and processing

Python for orchestration and custom UDFs

Visualization libraries: Seaborn, Matplotlib (inside Databricks); WordCloud generated externally using Google Colab due to Databricks limitations.

**Results**

Naive Bayes performed best with a weighted F1 score of ~0.805.

Logistic Regression closely followed with a weighted F1 score of ~0.798.

Random Forest performed poorly on this text classification task with low F1 score (~0.19).

Visualization of confusion matrices highlighted model strengths and areas of confusion.

WordCloud visualizations provided insights on category-specific keywords and common noise words.

**Enhancements after Feedback**

Dataset expanded from a smaller sample (3,678 rows) to the full 98,104 rows for better representativeness and reliability.

WordCloud visualizations added to better communicate key textual features per category, generated externally to overcome platform constraints.

**How to Run**

Load the Udemy dataset into the Spark environment.

Perform preprocessing and feature extraction as per the pipeline.

Train and evaluate models using the provided code pipeline.

Use the decoding function to interpret numeric predictions.

Test models with new course title inputs for classification.

**Future Work**

Further refinement of preprocessing steps to better handle noisy or ambiguous course titles.

Explore additional feature extraction methods or deep learning models for improved accuracy.

Automate deployment pipeline for real-time course classification on Udemy or similar platforms.

