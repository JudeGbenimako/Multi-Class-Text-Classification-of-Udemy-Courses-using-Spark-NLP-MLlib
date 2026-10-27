# Multi-Class Udemy Course Classification with PySpark

**Independent academic project | Big Data Analytics | NOVA IMS | June 2025**

## Overview

This project uses natural language processing (NLP) and distributed machine learning in **PySpark** to classify Udemy course titles into six subject areas. The goal is to explore how automated text classification could support clearer course categorization and discoverability on online learning platforms.

Developed independently for the MSc in Data Science and Advanced Analytics at NOVA Information Management School (NOVA IMS).

## Results at a glance

The best-performing model was **multinomial Naive Bayes** using unigram and bigram features with TF-IDF weighting.

| Classifier | Test accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| **Naive Bayes** | **0.8055** | **0.8052** | **0.8025** |
| Logistic Regression | 0.7973 | 0.7980 | 0.7961 |
| Random Forest | 0.2780 | 0.1906 | Not reported |

The reported results were obtained using an approximately **80/20 stratified train/test split**. The results describe an academic evaluation; they do not measure improvements in Udemy's real-world discovery or recommendation outcomes.

## Dataset

- **Source:** [Udemy Course Dataset — Categories, Ratings and Trends (Kaggle)](https://www.kaggle.com/datasets/emrebayirr/udemy-course-dataset-categories-ratings-and-trends/data)
- **Original dataset:** 98,104 records with 13 columns
- **After title cleaning:** 98,073 records reported in the notebook
- **Target:** Course subject, consolidated from 13 original categories into six broader classes

| Final class | Example original categories |
|---|---|
| Tech | Development; IT & Software |
| Business | Business; Finance & Accounting; Office Productivity |
| Education | Teaching & Academics; Personal Development |
| Creative Arts | Design; Music; Photography & Video |
| Health & Lifestyle | Health & Fitness; Lifestyle |
| Marketing | Marketing |

## Approach

1. **Exploration and cleaning:** Inspected category distributions and course titles, handled unwanted characters, standardized text, and removed malformed titles.
2. **Category consolidation:** Merged related source categories into six classes and evaluated class imbalance.
3. **Text preprocessing:** Applied tokenization and English stopword removal.
4. **Feature engineering:** Combined unigrams and bigrams, then used CountVectorizer and inverse document frequency (TF-IDF) weighting to construct sparse numerical features.
5. **Modeling:** Used the PySpark ML Pipeline API to compare Logistic Regression, multinomial Naive Bayes, and Random Forest.
6. **Evaluation:** Reported accuracy, weighted precision and recall, weighted F1, macro F1, class-level scores, and confusion matrices.
7. **Inference:** Tested predictions on new individual course titles and batches of example titles.

## Tools and environment

- **Python, PySpark, Apache Spark:** Spark SQL DataFrames, transformations, and the DataFrame-based MLlib API (`pyspark.ml`)
- **NLP features:** `Tokenizer`, `StopWordsRemover`, `NGram`, `CountVectorizer`, and `IDF` from `pyspark.ml.feature`
- **Models:** `LogisticRegression`, `NaiveBayes`, and `RandomForestClassifier` from `pyspark.ml.classification`
- **Databricks:** Databricks Community Edition with Databricks Runtime 12.2 LTS (Spark 3.3.2), as documented in the original notebook
- **Visuals:** Matplotlib and Seaborn; word clouds were prepared separately using Google Colab

**Technical naming note:** The project applies *NLP techniques using PySpark ML*. It does not import the separate third-party **Spark NLP** library. “PySpark ML” therefore describes the implementation more precisely than “Spark NLP” as a package name.

## Repository files

- [Multi-Class Text Classification of Udemy Courses using Spark NLP & MLlib.ipynb](Multi-Class%20Text%20Classification%20of%20Udemy%20Courses%20using%20Spark%20NLP%20%26%20MLlib.ipynb) — Exploration, preprocessing, model comparisons, evaluation, and inference

## Running the notebook

The original workflow was developed in Databricks, where a Spark session (`spark`) and Databricks display utilities were available by default.

1. Set up a compatible Spark/Databricks environment.
2. Download the source dataset from the Kaggle link above, subject to the dataset's terms.
3. Upload the data to the workspace and update the notebook's original local path (`/FileStore/tables/udemy__courses.csv`) if necessary.
4. Run the notebook cells in order to reproduce the preprocessing, modeling, and evaluation workflow.

The notebook has not been independently re-executed in a new environment for this GitHub documentation update. Environment and package differences may require adjustments.

## Conclusions and limitations

Naive Bayes offered the strongest reported results for this six-class categorization task, while Logistic Regression performed comparably and Random Forest substantially worse. The project illustrates how sparse text features and Spark ML pipelines can support multiclass classification at this dataset scale.

These evaluations do **not** establish a deployed Udemy integration, a recommendation engine, or measured improvements in course discoverability. A production use case would require additional validation, monitoring, and testing on newer or more ambiguous course titles.

## Author and academic context

**Jude Gbenimako** — Independent project, Group 40  
NOVA Information Management School (NOVA IMS)  
MSc in Data Science and Advanced Analytics, Business Analytics specialization  
Big Data Analytics | June 2025
