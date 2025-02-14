# Spam Detection using Naive Bayes and PyCaret

## Overview

This project aims to classify SMS messages as either "spam" or "ham" (non-spam) using machine learning techniques. The dataset consists of labeled SMS messages, and various text preprocessing steps and machine learning models are applied to build a spam classifier.

## Steps Involved

### 1. **Data Exploration and Preprocessing**
   - Loaded a dataset of SMS messages labeled as "spam" or "ham".
   - Explored the dataset by checking basic statistics and distribution of characters, words, and sentences.
   - Preprocessed the text by:
     - Tokenizing the text and removing special characters.
     - Removing stopwords.
     - Applying stemming to reduce words to their root form.

### 2. **Text Vectorization**
   - Used the `TfidfVectorizer` from `sklearn` to convert the text data into numerical features.
   - This transformation results in a sparse matrix where each row corresponds to an SMS, and each column corresponds to the TF-IDF score of a specific word in the corpus.

### 3. **Model Building**
   - Split the data into training and testing sets (80% training, 20% testing).
   - Trained multiple Naive Bayes classifiers:
     - **Gaussian Naive Bayes**: Achieved an accuracy of 87.62% and a precision score of 0.523.
     - **Multinomial Naive Bayes**: Achieved an accuracy of 95.94% and a precision score of 1.0.
     - **Bernoulli Naive Bayes**: Achieved an accuracy of 97.0% and a precision score of 0.973.

### 4. **Model Evaluation**
   - Evaluated each model's performance using:
     - Accuracy Score
     - Precision Score
     - Confusion Matrix

### 5. **PyCaret Integration**
   - Used **PyCaret**, a machine learning library, to automate the process of model comparison and selection.
   - The best model was selected using `compare_models()` from PyCaret.


## Dependencies

- `nltk` (for text processing)
- `sklearn` (for machine learning models)
- `pycaret` (for model comparison)
- `seaborn` (for data visualization)
- `matplotlib` (for plotting)
- `pandas` (for data manipulation)

To install dependencies:

```bash
pip install nltk sklearn pycaret seaborn matplotlib pandas
```

## Files

- **spam_detection.ipynb**: Main Jupyter notebook containing the entire workflow of data exploration, preprocessing, model training, and evaluation.
- **spam.csv**: Dataset

## Instructions

1. Clone the repository or download the notebook.
2. Install the required dependencies.
3. Run the Jupyter notebook (`spam_detection.ipynb`) step by step to follow the entire workflow.
4. Check the accuracy and performance of different Naive Bayes models and PyCaret model comparison.

## Conclusion

The project successfully demonstrates spam detection using traditional machine learning models. The **Bernoulli Naive Bayes** classifier performs the best with an accuracy of 97.0%. PyCaret provides an automated approach to model comparison and selection, making the process efficient.
