# 📧 Spam Email Detection - Machine Learning Project

A comprehensive machine learning project that builds, trains, and compares **7 different classification models** to detect spam emails. This project demonstrates the complete ML pipeline from data preprocessing to model evaluation with detailed visualizations.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Process](#project-process)
- [Models Implemented](#models-implemented)
- [Performance Metrics](#performance-metrics)
- [Key Findings](#key-findings)
- [Visualizations](#visualizations)
- [Installation & Usage](#installation--usage)
- [Results Summary](#results-summary)
- [Lessons Learned](#lessons-learned)

---

## 🎯 Project Overview

This project tackles the classic problem of **spam email detection** using natural language processing (NLP) and machine learning. The goal is to automatically classify emails as either:

- **Spam (1)**: Unwanted, unsolicited emails
- **Ham (0)**: Legitimate, wanted emails

### Why This Matters

- Email users receive billions of spam messages daily
- False positives (legitimate emails marked as spam) are costly
- False negatives (spam getting through) reduce user experience
- Finding the right balance between precision and recall is crucial

---

## 📊 Dataset

### Data Sources

- **Training Data**: Combined from two sources
  - `spam_train1.csv` (2,229 samples)
  - `spam_train2.csv` (2,067 samples)
  - **Total Training**: 4,296 emails
- **Test Data**: 860 emails (20% of training data via train-test split)

### Dataset Characteristics

- **Class Distribution**:
  - Ham (Legitimate): 78.38% (3,367 emails)
  - Spam: 21.62% (929 emails)
- **Imbalanced Dataset**: More ham than spam (realistic scenario)
- **Features**: Email text content
- **Labels**: Binary classification (0=ham, 1=spam)

### Data Preprocessing Steps

1. **Loading**: Merged two CSV files with different formats
2. **Cleaning**: Removed missing/invalid labels
3. **Standardization**: Converted all labels to lowercase
4. **Encoding**: Mapped text labels (spam/ham) to numeric (1/0)
5. **Stratified Split**: 80/20 train-test split maintaining class ratios

---

## 🔄 Project Process

### 1. **Data Preparation**

```python
# Combined two training datasets
# Cleaned and standardized labels
# Handled missing values
# Split into train (80%) and test (20%) sets
```

### 2. **Feature Engineering**

- **Method**: TF-IDF Vectorization (Term Frequency-Inverse Document Frequency)
- **Purpose**: Convert text into numerical features
- **Configuration**:
  - Maximum features: 3,000 words
  - Stop words: Removed common English words
  - Case: All lowercase
  - Min document frequency: 1
- **Output**: 3,000-dimensional feature vectors per email

### 3. **Model Training**

Trained 7 different algorithms:

1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine (SVM)
4. Random Forest
5. Gradient Boosting
6. Decision Tree
7. K-Nearest Neighbors

### 4. **Model Evaluation**

Evaluated each model using comprehensive metrics:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Specificity
- False Positive Rate (Type I Error)
- False Negative Rate (Type II Error)
- ROC-AUC Score

### 5. **Visualization & Analysis**

Created 5 detailed visualizations to compare model performance

---

## 🤖 Models Implemented

### 1. **Logistic Regression**

- **Type**: Linear probabilistic classifier
- **Strengths**: Fast, interpretable, works well for text
- **Best For**: Baseline model, when interpretability matters

### 2. **Naive Bayes (MultinomialNB)**

- **Type**: Probabilistic classifier based on Bayes' theorem
- **Strengths**: Excellent for text classification, fast training
- **Best For**: Text data, small datasets, real-time predictions

### 3. **Support Vector Machine (SVM)**

- **Type**: Margin-based classifier
- **Strengths**: Effective in high-dimensional spaces, robust
- **Best For**: Complex decision boundaries, high-dimensional data

### 4. **Random Forest**

- **Type**: Ensemble of decision trees
- **Strengths**: Handles non-linear relationships, resistant to overfitting
- **Best For**: When you need robustness and don't mind black-box models

### 5. **Gradient Boosting**

- **Type**: Sequential ensemble method
- **Strengths**: High accuracy, handles complex patterns
- **Best For**: Competitions, when accuracy is paramount

### 6. **Decision Tree**

- **Type**: Rule-based classifier
- **Strengths**: Highly interpretable, easy to visualize
- **Best For**: When you need to explain decisions

### 7. **K-Nearest Neighbors (KNN)**

- **Type**: Instance-based learner
- **Strengths**: Simple concept, no training phase
- **Best For**: Small datasets, when similar examples cluster together

---

## 📈 Performance Metrics

### Metrics Explained

| Metric          | Formula               | What It Measures                       | Ideal Value |
| --------------- | --------------------- | -------------------------------------- | ----------- |
| **Accuracy**    | (TP + TN) / Total     | Overall correctness                    | Higher ↑    |
| **Precision**   | TP / (TP + FP)        | How many spam predictions are correct  | Higher ↑    |
| **Recall**      | TP / (TP + FN)        | How much spam we catch                 | Higher ↑    |
| **F1-Score**    | 2 × (P × R) / (P + R) | Balance between precision & recall     | Higher ↑    |
| **Specificity** | TN / (TN + FP)        | How well we identify legitimate emails | Higher ↑    |
| **FPR**         | FP / (FP + TN)        | Legitimate emails marked as spam       | Lower ↓     |
| **FNR**         | FN / (FN + TP)        | Spam that slips through                | Lower ↓     |

**Where:**

- TP = True Positives (Spam correctly identified)
- TN = True Negatives (Ham correctly identified)
- FP = False Positives (Ham marked as spam) ← **Most costly!**
- FN = False Negatives (Spam marked as ham)

---

## 🏆 Key Findings

### Overall Best Model: **Support Vector Machine (SVM)**

```
🥇 F1-Score: 0.9201 (Best balanced performance)
   Accuracy: 96.63%
   Precision: 94.35%
   Recall: 89.78%
   Specificity: 98.52%
   FPR: 0.0148 (Only 1.48% false alarms!)
   FNR: 0.1022 (10.22% spam slipped through)
```

### Model Performance Ranking (by F1-Score)

| Rank | Model               | F1-Score   | Accuracy | Precision | Recall | Best Use Case             |
| ---- | ------------------- | ---------- | -------- | --------- | ------ | ------------------------- |
| 🥇 1 | **SVM**             | **0.9201** | 96.63%   | 94.35%    | 89.78% | **Production deployment** |
| 🥈 2 | Random Forest       | 0.8971     | 95.81%   | 95.73%    | 84.41% | Robust predictions        |
| 🥉 3 | Naive Bayes         | 0.8481     | 93.84%   | 90.80%    | 79.57% | Real-time filtering       |
| 4    | Logistic Regression | 0.8476     | 94.19%   | 97.89%    | 74.73% | High precision needed     |
| 5    | Gradient Boosting   | 0.8000     | 92.33%   | 91.67%    | 70.97% | Complex patterns          |
| 6    | Decision Tree       | 0.7969     | 90.93%   | 77.27%    | 82.26% | Interpretability          |
| 7    | KNN                 | 0.2523     | 81.40%   | 96.43%    | 14.52% | Not recommended           |

### Best Models by Specific Metric

| Metric                | Best Model          | Score  | Why It Matters            |
| --------------------- | ------------------- | ------ | ------------------------- |
| **Highest Accuracy**  | SVM                 | 96.63% | Overall correctness       |
| **Highest Precision** | Logistic Regression | 97.89% | Fewest false alarms       |
| **Highest Recall**    | SVM                 | 89.78% | Catches most spam         |
| **Best F1-Score**     | SVM                 | 0.9201 | Best balance              |
| **Lowest FPR**        | KNN                 | 0.15%  | Rarely marks ham as spam  |
| **Lowest FNR**        | SVM                 | 10.22% | Fewest spam slips through |

---

## 📊 Visualizations

The project generates 5 comprehensive visualizations:

### 1. **Model Comparison Metrics** (`model_comparison_metrics.png`)

- 4-panel comparison showing:
  - Main performance metrics (Accuracy, Precision, Recall, F1)
  - Error rates and specificity
  - F1-Score ranking
  - Accuracy vs Recall scatter plot

### 2. **Confusion Matrices** (`confusion_matrices.png`)

- 7 confusion matrices (one per model)
- Shows True Positives, True Negatives, False Positives, False Negatives
- Color-coded heatmaps for easy interpretation

### 3. **ROC Curves** (`roc_curves.png`)

- ROC curves for all models with AUC scores
- Shows trade-off between True Positive Rate and False Positive Rate
- Higher AUC = better model discrimination

### 4. **Metrics Heatmap** (`metrics_heatmap.png`)

- Color-coded comparison of all metrics across all models
- Green = good performance, Red = poor performance
- Easy to spot strengths and weaknesses

### 5. **Error Analysis** (`error_analysis.png`)

- Detailed analysis of Type I and Type II errors
- Shows which models make fewer costly mistakes
- Critical for production deployment decisions

---

## 💻 Installation & Usage

### Prerequisites

```bash
Python 3.12+
pip (Python package manager)
```

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/courtneyleora/Spam-email-detection.git
cd Spam-email-detection
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install scikit-learn pandas numpy matplotlib seaborn wordcloud
```

4. **Run the project**

```bash
python main.py
```

### Expected Output

- Console output with detailed metrics for each model
- 5 PNG visualizations saved to project directory
- Complete comparison table of all models

---

## 📊 Results Summary

### Confusion Matrix Analysis (Best Model - SVM)

|                 | Predicted Ham | Predicted Spam |
| --------------- | ------------- | -------------- |
| **Actual Ham**  | 664 (TN) ✅   | 10 (FP) ❌     |
| **Actual Spam** | 19 (FN) ❌    | 167 (TP) ✅    |

**Interpretation:**

- **True Negatives (664)**: Correctly identified legitimate emails
- **True Positives (167)**: Correctly caught spam
- **False Positives (10)**: Only 10 legitimate emails wrongly marked as spam
- **False Negatives (19)**: 19 spam emails slipped through

### Error Analysis

**Type I Error (False Positives) - Most Critical**

- SVM: 1.48% (Best among top performers)
- Random Forest: 1.04%
- Logistic Regression: 0.45%

**Type II Error (False Negatives)**

- SVM: 10.22% (Best overall)
- Random Forest: 15.59%
- Naive Bayes: 20.43%

### ROC-AUC Scores

All models showed excellent discrimination:

- SVM: 0.941
- Random Forest: 0.917
- Logistic Regression: 0.871

---

## 🎓 Lessons Learned

### Technical Insights

1. **Feature Engineering Matters**

   - TF-IDF effectively captures word importance
   - Limiting to 3,000 features balanced performance and efficiency

2. **Model Selection**

   - SVM excelled with high-dimensional text data
   - Ensemble methods (Random Forest, Gradient Boosting) were robust
   - Simple models (Naive Bayes, Logistic Regression) performed surprisingly well

3. **Imbalanced Data Handling**

   - Stratified splitting maintained class ratios
   - F1-Score proved more valuable than accuracy for imbalanced data

4. **Evaluation Metrics**
   - Single metrics (accuracy) can be misleading
   - Consider business context (false positives vs false negatives)
   - F1-Score provides good balance for most cases

### Practical Applications

1. **Production Deployment**

   - Choose SVM for best overall performance
   - Use Logistic Regression if precision is critical
   - Consider Naive Bayes for real-time processing

2. **Trade-offs**

   - High precision (fewer false alarms) may reduce recall
   - Balance depends on business requirements
   - User tolerance for errors varies by application

3. **Continuous Improvement**
   - Models need retraining as spam evolves
   - Monitor performance metrics in production
   - Consider ensemble approaches

---

## 🚀 Future Enhancements

### Potential Improvements

1. **Advanced Feature Engineering**

   - Word embeddings (Word2Vec, GloVe)
   - Deep learning (LSTM, BERT)
   - Email metadata features

2. **Model Optimization**

   - Hyperparameter tuning (Grid Search, Random Search)
   - Cross-validation for more robust evaluation
   - Ensemble methods combining multiple models

3. **Data Augmentation**

   - Collect more training data
   - Handle class imbalance (SMOTE, oversampling)
   - Include more diverse spam types

4. **Production Features**
   - Real-time prediction API
   - Model versioning and A/B testing
   - Performance monitoring dashboard

---

## 📚 Technologies Used

- **Python 3.12**
- **scikit-learn**: Machine learning algorithms and metrics
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **wordcloud**: Text visualization (optional)

---

## 🤝 Contributing

This is a learning project, but suggestions are welcome! Feel free to:

- Open issues for bugs or improvements
- Submit pull requests with enhancements
- Share your own findings and experiments

---

## 📝 License

This project is open source and available for educational purposes.

---

## 👤 Author

**Courtney Ross**

- GitHub: [@courtneyleora](https://github.com/courtneyleora)
- Email: cross52@student.gsu.edu

---

## 🙏 Acknowledgments

- Dataset sourced from spam email collections
- Inspired by classic spam filtering problems
- Built as a first machine learning project to learn the complete ML pipeline

---

## 📖 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Understanding TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [ROC Curves Explained](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Precision vs Recall](https://en.wikipedia.org/wiki/Precision_and_recall)

---

**⭐ If you found this project helpful, please consider giving it a star!**
