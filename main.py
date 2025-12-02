# ============================================================================
# SPAM EMAIL DETECTION WITH COMPREHENSIVE MODEL COMPARISON
# ============================================================================
# This project builds and compares multiple machine learning models to detect
# spam emails. It trains 7 different algorithms and evaluates their performance
# using various metrics to find the best classifier.
# ============================================================================

# Import necessary libraries for data manipulation and analysis
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For data manipulation and CSV file handling
import matplotlib.pyplot as plt  # For creating plots and visualizations
import seaborn as sns  # For enhanced statistical visualizations
from wordcloud import WordCloud, STOPWORDS  # For creating word clouds (optional feature)
import warnings  # To suppress warning messages for cleaner output

# Suppress warnings to keep output clean (e.g., deprecation warnings)
warnings.filterwarnings("ignore")

# Sklearn (Scikit-learn) imports for machine learning functionality
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into numerical features

# Import metrics to evaluate model performance
from sklearn.metrics import (
    classification_report,  # Detailed report of precision, recall, f1-score
    confusion_matrix,  # Shows true/false positives and negatives
    accuracy_score,  # Percentage of correct predictions
    precision_score,  # Proportion of spam predictions that are actually spam
    recall_score,  # Proportion of actual spam that was caught
    f1_score,  # Harmonic mean of precision and recall (balanced metric)
    roc_auc_score,  # Area under ROC curve (model discrimination ability)
    roc_curve,  # For plotting ROC curves
)

# Import 7 different machine learning algorithms to compare
from sklearn.linear_model import LogisticRegression  # Linear model for binary classification
from sklearn.naive_bayes import MultinomialNB  # Probabilistic classifier, good for text
from sklearn.svm import SVC  # Support Vector Machine for classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble methods
from sklearn.tree import DecisionTreeClassifier  # Tree-based classifier
from sklearn.neighbors import KNeighborsClassifier  # Instance-based learning algorithm

# Configure visualization style for better-looking plots
sns.set_style("whitegrid")  # Set seaborn style with grid background
plt.rcParams["figure.figsize"] = (12, 8)  # Set default figure size for all plots


# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
# In this section, we load the training data from CSV files and prepare it
# for machine learning by cleaning and standardizing the format.
# ============================================================================

print("=" * 80)
print("LOADING DATA...")
print("=" * 80)

# Load the two training datasets from CSV files
# df1 contains spam_train1.csv (different format with v1, v2 columns)
# df2 contains spam_train2.csv (different format with label, text columns)
df1 = pd.read_csv("spam_train1.csv")
df2 = pd.read_csv("spam_train2.csv")

# Display the column names to understand the structure of each dataset
print(f"\nTrain1 columns: {df1.columns.tolist()}")
print(f"Train2 columns: {df2.columns.tolist()}")

# Standardize train1 format: rename columns to match train2
# Check if train1 has the old format (v1=label, v2=message text)
if "v1" in df1.columns and "v2" in df1.columns:
    df1 = df1[["v1", "v2"]].copy()  # Keep only relevant columns
    df1.columns = ["label", "text"]  # Rename to standard format

# Standardize train2 format: keep only label and text columns
if "label" in df2.columns and "text" in df2.columns:
    df2 = df2[["label", "text"]].copy()

# Combine both training datasets into one DataFrame
# ignore_index=True creates a new continuous index for the combined data
data = pd.concat([df1, df2], ignore_index=True)

# Replace any NaN (missing) values with empty strings to avoid errors
data = data.where((pd.notnull(data)), "")

print(f"\nCombined training data shape: {data.shape}")
print(f"Sample data:\n{data.head()}")  # Show first 5 rows

# ============================================================================
# CLEAN AND STANDARDIZE LABELS
# ============================================================================
# Labels might have inconsistent formatting (uppercase, spaces, etc.)
# We need to convert "spam"/"ham" text labels into numeric format (1/0)
# ============================================================================

# Convert labels to string and clean them (lowercase, remove whitespace)
data["label"] = data["label"].astype(str).str.strip().str.lower()

# Replace invalid values (empty strings, 'none', 'nan') with pandas NA (missing)
data["label"].replace({"": pd.NA, "none": pd.NA, "nan": pd.NA}, inplace=True)

# Create a mapping dictionary to convert text labels to numbers
# spam = 1 (positive class - what we want to detect)
# ham = 0 (negative class - legitimate emails)
label_map = {"spam": 1, "ham": 0}
data["label"] = data["label"].map(label_map)

# Find rows with missing or invalid labels (couldn't be mapped)
missing_mask = data["label"].isna()
if missing_mask.any():
    print(f"\nDropping {missing_mask.sum()} rows with invalid/missing labels")
    # Keep only rows where label is not missing (~missing_mask means "not missing")
    data = data.loc[~missing_mask].reset_index(drop=True)

# Display dataset information after cleaning
print(f"\nAfter cleaning - shape: {data.shape}")
print(f"\nLabel distribution:\n{data['label'].value_counts()}")  # Count of spam vs ham
print(f"\nLabel percentages:\n{data['label'].value_counts(normalize=True) * 100}")

# ============================================================================
# PREPARE FEATURES (X) AND LABELS (Y)
# ============================================================================
# X = input features (the email text)
# Y = output labels (spam=1 or ham=0)
# ============================================================================

# X contains all email text messages (input features)
X = data["text"].astype(str)

# Y contains the corresponding labels (0 or 1) for each email
Y = data["label"].astype(int)

# ============================================================================
# SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
# We need to split our data to:
# 1. Train the model on training data (80%)
# 2. Evaluate the model on unseen testing data (20%)
# This prevents overfitting and tests real-world performance
# ============================================================================

# train_test_split randomly divides data into train and test sets
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 ensures reproducibility (same split every time)
# stratify=Y ensures both sets have similar spam/ham ratios
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTrain label distribution:\n{Y_train.value_counts()}")
print(f"Test label distribution:\n{Y_test.value_counts()}")


# ============================================================================
# 2. FEATURE EXTRACTION (CONVERTING TEXT TO NUMBERS)
# ============================================================================
# Machine learning models can't work with raw text - they need numbers!
# We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert
# text into numerical features that represent the importance of words.
#
# How TF-IDF works:
# - TF (Term Frequency): How often a word appears in a document
# - IDF (Inverse Document Frequency): How unique/rare a word is across all documents
# - Common words like "the", "is" get lower scores
# - Unique, informative words get higher scores
# ============================================================================

print("\n" + "=" * 80)
print("EXTRACTING FEATURES WITH TF-IDF...")
print("=" * 80)

# Create a TF-IDF vectorizer with specific parameters
# min_df=1: Include words that appear in at least 1 document
# stop_words='english': Remove common English words (the, is, and, etc.)
# lowercase=True: Convert all text to lowercase for consistency
# max_features=3000: Keep only the 3000 most important words
feature_extractor = TfidfVectorizer(
    min_df=1, stop_words="english", lowercase=True, max_features=3000
)

# fit_transform: Learn vocabulary from training data AND transform it to numbers
# This creates a matrix where each row is an email and each column is a word's TF-IDF score
X_train_features = feature_extractor.fit_transform(X_train)

# transform: Convert test data using the SAME vocabulary learned from training
# We don't use fit_transform here to avoid data leakage
X_test_features = feature_extractor.transform(X_test)

print(f"\nFeature matrix shape: {X_train_features.shape}")
print(f"Number of unique words: {len(feature_extractor.vocabulary_)}")
print(
    f"  Each email is represented by {X_train_features.shape[1]} numerical features"
)


# ============================================================================
# 3. TRAIN MULTIPLE MACHINE LEARNING MODELS
# ============================================================================
# We'll train 7 different algorithms and compare their performance.
# Each algorithm has different strengths and approaches to classification.
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING MULTIPLE MODELS...")
print("=" * 80)

# Define a dictionary of models to train and compare
# Each model uses different mathematical approaches to classify emails
models = {
    # 1. Logistic Regression: Linear model that predicts probability
    #    Good for: Binary classification, interpretable results
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    
    # 2. Naive Bayes: Probabilistic classifier based on Bayes' theorem
    #    Good for: Text classification, fast training, works well with small datasets
    "Naive Bayes": MultinomialNB(),
    
    # 3. Support Vector Machine (SVM): Finds optimal decision boundary
    #    Good for: High-dimensional data, effective with clear margins
    "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42),
    
    # 4. Random Forest: Ensemble of decision trees (votes from multiple trees)
    #    Good for: Handling non-linear relationships, resistant to overfitting
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    
    # 5. Gradient Boosting: Builds trees sequentially, each correcting previous errors
    #    Good for: High accuracy, handles complex patterns
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    
    # 6. Decision Tree: Creates a tree of if-then rules
    #    Good for: Interpretable, easy to visualize
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    
    # 7. K-Nearest Neighbors: Classifies based on similar examples
    #    Good for: Simple concept, no training phase
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

# Dictionary to store results for each model
results = {}

# Loop through each model and train it
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # TRAINING PHASE: Model learns patterns from training data
    # fit() is where the actual learning happens
    model.fit(X_train_features, Y_train)
    
    # PREDICTION PHASE: Model makes predictions on test data (unseen during training)
    Y_pred = model.predict(X_test_features)  # Predicted labels (0 or 1)
    
    # Get probability predictions if the model supports it
    # Some models output probabilities, useful for ROC curves
    Y_pred_proba = (
        model.predict_proba(X_test_features)[:, 1]  # Probability of spam (class 1)
        if hasattr(model, "predict_proba")
        else Y_pred
    )

    # ========================================================================
    # CALCULATE CONFUSION MATRIX COMPONENTS
    # ========================================================================
    # Confusion matrix shows four types of predictions:
    # - True Negative (TN): Correctly predicted ham as ham ✓
    # - False Positive (FP): Incorrectly predicted ham as spam ✗ (Type I Error)
    # - False Negative (FN): Incorrectly predicted spam as ham ✗ (Type II Error)
    # - True Positive (TP): Correctly predicted spam as spam ✓
    # ========================================================================
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()

    # ========================================================================
    # CALCULATE ALL PERFORMANCE METRICS
    # ========================================================================
    
    # ACCURACY: Overall correctness (correct predictions / total predictions)
    # Range: 0 to 1 (higher is better)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # PRECISION: Of all emails predicted as spam, how many were actually spam?
    # Formula: TP / (TP + FP)
    # High precision = Few false alarms (legitimate emails marked as spam)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    
    # RECALL (Sensitivity): Of all actual spam, how many did we catch?
    # Formula: TP / (TP + FN)
    # High recall = Catches most spam (few spam emails slip through)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    
    # F1-SCORE: Harmonic mean of precision and recall (balanced metric)
    # Formula: 2 * (Precision * Recall) / (Precision + Recall)
    # Use F1 when you need balance between precision and recall
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    
    # SPECIFICITY: Of all actual ham, how many did we correctly identify?
    # Formula: TN / (TN + FP)
    # High specificity = Good at identifying legitimate emails
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # FALSE POSITIVE RATE (Type I Error): Ham incorrectly marked as spam
    # Formula: FP / (FP + TN)
    # Lower is better - users hate legitimate emails going to spam!
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # FALSE NEGATIVE RATE (Type II Error): Spam that slips through
    # Formula: FN / (FN + TP)
    # Lower is better - but usually less critical than FPR
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Store all results for this model in the results dictionary
    results[model_name] = {
        "model": model,  # The trained model object
        "predictions": Y_pred,  # Predicted labels
        "predictions_proba": Y_pred_proba,  # Prediction probabilities
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": confusion_matrix(Y_test, Y_pred),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    # Display key metrics for this model
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")


# ============================================================================
# 4. CREATE COMPREHENSIVE VISUALIZATIONS
# ============================================================================
# Visualization helps us understand and compare model performance at a glance.
# We'll create 5 different visualizations to analyze the results.
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS...")
print("=" * 80)

# Create a DataFrame (table) for easy comparison of all metrics
# This organizes our results into a clean, readable format
metrics_df = pd.DataFrame(
    {
        "Model": list(results.keys()),
        "Accuracy": [results[m]["accuracy"] for m in results],
        "Precision": [results[m]["precision"] for m in results],
        "Recall": [results[m]["recall"] for m in results],
        "F1-Score": [results[m]["f1_score"] for m in results],
        "Specificity": [results[m]["specificity"] for m in results],
        "FPR": [results[m]["fpr"] for m in results],
        "FNR": [results[m]["fnr"] for m in results],
    }
)

# Print the comparison table to console
print("\n" + "=" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 80)
print(metrics_df.to_string(index=False))

# Find the best model based on F1-Score (balanced metric)
# F1-Score is often the best single metric for imbalanced datasets
metrics_df_sorted = metrics_df.sort_values("F1-Score", ascending=False)
best_model_name = metrics_df_sorted.iloc[0]["Model"]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score: {metrics_df_sorted.iloc[0]['F1-Score']:.4f}")


# ============================================================================
# VISUALIZATION 1: Metrics Comparison Bar Chart (4 subplots)
# ============================================================================
# This creates a 2x2 grid of plots showing different aspects of performance
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Create 2x2 grid
fig.suptitle("Model Performance Metrics Comparison", fontsize=16, fontweight="bold")

# --- SUBPLOT 1: Main Performance Metrics ---
# Shows the four most important metrics side-by-side for each model
ax1 = axes[0, 0]
metrics_subset = metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]]
# melt() transforms wide format to long format for grouped bar chart
metrics_melted = metrics_subset.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", ax=ax1)
ax1.set_title("Main Performance Metrics", fontsize=12, fontweight="bold")
ax1.set_xlabel("Model", fontsize=10)
ax1.set_ylabel("Score", fontsize=10)
ax1.set_ylim([0, 1.05])  # Set y-axis from 0 to 1.05
ax1.legend(loc="lower right")
ax1.tick_params(axis="x", rotation=45)
# Add value labels on top of each bar
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.3f", fontsize=7)

# --- SUBPLOT 2: Specificity and Error Rates ---
# Shows how well models avoid errors
ax2 = axes[0, 1]
error_metrics = metrics_df[["Model", "Specificity", "FPR", "FNR"]]
error_melted = error_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=error_melted, x="Model", y="Score", hue="Metric", ax=ax2)
ax2.set_title("Specificity and Error Rates", fontsize=12, fontweight="bold")
ax2.set_xlabel("Model", fontsize=10)
ax2.set_ylabel("Score", fontsize=10)
ax2.set_ylim([0, 1.05])
ax2.legend(loc="upper right")
ax2.tick_params(axis="x", rotation=45)
for container in ax2.containers:
    ax2.bar_label(container, fmt="%.3f", fontsize=7)

# --- SUBPLOT 3: F1-Score Ranking ---
# Horizontal bar chart showing models ranked by F1-Score
# Colors indicate performance: green (excellent), yellow (good), red (needs improvement)
ax3 = axes[1, 0]
metrics_sorted = metrics_df.sort_values("F1-Score", ascending=True)  # Sort for ranking
# Color-code bars based on performance thresholds
colors = ["#ff6b6b" if x < 0.9 else "#51cf66" if x > 0.95 else "#ffd43b" for x in metrics_sorted["F1-Score"]]
ax3.barh(metrics_sorted["Model"], metrics_sorted["F1-Score"], color=colors)
ax3.set_title("F1-Score Ranking (Overall Performance)", fontsize=12, fontweight="bold")
ax3.set_xlabel("F1-Score", fontsize=10)
ax3.set_xlim([0, 1.05])
# Add value labels at the end of each bar
for i, v in enumerate(metrics_sorted["F1-Score"]):
    ax3.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)

# --- SUBPLOT 4: Accuracy vs Recall Scatter Plot ---
# Shows relationship between accuracy and recall
# Bubble size represents F1-Score (bigger = better overall performance)
ax4 = axes[1, 1]
scatter = ax4.scatter(
    metrics_df["Accuracy"],
    metrics_df["Recall"],
    s=metrics_df["F1-Score"] * 500,  # Size proportional to F1-Score
    c=range(len(metrics_df)),  # Color by index
    cmap="viridis",  # Color scheme
    alpha=0.6,  # Transparency
    edgecolors="black",
)
# Add model names as labels
for i, model in enumerate(metrics_df["Model"]):
    ax4.annotate(
        model,
        (metrics_df.iloc[i]["Accuracy"], metrics_df.iloc[i]["Recall"]),
        fontsize=8,
        ha="center",
    )
ax4.set_title("Accuracy vs Recall (bubble size = F1-Score)", fontsize=12, fontweight="bold")
ax4.set_xlabel("Accuracy", fontsize=10)
ax4.set_ylabel("Recall", fontsize=10)
ax4.set_xlim([0.85, 1.02])
ax4.set_ylim([0.85, 1.02])
ax4.grid(True, alpha=0.3)

plt.tight_layout()  # Adjust spacing between subplots
plt.savefig("model_comparison_metrics.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: model_comparison_metrics.png")


# ============================================================================
# VISUALIZATION 2: Confusion Matrices for All Models
# ============================================================================
# A confusion matrix shows the four types of predictions in a 2x2 grid:
#           Predicted Ham | Predicted Spam
# Actual Ham:     TN      |      FP
# Actual Spam:    FN      |      TP
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows, 4 columns
fig.suptitle("Confusion Matrices for All Models", fontsize=16, fontweight="bold")
axes = axes.ravel()  # Flatten the 2D array to 1D for easy iteration

# Create a confusion matrix heatmap for each model
for idx, (model_name, result) in enumerate(results.items()):
    cm = result["confusion_matrix"]
    # Create heatmap with annotations (numbers in each cell)
    sns.heatmap(
        cm,
        annot=True,  # Show numbers in cells
        fmt="d",  # Format as integers
        cmap="Blues",  # Color scheme (light to dark blue)
        ax=axes[idx],
        cbar=False,  # No color bar needed
        square=True,  # Make cells square-shaped
        xticklabels=["Ham", "Spam"],  # X-axis labels
        yticklabels=["Ham", "Spam"],  # Y-axis labels
    )
    # Title shows model name and accuracy
    axes[idx].set_title(f"{model_name}\nAcc: {result['accuracy']:.3f}", fontsize=10)
    axes[idx].set_ylabel("Actual")
    axes[idx].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")
print("✓ Saved: confusion_matrices.png")


# ============================================================================
# VISUALIZATION 3: ROC Curves (Receiver Operating Characteristic)
# ============================================================================
# ROC curve shows the trade-off between True Positive Rate and False Positive Rate
# AUC (Area Under Curve) measures overall model performance (1.0 = perfect)
# A model with AUC = 0.5 is no better than random guessing
# ============================================================================

plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(results)))  # Generate distinct colors

# Plot ROC curve for each model that supports probability predictions
for idx, (model_name, result) in enumerate(results.items()):
    if hasattr(result["model"], "predict_proba"):
        # Calculate ROC curve points
        fpr, tpr, _ = roc_curve(Y_test, result["predictions_proba"])
        # Calculate AUC score
        auc_score = roc_auc_score(Y_test, result["predictions_proba"])
        # Plot the curve
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})", color=colors[idx], linewidth=2)

# Plot diagonal line (represents random classifier)
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300, bbox_inches="tight")
print("✓ Saved: roc_curves.png")


# ============================================================================
# VISUALIZATION 4: Detailed Metrics Heatmap
# ============================================================================
# Heatmap provides a color-coded view of all metrics across all models
# Green = good performance, Red = poor performance
# ============================================================================

plt.figure(figsize=(12, 8))
metrics_for_heatmap = metrics_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "FPR", "FNR"]
]
sns.heatmap(
    metrics_for_heatmap,
    annot=True,  # Show values in cells
    fmt=".4f",  # Format with 4 decimal places
    cmap="RdYlGn",  # Red-Yellow-Green color scheme
    center=0.5,  # Center color scale at 0.5
    linewidths=0.5,  # Add gridlines
    cbar_kws={"label": "Score"},  # Color bar label
)
plt.title("Comprehensive Metrics Heatmap", fontsize=14, fontweight="bold")
plt.xlabel("Metrics", fontsize=12)
plt.ylabel("Models", fontsize=12)
plt.tight_layout()
plt.savefig("metrics_heatmap.png", dpi=300, bbox_inches="tight")
print("✓ Saved: metrics_heatmap.png")


# ============================================================================
# VISUALIZATION 5: Error Analysis (Type I and Type II Errors)
# ============================================================================
# Type I Error (False Positive): Legitimate email marked as spam
#   - Annoying for users, might miss important emails
# Type II Error (False Negative): Spam email not caught
#   - Less critical, users can delete these manually
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Error Analysis Across Models", fontsize=16, fontweight="bold")

# --- Left Plot: False Positive Rate (Type I Error) ---
ax1 = axes[0]
fpr_sorted = metrics_df.sort_values("FPR", ascending=True)
# Color-code bars: green (< 1%), yellow (1-5%), red (> 5%)
colors_fpr = ["#51cf66" if x < 0.01 else "#ffd43b" if x < 0.05 else "#ff6b6b" for x in fpr_sorted["FPR"]]
ax1.barh(fpr_sorted["Model"], fpr_sorted["FPR"], color=colors_fpr)
ax1.set_title("False Positive Rate (Type I Error)\nLegitimate emails marked as spam", fontsize=12, fontweight="bold")
ax1.set_xlabel("False Positive Rate", fontsize=10)
for i, v in enumerate(fpr_sorted["FPR"]):
    ax1.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

# --- Right Plot: False Negative Rate (Type II Error) ---
ax2 = axes[1]
fnr_sorted = metrics_df.sort_values("FNR", ascending=True)
colors_fnr = ["#51cf66" if x < 0.01 else "#ffd43b" if x < 0.05 else "#ff6b6b" for x in fnr_sorted["FNR"]]
ax2.barh(fnr_sorted["Model"], fnr_sorted["FNR"], color=colors_fnr)
ax2.set_title("False Negative Rate (Type II Error)\nSpam emails that slip through", fontsize=12, fontweight="bold")
ax2.set_xlabel("False Negative Rate", fontsize=10)
for i, v in enumerate(fnr_sorted["FNR"]):
    ax2.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("error_analysis.png", dpi=300, bbox_inches="tight")
print("✓ Saved: error_analysis.png")


# ============================================================================
# 5. DETAILED CLASSIFICATION REPORTS
# ============================================================================
# Classification reports provide detailed metrics for each class (ham/spam)
# including precision, recall, f1-score, and support (number of samples)
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 80)

for model_name, result in results.items():
    print(f"\n{model_name}:")
    print("-" * 60)
    # classification_report generates a detailed text report
    print(classification_report(Y_test, result["predictions"], target_names=["Ham", "Spam"]))
    
    # Explain confusion matrix components in plain English
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {result['tn']:4d} - Correctly identified as Ham")
    print(f"  False Positives (FP): {result['fp']:4d} - Ham incorrectly marked as Spam")
    print(f"  False Negatives (FN): {result['fn']:4d} - Spam incorrectly marked as Ham")
    print(f"  True Positives (TP):  {result['tp']:4d} - Correctly identified as Spam")


# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================
# This section identifies which model performs best for each specific metric
# Different applications might prioritize different metrics
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nBest performing models by metric:")
# idxmax() finds the row with maximum value for each metric
print(f"  🎯 Highest Accuracy:    {metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']} ({metrics_df['Accuracy'].max():.4f})")
print(f"  🎯 Highest Precision:   {metrics_df.loc[metrics_df['Precision'].idxmax(), 'Model']} ({metrics_df['Precision'].max():.4f})")
print(f"  🎯 Highest Recall:      {metrics_df.loc[metrics_df['Recall'].idxmax(), 'Model']} ({metrics_df['Recall'].max():.4f})")
print(f"  🎯 Highest F1-Score:    {metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']} ({metrics_df['F1-Score'].max():.4f})")
print(f"  🎯 Highest Specificity: {metrics_df.loc[metrics_df['Specificity'].idxmax(), 'Model']} ({metrics_df['Specificity'].max():.4f})")
# idxmin() finds the row with minimum value (lower is better for error rates)
print(f"  🎯 Lowest FPR:          {metrics_df.loc[metrics_df['FPR'].idxmin(), 'Model']} ({metrics_df['FPR'].min():.4f})")
print(f"  🎯 Lowest FNR:          {metrics_df.loc[metrics_df['FNR'].idxmin(), 'Model']} ({metrics_df['FNR'].min():.4f})")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated visualizations:")
print("  1. model_comparison_metrics.png - Comprehensive metrics comparison")
print("  2. confusion_matrices.png - All confusion matrices")
print("  3. roc_curves.png - ROC curves for all models")
print("  4. metrics_heatmap.png - Detailed metrics heatmap")
print("  5. error_analysis.png - Type I and Type II error analysis")
print("\n" + "=" * 80)

# ============================================================================
# KEY TAKEAWAYS FOR YOUR FIRST ML PROJECT:
# ============================================================================
# 1. Data Preparation: Clean data is crucial - handle missing values and format
# 2. Feature Engineering: Convert text to numbers using TF-IDF
# 3. Train-Test Split: Always evaluate on unseen data to test real performance
# 4. Multiple Models: Try different algorithms - no single model is always best
# 5. Metrics Matter: Choose metrics based on your problem (F1 for balance)
# 6. Visualization: Plots help understand and communicate results
# 7. Confusion Matrix: Understand the types of errors your model makes
# 8. Trade-offs: High precision might mean lower recall and vice versa
# ============================================================================