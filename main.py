import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


df1 = pd.read_csv("spam_train1.csv")
df2 = pd.read_csv("spam_train2.csv")
df = pd.concat([df1, df2])
print(df)
