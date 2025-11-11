import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


df1 = pd.read_csv("spam_train1.csv")
df2 = pd.read_csv("spam_train2.csv")
df = pd.concat([df1, df2])

data = df.where((pd.notnull(df)), '')
# print(data.info())
# print(data.shape)



data.loc[data['label'] == 'spam', 'label'] = 1
data.loc[data['label'] == 'ham', 'label'] = 0

X = data['text']
Y = data['label']

print (X)
print (Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X_train.shape)
print(X_test.shape)


print(Y_train.shape)
print(Y_test.shape)