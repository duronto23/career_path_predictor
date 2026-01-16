# %% [markdown]
# # Logistic Regression

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# %%
print(X_train)

# %%
print(y_train)

# %%
print(X_test)

# %%
print(y_test)

# %%
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()

X_train = ord_enc.fit_transform(X_train)

X_test = ord_enc.fit_transform(X_test)

# %%
print(X_train)

# %%
print(X_test)

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
print(X_train)

# %%
print(X_test)

# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# %%
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
print(cm)
accuracy_score(y_test, y_pred)

# %%
displ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
_, ax = plt.subplots(figsize=(10,10))
displ.plot(cmap="Greens", ax=ax)
plt.show()
displ = ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm), display_labels=classifier.classes_)
_, ax = plt.subplots(figsize=(10,10))
displ.plot(cmap="Blues", values_format='.2%', ax=ax)
plt.show()


