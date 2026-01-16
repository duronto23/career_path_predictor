# %% [markdown]
# # Artificial Neural Network

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# %%
print(X)

# %%
print(y)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ctX = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44])], remainder='passthrough')
X = ctX.fit_transform(X)
X = X.toarray()

# %%
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
y = encoder.fit_transform(y)

# %%
print(X)

# %%
print(y)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
import tensorflow.keras as keras
ann = keras.models.Sequential()

# %%
ann.add(keras.layers.Dense(units=6, activation='relu'))

# %%
ann.add(keras.layers.Dense(units=6, activation='relu'))

# %%
ann.add(keras.layers.Dense(units=3, activation='sigmoid'))

# %%
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# %%
ann.fit(X_train, y_train, batch_size = 32, epochs = 500)

# %%
y_pred = ann.predict(X_test)
y_pred = np.around(y_pred)
y_pred = y_pred.reshape(len(y_pred),3)
y_test = y_test.reshape(len(y_test),3)

y_pred = encoder.inverse_transform(y_pred)
y_test = encoder.inverse_transform(y_test)

print(np.concatenate((y_pred, y_test),1))

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
y_pred[y_pred == None] = "None"
cm = confusion_matrix(y_test, y_pred, labels=["Business Related Profession", "Doctor", "Engineer", "None"])
print(cm)
accuracy_score(y_test, y_pred)

# %%
displ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Business Related Profession", "Doctor", "Engineer", "None"])
_, ax = plt.subplots(figsize=(10,10))
displ.plot(cmap="Greens", ax=ax)
plt.show()
displ = ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm), display_labels=["Business Related Profession", "Doctor", "Engineer", "None"])
_, ax = plt.subplots(figsize=(10,10))
displ.plot(cmap="Blues", values_format='.2%', ax=ax)
plt.show()


