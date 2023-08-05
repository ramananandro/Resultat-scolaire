#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


data=pd.read_csv('data_primaire.csv', sep=';')


# In[6]:


data.head()


# # Nettoyage

# In[25]:


# data.info()


# In[9]:


col_str = ['milieu', 'condition_scolaire']
col_int = ['code','effectif']
col_data = data.columns


# In[11]:


for col in col_data:
    if col not in col_int:
        data[col] = data[col].astype(str)


# ## Remplacer virgule par point

# In[22]:


for col in col_data:
    if col not in col_int:
        data[col] = data[col].str.replace(',','.')


# In[23]:


for col in col_data:
    if col not in col_int and col not in col_str:
        print(col)
        data[col] = data[col].astype(float)


# In[26]:


data['condition_scolaire_binaire']  = np.where(data['condition_scolaire'] == 'DEFAVORABLE', 0,1) 


# In[27]:


data.head()


# In[28]:


col_float = []
for col in col_data:
    if col not in col_int and col not in col_str:
        col_float.append(col)
print(col_float)


# In[29]:


col_float.append('condition_scolaire_binaire')


# In[30]:


X = data[col_float]


# In[31]:


X.head()


# In[35]:


X.corr()['condition_scolaire_binaire']


# In[42]:


plt.scatter(X['eleve_par_chaise'],X['eleve_par_ens'],c=X['condition_scolaire_binaire'])
legend = plt.legend(title='Condition', loc='upper left')
plt.show()


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[46]:


y = X['condition_scolaire_binaire']
X.pop('condition_scolaire_binaire')
X1 = X[['eleve_par_chaise','eleve_par_ens']]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)


# In[47]:


# Create a SVM classifier with the OvO strategy
svm_classifier = SVC(kernel='linear', decision_function_shape='ovo')

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)


# In[50]:


# Plot the decision boundaries
# Create a mesh to plot the decision boundaries
h = .02  # Step size in the mesh
x_min, x_max = X1['eleve_par_chaise'].min() - 1, X1['eleve_par_chaise'].max() + 1
y_min, y_max = X1['eleve_par_ens'].min() - 1, X1['eleve_par_ens'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Make predictions on the mesh grid
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot the training points
plt.scatter(X_train['eleve_par_chaise'], X_train['eleve_par_ens'], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('eleve_par_chaise')
plt.ylabel('eleve_par_ens')
plt.title('SVM Decision Boundaries (OvO)')
plt.show()


# In[51]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[52]:


# Test the classifier on the test data
y_pred = svm_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", conf_matrix)


# In[53]:


X_pred = [[13.,15.5],[23.8,50.7],[1.5,6.8]]
pred_ = svm_classifier.predict(X_pred)


# In[54]:


print(pred_)


# ## Decision Tree: arbre de décision

# In[55]:


from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree classifier with default parameters
tree_classifier = DecisionTreeClassifier()

# Train the classifier on the training data
tree_classifier.fit(X_train, y_train)

# Test the classifier on the test data
y_pred = tree_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", conf_matrix)


# ## Random Forest

# In[56]:


from sklearn.ensemble import RandomForestClassifier


# In[57]:


# Create a Random Forest classifier with default parameters
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Test the classifier on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", conf_matrix)


# # Regression

# In[58]:


col_new = []
for col in col_data:
    if col != 'code' and col != 'condition_scolaire' and col != 'condition_scolaire_binaire':
        col_new.append(col)
X_new = data[col_new]


# In[59]:


X_new.info()


# In[60]:


X_new.corr()['tx_promotion']


# In[61]:


# Perform one-hot encoding on the 'X2' column
one_hot_encoded = pd.get_dummies(X_new['milieu'], prefix='Category')

# Add the one-hot encoded columns to the original DataFrame
X_new = pd.concat([X_new, one_hot_encoded], axis=1)
X_new.head()


# In[62]:


X_new.pop('milieu')
X_new.info()


# In[63]:


X_new.corr()['tx_promotion']


# In[64]:


X_new.corr()


# In[69]:


plt.scatter(X_new['eleve_par_ens'],X_new['tx_promotion'])
plt.show()


# In[70]:


variable = ['eleve_par_salle', 'eleve_par_chaise', 'eleve_par_ens','kit_par_eleve', 'Category_URB H']
X_reg = X_new[variable]
y = X_new['tx_promotion']


# In[71]:


from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[72]:


# Make predictions on the test data
predictions = model.predict(X_test)


# In[73]:


from sklearn.metrics import mean_squared_error
# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)
# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)


# In[76]:


predictions = model.predict([[60.,2.,45.,2.,0],[30.,2.,20.,2.,1]])
print(predictions)


# In[ ]:




