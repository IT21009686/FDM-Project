#!/usr/bin/env python
# coding: utf-8

# # **Libraries and Utilities**

# In[1]:


import warnings
warnings.filterwarnings('ignore')

# basic libraries
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

#visulaization modules
import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

#Common model helpers
from sklearn.preprocessing import (StandardScaler,
                                   LabelEncoder,
                                   OneHotEncoder)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score,
                             auc,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_curve,
                             roc_auc_score,
                             confusion_matrix,
                             classification_report)

from sklearn.model_selection import (GridSearchCV,
                                     StratifiedKFold,
                                     cross_val_score)

# imbalance dataset handling

from imblearn.over_sampling import (SMOTE)

# model algorithams
from sklearn.ensemble import (RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# ## Importing the dataset

# In[2]:


# Specify the file path
file_path = "D:/2023/Y3S2/FDM/Project/Ours/healthcare-dataset-stroke-data.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)


# # **Data Exploration and Understanding**

# In[3]:


# Explore the data types of the columns
df.dtypes


# In[4]:


df.describe()


# In[5]:


df.head()


# In[6]:


#To gets the columns that are most correlated with the class column
cor = df.corr(numeric_only = True)['stroke'].sort_values
cor


# The 'age' feature appears to have a positive correlation with 'status', suggesting that with increasing age there is a liklihood of stroke.
# 
# The 'Bmi' feature shows the least corelation out of all the features, suggesting it may not be a strong predictor

# In[7]:


#Perform a chisquared test of indipendence for each categorical coloumn, to determine if there is a significant assosiation
categorical_col = df.select_dtypes(include=['object']).columns
  #object data type represents string data

#creating a dictionary to store chi squared stats and p-values
chi2_results = {}

#perfrom chi-squared for each categorical col
for col in categorical_col:
  contingency_table = pd.crosstab(df[col], df['stroke'])
  chi2, p, _, _ = chi2_contingency(contingency_table)
  chi2_results[col] = {'Chi-Square': chi2, 'p-value': p}

# Create a DataFrame from the results
chi2_results_df = pd.DataFrame(chi2_results).T

# Sort by p-value (smaller p-values indicate stronger association)
chi2_results_df.sort_values(by='p-value', ascending=True, inplace=True)

# Print the results
print(chi2_results_df)


# features such as 'ever_married', 'work_type', 'smoking_status' have very low p-values close to 0 indicating a strong assosiation with target variable ('stroke')
# 
# 
# And other such as 'residence_type', 'gender' have a very high p-value suggests that no significant assosiation to stroke
# 

# # **Data Preprocessing**

# ## Data Cleaning

# ### 1 -  *Remove unnecessary coloumns*

# In[8]:


df.head()


# In[9]:


df.drop(['id'],axis=1,inplace = True)


# In[10]:


df.info()


# ### 2 - *Null handling*

# checking for null values and dropping null values

# In[11]:


#check the null value count in the filtered dataset
df.isnull().sum()


# In[12]:


df.dropna(inplace=True)
df.isnull().sum()


# ### 3 - *Checking for duplicate values*

# In[13]:


duplicate_values = df.drop_duplicates(inplace=True)
 #'inplace = true' means to change within original dataset
print(duplicate_values)

#No duplicate valuess


# In[14]:


df.info()


# ### 4 -  *Handling Outliers*

# In[15]:


# List of numerical columns to be analyzed
num_cols = ['age', 'bmi', 'avg_glucose_level']

# Set up subplots for box plots of numerical columns
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)

    # Create box plots for each numerical column
    sns.boxplot(x=df[num_cols[i]])
    plt.title(num_cols[i])  # Set subplot title
plt.show()


# In[16]:


# Function to detect outliers using IQR method
def detect_outliers(data, column):
    q1 = df[column].quantile(.25)  # First quartile (25th percentile)
    q3 = df[column].quantile(.75)  # Third quartile (75th percentile)
    IQR = q3 - q1  # Interquartile Range (IQR)

    # Calculate lower and upper bounds for outlier detection
    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + (1.5 * IQR)

    # Find indices of outliers based on lower and upper bounds
    outlier_indices = df.index[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return outlier_indices


# In[17]:


# List to store indices of outliers
index_list = []

# Iterate through numerical columns and find outliers using the detect_outliers function
for column in num_cols:
    index_list.extend(detect_outliers(df, column))

# Remove duplicates and sort the list of outlier indices
index_list = sorted(set(index_list))

# Get shape of the DataFrame before and after removing outliers
before_remove = df.shape
df = df.drop(index_list)  # Drop rows with outlier indices
after_remove = df.shape

# Print the shape of the data before and after removing outliers
print(f'Shape of data before removing outliers: {before_remove}')
print(f'Shape of data after removing outliers: {after_remove}')


# This indicates that the original dataset had 4909 rows and 11 CoL before removing outliers. And after the rows reduced to 4260.

# In[18]:


# List of numerical columns to be analyzed
num_cols = ['age', 'bmi', 'avg_glucose_level']

# Create box plots for numerical columns after removing outliers
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)

    # Create box plots for each numerical column
    sns.boxplot(x=df[num_cols[i]], color='#6DA59D')
    plt.title(num_cols[i])  # Set subplot title
plt.show()


# ## Feature Engineering

# ### 1. Rounding up age to the nearest whole number

# In[19]:


#round up the age coloumn to the nearest whole number
df['age'] = np.ceil(df['age'])


# In[20]:


df['age'].describe()


# In[21]:


(df['age'] <= 1).sum()


# since there are 46 coloumns with age less than 1, we have decided to keep these values as they could represent children.
# Proven with studies that children as young as newborns are also suseptible to having a stroke

# ### 2. Removing 'Other from Gender'

# **Removing unwanted coloumns**
# - Gender

# In[22]:


#getting the gender and their counts
df['gender'].value_counts()


# In[23]:


#removing the gender 'other' using boolean masking
df=df[(df['gender'] != 'Other')]

#getting the gender and their counts
df['gender'].value_counts()


# In[ ]:





# ## Handling Class Imbalance (using upSampling)

# In[24]:


print(df.columns)


# In[25]:


#split the df into two subsets based on the 'stroke' column
df_0 = df[df.iloc[:,-1] == 0]  #subset containing rows where 'stroke' is 0 (no stroke)
df_1 = df[df.iloc[:,-1] == 1]  #subset containing rows where 'stroke' is 1 (stroke)

#counts of 'stroke' values in the original DataFrame
stroke_counts = df['stroke'].value_counts()
print(stroke_counts)


# upSampling duplicates the minority class rather than genrating diverse classes

# In[26]:


from sklearn.utils import resample

#upsample the minority class (stroke = 1) to match the number of majority class samples (stroke = 0)
df_1 = resample(df_1,replace=True , n_samples=df_0.shape[0] ,
                random_state=123 )


# In[27]:


df.info()


# In[28]:


#concatenate upsampled data
df = np.concatenate((df_0, df_1))

#create the balanced dataframe
df = pd.DataFrame(df, columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke'])

# visualize balanced data
stroke = dict(df['stroke'].value_counts())
fig = px.pie(names = ['False','True'],values = stroke.values(),
             title = 'Stroke Occurance',
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_traces(textposition='inside', textinfo='percent+label')


# ## Instead of leaving 'Unknown' we want to predict as one of three categories

# In[29]:


#check for 'unknown' smoking
unknown_smoking_count = df['smoking_status'].value_counts().get('Unknown', 0)

# Print the number of unknown values
print(f"Number of 'Unknown' values in 'smoking_status': {unknown_smoking_count}")


# In[30]:


df_encode = pd.get_dummies(data =df ,
                              columns =  ['gender','ever_married','work_type',
                                          'Residence_type'] ,
                              drop_first=True )


# In[31]:


#split data into 2 subsets: one with smoking status values and one with unknown values
known_smoking_data = df_encode[df_encode['smoking_status'] != 'Unknown']
unknown_smoking_data = df_encode[df_encode['smoking_status'] == 'Unknown']


# In[32]:


#seperate features(x_known) and target(y_known) from known smoking data
X_known = known_smoking_data.drop(['smoking_status'], axis=1)
y_known = known_smoking_data['smoking_status']


# In[33]:


#Separate features (X_unknown) from the unknown smoking data
X_unknown = unknown_smoking_data.drop(['smoking_status'], axis=1)


# In[34]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=17)
model.fit(X_known, y_known)


# In[35]:


#use trained model to predict 'smoking_status' for rows where it was 'unknown'
unknown_smoking_predictions = model.predict(X_unknown)


# In[36]:


df.loc[df['smoking_status'] == 'Unknown', 'smoking_status'] = unknown_smoking_predictions
  #this replaces the unknown values with the predictd values


# In[37]:


#check again to confirm for 'unknown' smoking
unknown_smoking_count = df['smoking_status'].value_counts().get('Unknown', 0)

# Print the number of unknown values
print(f"Number of 'Unknown' values in 'smoking_status': {unknown_smoking_count}")


# ## Encoding the data using Dummy Variable encoding.
# We could use techniques such as one-hot encoding as well but in this case it is better to reduce the dimensionality by reducing the number of variable.
# 
# For a categorical feature with N categories, N-1 binary columns are created, and the Nth category is represented implicitly when all other binary columns are 0. Setting drop_first=True in the pd.get_dummies() function achieves this behavior, which is characteristic of dummy variable encoding.[link text]

# We do not need to manually encode as python automatically asigns values as '1' and '0'.
# On with encoding

# In[38]:


#creating a new variable to hold the dummy variables so we can preserve the old data
df_resampled = pd.get_dummies(data =df ,
                              columns =  ['gender','ever_married','work_type',
                                          'Residence_type','smoking_status'] ,
                              drop_first=True )
    #Dropping the first column reduces the multicollinearity in the dataset


# In[39]:


df_resampled.head()


# save to csv

# In[40]:


df.to_csv("cleaned_dataset.csv", index = False)


# ## Split the Data
# 
# 

# Split - Target and validation

# In[41]:


#defining features x and target y

x = df_resampled.drop('stroke', axis = 1)
  # contains all the feature cols except for the stroke col which is target variable
y = pd.to_numeric( df_resampled['stroke'])
  #represents target variable 'stroke'


# In[42]:


#Data scaling
  #standardize the range of independent variables or features in the data
scaler = StandardScaler()

x = scaler.fit_transform(x)


# Split - Train and Test

# In[43]:


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size = .20,
                                                    random_state=42)
  #test_size means 20% if data used for testing and 80% used for training
  #random_state ensures that data will be split in the same way, ensures consistency


# # **Building Machine Learning Models**

# We will be building 6 models
# - Logistic Regression
# - Decison Tree
# - KNN
# - SVM
# - Naive Bayes
# - Random Forest (Is an ensemble technique)
# 
# 
# Model is evaluated using
# 1.   Accuracy
# 2.   ROC Curve
#       -  visualize the true positive rate against the false positive rate. AUC (Area Under the Curve) is a metric to evaluate the model's ability to distinguish between positive and negative classes
# 3.   Confusion Matrix
#       -  heatmap of the confusion matrix is plotted to visualize the true positive, true negative, false positive, and false negative predictions. It provides a detailed breakdown of the model's performance for each class.
# 4.   Classification report
# 
# 
# 
# 
# 

# ## **Model 1 - Logistic Regression**

# In[44]:


# Define the parameter grid for the grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Create a Logistic Regression classifier
logistic_classifier = LogisticRegression()

# Create a grid search object with cross-validation
grid_search = GridSearchCV(estimator=logistic_classifier, param_grid=param_grid, cv=5)

# Fit the grid search to the training data
grid_search.fit(x_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_logistic_classifier = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred_Lg = best_logistic_classifier.predict(x_test)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred_Lg)
print("Best Parameters:", best_params)
print("Accuracy on Test Set:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred_Lg))

#ROC curve
roc_auc = roc_auc_score(y_test, best_logistic_classifier.predict_proba(x_test)[:, 1])
fpr, tpr, _ = roc_curve(y_test, best_logistic_classifier.predict_proba(x_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
conf_matrix_Lg = confusion_matrix(y_test, y_pred_Lg)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_Lg, annot=True, fmt="d", cmap='Blues', linewidths=2.5, cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.xticks([0.5, 1.5], ['Predicted Non-Stroke', 'Predicted Stroke'])
plt.yticks([0.5, 1.5], ['Actual Non-Stroke', 'Actual Stroke'])
plt.show()


# ## **Model 2 - Decision Tree**

# In[45]:


# Define the parameter grid for the grid search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Create a grid search object with cross-validation
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5)

# Fit the grid search to the training data
grid_search.fit(x_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_dt_classifier = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred_dt = best_dt_classifier.predict(x_test)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred_dt)
print("Best Parameters:", best_params)
print("Accuracy on Test Set:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# ROC curve
roc_auc = roc_auc_score(y_test, best_dt_classifier.predict_proba(x_test)[:, 1])
fpr, tpr, _ = roc_curve(y_test, best_dt_classifier.predict_proba(x_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap='Blues', linewidths=2.5, cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.xticks([0.5, 1.5], ['Predicted Non-Stroke', 'Predicted Stroke'])
plt.yticks([0.5, 1.5], ['Actual Non-Stroke', 'Actual Stroke'])
plt.show()


# In[46]:


df_resampled.head()


# ## **Model 3 - K-Nearest Neighbour**

# In[47]:


# Define the parameter grid for the grid search
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

# Create a K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()

# Create a grid search object with cross-validation
grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=5)

# Fit the grid search to the training data
grid_search.fit(x_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_knn_classifier = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred_knn = best_knn_classifier.predict(x_test)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred_knn)
print("Best Parameters:", best_params)
print("Accuracy on Test Set:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# ROC curve
roc_auc = roc_auc_score(y_test, best_knn_classifier.predict_proba(x_test)[:, 1])
fpr, tpr, _ = roc_curve(y_test, best_knn_classifier.predict_proba(x_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap='Blues', linewidths=2.5, cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.xticks([0.5, 1.5], ['Predicted Non-Stroke', 'Predicted Stroke'])
plt.yticks([0.5, 1.5], ['Actual Non-Stroke', 'Actual Stroke'])
plt.show()


# ## **Model 4 - Naive Bayes**

# In[48]:


# Define the parameter grid for the grid search
param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# Create a Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Create a grid search object with cross-validation
grid_search_nb = GridSearchCV(estimator=nb_classifier, param_grid=param_grid_nb, cv=5)

# Fit the grid search to the training data
grid_search_nb.fit(x_train, y_train)

# Get the best parameters and the best estimator
best_params_nb = grid_search_nb.best_params_
best_nb_classifier = grid_search_nb.best_estimator_

# Evaluate the model on the test set
y_pred_nb = best_nb_classifier.predict(x_test)

# Calculate accuracy and print the classification report
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Best Parameters:", best_params_nb)
print("Accuracy on Test Set:", accuracy_nb)
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

# ROC curve
roc_auc_nb = roc_auc_score(y_test, best_nb_classifier.predict_proba(x_test)[:, 1])
fpr_nb, tpr_nb, _ = roc_curve(y_test, best_nb_classifier.predict_proba(x_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_nb))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, fmt="d", cmap='Blues', linewidths=2.5, cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.xticks([0.5, 1.5], ['Predicted Non-Stroke', 'Predicted Stroke'])
plt.yticks([0.5, 1.5], ['Actual Non-Stroke', 'Actual Stroke'])
plt.show()


# ## **Model 5 - SVM (Support Vector Model)**

# In[49]:


# Define the parameter grid for the grid search
param_grid_svm = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf']  # Kernel type (linear or radial basis function)
}

# Create a Support Vector Machine classifier
svm_classifier = SVC(probability=True)  # probability=True enables probability estimates

# Create a grid search object with cross-validation
grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid_svm, cv=5)

# Fit the grid search to the training data
grid_search_svm.fit(x_train, y_train)

# Get the best parameters and the best estimator
best_params_svm = grid_search_svm.best_params_
best_svm_classifier = grid_search_svm.best_estimator_

# Evaluate the model on the test set
y_pred_svm = best_svm_classifier.predict(x_test)

# Calculate accuracy and print the classification report
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Best Parameters:", best_params_svm)
print("Accuracy on Test Set:", accuracy_svm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

# ROC curve
roc_auc_svm = roc_auc_score(y_test, best_svm_classifier.predict_proba(x_test)[:, 1])
fpr_svm, tpr_svm, _ = roc_curve(y_test, best_svm_classifier.predict_proba(x_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_svm))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap='Blues', linewidths=2.5, cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.xticks([0.5, 1.5], ['Predicted Non-Stroke', 'Predicted Stroke'])
plt.yticks([0.5, 1.5], ['Actual Non-Stroke', 'Actual Stroke'])
plt.show()


# ## **Ensemble Model 6 - Random Forest**
# Random Forest is a technique of Ensamble Learning that builds multiple decision trees on different samples and takes their majority vote for classification and average in case of regression.

# In[50]:


#Model Instantiation Without Tuning Any Parameters
from sklearn.ensemble import RandomForestClassifier

#Hyperparameter tuning (GridSearchCV)
#define grid of hyperparameters to search
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf_classifier = RandomForestClassifier(random_state=42)

#create a gridsearchCV object
grid_search_rf = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid_rf,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
)

# Perform grid search to find the best hyperparameters
grid_search_rf.fit(x_train, y_train)

# Get the best hyperparameters
best_params_rf = grid_search_rf.best_params_

#Model refitting with best hyperparameters
best_rf_classifier = RandomForestClassifier(random_state=42,
                                            **best_params_rf)
best_rf_classifier.fit(x_train, y_train)

y_pred_rf = best_rf_classifier.predict(x_test)

#Accuracy calculation
#calculate accuracy on training and testing sets with tuned hyperparameters
accuracy_rf_train = best_rf_classifier.score(x_train, y_train)
accuracy_rf_test = accuracy_score(y_test, y_pred_rf)

print("Best Hyperparameters for Random Forest:", best_params_rf)
print(f"Accuracy for Random Forest (Train): {accuracy_rf_train:.4f}")
print(f"Accuracy for Random Forest (Test): {accuracy_rf_test:.4f}")

# ROC curve
roc_auc_rf = roc_auc_score(y_test, best_rf_classifier.predict_proba(x_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf_classifier.predict_proba(x_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_rf))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
#calculate confusion matrix on the testing set
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap='Blues', linewidths=2.5, cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix - Random Forest')
plt.xticks([0.5, 1.5], ['Predicted Non-Stroke', 'Predicted Stroke'])
plt.yticks([0.5, 1.5], ['Actual Non-Stroke', 'Actual Stroke'])
plt.show()

#F1 Score Calculation
f1_score_value = f1_score(y_test,y_pred_rf)
print(f'F1 Score: {f1_score_value:.4f}')

# Classification Report
class_report_rf = classification_report(y_test, y_pred_rf)
print("Classification Report for Random Forest:\n", class_report_rf)


# In[51]:


rf_model = RandomForestClassifier(n_estimators=150,criterion='entropy',random_state = 123)
rf_model.fit(x_train,y_train)

RandomForestClassifier(criterion='entropy', n_estimators=150, random_state=123)
y_pred = rf_model.predict(x_test)
accuracy_score(y_test,y_pred)


# ## **F1-Score + Precision + Recall**

# In[52]:


from sklearn.metrics import f1_score

#using the predictions from the models we can find the F1 score

#LogisticRegression
f1_LG = f1_score(y_test, y_pred_Lg)
precision_LG = precision_score(y_test, y_pred_Lg, average=None)
recall_LG = recall_score(y_test, y_pred_Lg, average=None)

# Predictions from KNN model
f1_knn = f1_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average=None)
recall_knn = recall_score(y_test, y_pred_knn, average=None)

# Predictions from Naive Bayes model
f1_nb = f1_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average=None)
recall_nb = recall_score(y_test, y_pred_nb, average=None)

# Predictions from Decision Tree model
f1_dt = f1_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average=None)
recall_dt = recall_score(y_test, y_pred_dt, average=None)

# Predictions from Random Forest model
f1_rf = f1_score(y_test, y_pred_rf)

# Predictions from SVM
f1_svm = f1_score(y_test, y_pred_svm)

# Store the metrics in dictionaries for easy plotting
precision_scores = {
    'Logistic Regression': precision_LG,
    'KNN': precision_knn,
    'Naive Bayes': precision_nb,
    'Decision Tree': precision_dt
}

recall_scores = {
    'Logistic Regression': recall_LG,
    'KNN': recall_knn,
    'Naive Bayes': recall_nb,
    'Decision Tree': recall_dt
}

f1_scores = {
    'Logistic Regression': f1_LG,
    'KNN': f1_knn,
    'Naive Bayes': f1_nb,
    'Decision Tree': f1_dt
}

# Create a grouped bar chart for precision
plt.figure(figsize=(12, 6))
barWidth = 0.2
r1 = range(len(precision_scores['Logistic Regression']))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, precision_scores['Logistic Regression'], color='b', width=barWidth, edgecolor='grey', label='Logistic Regression')
plt.bar(r2, precision_scores['KNN'], color='c', width=barWidth, edgecolor='grey', label='KNN')
plt.bar(r3, precision_scores['Naive Bayes'], color='m', width=barWidth, edgecolor='grey', label='Naive Bayes')

plt.xlabel('Stroke Instances (0 and 1)', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(precision_scores['Logistic Regression']))], ['0', '1'])
plt.ylabel('Precision', fontweight='bold', fontsize=15)
plt.legend()
plt.show()

# Create a grouped bar chart for recall
plt.figure(figsize=(12, 6))
plt.bar(r1, recall_scores['Logistic Regression'], color='b', width=barWidth, edgecolor='grey', label='Logistic Regression')
plt.bar(r2, recall_scores['KNN'], color='c', width=barWidth, edgecolor='grey', label='KNN')
plt.bar(r3, recall_scores['Naive Bayes'], color='m', width=barWidth, edgecolor='grey', label='Naive Bayes')

plt.xlabel('Stroke Instances (0 and 1)', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(recall_scores['Logistic Regression']))], ['0', '1'])
plt.ylabel('Recall', fontweight='bold', fontsize=15)
plt.legend()
plt.show()


# In[53]:


from sklearn.metrics import f1_score

#using the predictions from the models we can find the F1 score

#LogisticRegression
f1_LG_1 = f1_score(y_test, y_pred_Lg, pos_label=1)
precision_LG_1 = precision_score(y_test, y_pred_Lg,pos_label=1)
recall_LG_1 = recall_score(y_test, y_pred_Lg, pos_label=1)

f1_LG_0 = f1_score(y_test, y_pred_Lg, pos_label=0)
precision_LG_0 = precision_score(y_test, y_pred_Lg,pos_label=0)
recall_LG_0 = recall_score(y_test, y_pred_Lg, pos_label=0)

# Predictions from KNN model
f1_knn_1 = f1_score(y_test, y_pred_knn,pos_label=1)
precision_knn_1 = precision_score(y_test, y_pred_knn, pos_label=1)
recall_knn_1 = recall_score(y_test, y_pred_knn, pos_label=1)

f1_knn_0 = f1_score(y_test, y_pred_knn,pos_label=0)
precision_knn_0 = precision_score(y_test, y_pred_knn, pos_label=0)
recall_knn_0 = recall_score(y_test, y_pred_knn, pos_label=0)

# Predictions from Naive Bayes model
f1_nb_1 = f1_score(y_test, y_pred_nb,pos_label=1)
precision_nb_1 = precision_score(y_test, y_pred_nb, pos_label=1)
recall_nb_1 = recall_score(y_test, y_pred_nb, pos_label=1)

f1_nb_0 = f1_score(y_test, y_pred_nb,pos_label=0)
precision_nb_0 = precision_score(y_test, y_pred_nb, pos_label=0)
recall_nb_0 = recall_score(y_test, y_pred_nb, pos_label=0)

# Predictions from Decision Tree model
f1_dt_1 = f1_score(y_test, y_pred_dt,pos_label=1)
precision_dt_1 = precision_score(y_test, y_pred_dt, pos_label=1)
recall_dt_1 = recall_score(y_test, y_pred_dt, pos_label=1)

f1_dt_0 = f1_score(y_test, y_pred_dt,pos_label=0)
precision_dt_0 = precision_score(y_test, y_pred_dt, pos_label=0)
recall_dt_0 = recall_score(y_test, y_pred_dt, pos_label=0)


# Predictions from Random Forest model
f1_rf_1 = f1_score(y_test, y_pred_rf,pos_label=1)
precision_rf_1 = precision_score(y_test, y_pred_rf, pos_label=1)
recall_rf_1 = recall_score(y_test, y_pred_rf, pos_label=1)

f1_rf_0 = f1_score(y_test, y_pred_rf,pos_label=0)
precision_rf_0 = precision_score(y_test, y_pred_rf, pos_label=0)
recall_rf_0 = recall_score(y_test, y_pred_rf, pos_label=0)

# Predictions from SVM
f1_svm_1 = f1_score(y_test, y_pred_svm, pos_label=1)
precision_svm_1 = precision_score(y_test, y_pred_svm, pos_label=1)
recall_svm_1 = recall_score(y_test, y_pred_svm, pos_label=1)

f1_svm_0 = f1_score(y_test, y_pred_svm, pos_label=0)
precision_svm_0 = precision_score(y_test, y_pred_svm, pos_label=0)
recall_svm_0 = recall_score(y_test, y_pred_svm, pos_label=0)


models = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'SVM']

recall_class_1 = [recall_LG_1, recall_knn_1, recall_nb_1, recall_dt_1, recall_rf_1, recall_svm_1]
precision_class_1 = [precision_LG_1, precision_knn_1, precision_nb_1, precision_dt_1,precision_rf_1,precision_svm_1]
f1_class_1 = [f1_LG_1, f1_knn_1, f1_nb_1, f1_dt_1, f1_rf_1, f1_svm_1]

recall_class_0 = [recall_LG_0, recall_knn_0, recall_nb_0, recall_dt_0, recall_rf_0, recall_svm_0]
precision_class_0 = [precision_LG_0, precision_knn_0, precision_nb_0, precision_dt_0, precision_rf_0, precision_svm_0]
f1_class_0 = [f1_LG_0, f1_knn_0, f1_nb_0, f1_dt_0, f1_rf_0, f1_svm_0 ]

x = np.arange(len(models))  # the label locations
bar_width = 0.2 #bar width

# Bar chart for class 1 (stroke)
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - bar_width, recall_class_1, width=bar_width, label='Recall')
ax.bar(x, precision_class_1, width=bar_width, label='Precision')
ax.bar(x + bar_width, f1_class_1, width=bar_width, label='F1 Score')

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Metrics for Patients with Stroke (Class 1)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Bar chart for class 0 (no stroke)
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - bar_width, recall_class_0, width=bar_width, label='Recall')
ax.bar(x, precision_class_0, width=bar_width, label='Precision')
ax.bar(x + bar_width, f1_class_0, width=bar_width, label='F1 Score')

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Metrics for Patients without Stroke (Class 0)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.show()


# # **Saving the datasets**

# In[54]:


#save as csv
#df_resampled.to_csv('/content/drive/MyDrive/Y3S2/FDM/Project/pre_processed_data.csv', index = False)


# In[55]:


pip install joblib


# In[56]:


import joblib

# Save the trained model as a .sav file
joblib.dump(best_rf_classifier, 'random_forest_model.sav')


# In[57]:


df_resampled.to_csv("cleaned_dataset.csv", index = False)


# In[62]:


# Get user inputs
gender = input("Gender (Male/Female): ").capitalize()
age = int(input("Age (1-100): "))
hypertension = input("Hypertension (Yes/No): ").capitalize()
heart_disease = input("Heart Disease (Yes/No): ").capitalize()
ever_married = input("Ever Married (Yes/No): ").capitalize()
work_type = input("Work Type (Private/Self-employed/Children/Govt_job/Never_worked): ").capitalize()
residence_type = input("Residence Type (Urban/Rural): ").capitalize()
avg_glucose_level = float(input("Average Glucose Level: "))
bmi = float(input("BMI: "))
smoking_status = input("Smoking Status (Formerly Smoked/Never Smoked/Smokes): ").capitalize()    


# In[ ]:


prediction = 


# In[66]:


# Dummy variable mappings
gender_mapping = {'Male': 1, 'Female': 0}
ever_married_mapping = {'Yes': 1, 'No': 0}
residence_type_mapping = {'Urban': 1, 'Rural': 0}
hypertension_mapping = {'Yes': 1, 'No': 0}
heart_disease_mapping = {'Yes': 1, 'No': 0}
smoking_status_mapping = {'Never Smoked': 1, 'Smokes': 1, 'Formerly Smoked': 0}

# Get user inputs (assuming you have collected user inputs in variables like age, gender, etc.)
# For example:
# age = 35
# gender = 'Male'
# ... (other inputs)

# Create a dictionary with user inputs and mapped dummy variables
user_data_dict = {
    'age': [age],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'gender_Male': [gender_mapping.get(gender, 0)],
    'ever_married_Yes': [ever_married_mapping.get(ever_married, 0)],
    'Residence_type_Urban': [residence_type_mapping.get(residence_type, 0)],
    'hypertension': [hypertension_mapping.get(hypertension, 0)],
    'heart_disease': [heart_disease_mapping.get(heart_disease, 0)],
    'smoking_status_never smoked': [smoking_status_mapping.get(smoking_status, 0)],
    'smoking_status_smokes': [smoking_status_mapping.get(smoking_status, 0)],
    'work_type_Private': [1 if work_type == 'Private' else 0],
    'work_type_Self-employed': [1 if work_type == 'Self-employed' else 0],
    'work_type_children': [1 if work_type == 'children' else 0],
    'work_type_Never_worked': [1 if work_type == 'Never_worked' else 0]
}

# Create a DataFrame from the dictionary
user_data_df = pd.DataFrame(user_data_dict)

# Print input data for debugging
print("User Input Data:")
print(user_data_df)

# Make prediction
prediction = best_svm_classifier.predict(user_data_df)

print(prediction)

# Output prediction result
if prediction[0] == 0:
    print("Congratulations! You have a low risk of stroke.")
else:
    print("Warning! You are at a high risk of stroke.")

