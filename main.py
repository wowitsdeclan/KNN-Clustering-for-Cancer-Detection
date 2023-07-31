'''
CP468 Final Project
Declan Hollingworth - 190765210
Mubin Qureshi - 180181900
July 31st, 2023
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
#from google.colab import drive
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier

#drive.mount('/content/gdrive/')

#Data Exploration and Cleaning

"""
1. Sample code number: id number
2. Clump Thickness: 1 - 10
3. Uniformity of Cell Size: 1 - 10
4. Uniformity of Cell Shape: 1 - 10
5. Marginal Adhesion: 1 - 10
6. Single Epithelial Cell Size: 1 - 10
7. Bare Nuclei: 1 - 10
8. Bland Chromatin: 1 - 10
9. Normal Nucleoli: 1 - 10
10. Mitoses: 1 - 10
11. Class: (2 for benign, 4 for malignant)
"""
#Loading the Dataset from your PC where the .csv file is located

col_names = ["Sample code number", "Clump Thickness (1-10)", "Uniformity of Cell Size (1-10)", "Uniformity of Cell Shape (1-10)", "Marginal Adhesion (1-10)", "Single Epithelial Cell Size (1-10)", "Bare Nuclei (1-10)",
           "Bland Chromatin (1-10)", "Normal Nucleoli (1-10)", "Mitoses (1-10)", "Class (2 for benign, 4 for malignant)"]

data1 = pd.read_csv("breast-cancer-wisconsin.data", names=col_names)
#data1

# object data type inside of BareNuclei tells us that there are string or null
# values present in that datatype
data1.info()

# No Null values
data1.isnull().sum()

# No NaN values
data1.isna().sum()

# Investigation into the data set showed that the column had '?' values
# We remove those here and assign to data2
bare_nuc_missing = data1.loc[data1['Bare Nuclei (1-10)'] == '?']
missing_values = bare_nuc_missing.index.tolist()
data2 = data1.drop(index=missing_values)
data2.info()

# convert data2 to int64 now that the column is all of type integer
data2['Bare Nuclei (1-10)'] = data2['Bare Nuclei (1-10)'].astype(np.int64)
data2.info()

#map malignant and benign to 1 and 2 instead of 2 and 4
data2['Class (2 for benign, 4 for malignant)'] = data2['Class (2 for benign, 4 for malignant)'].map({2: 0, 4: 1})

#make sure that all sample code numbers are unique
data2['Sample code number'].is_unique

#Remove duplicates from data frame
sample_code = data2["Sample code number"]
data2[data2.isin(data2[data2.duplicated()])].sort_values("Sample code number")

# Check state of data after dropping duplicates
data2 = data2.drop_duplicates(subset='Sample code number', keep=False, inplace=False).reset_index(drop=True)
#data2

# Now that we know that we don't have any duplicate entities, we can
# remove the sample code from the data as it bears no significance
# to our classification
removedSampleCodeData = data2.drop('Sample code number', axis=1)

"""Cleaning performed on the data:

*   Consolidating duplicate entries - Where two entries have a duplicate sample code
    *  Determined which entries have duplicates and removed duplicate entities
    * removed any NaN columns
*   Handling null entries - in form of ? - specifically inside of the Bare Nuclei column
    * Removed entries w/ null values in this column
* Ensured that all features are correctly within the range of 1..10

# Finding Outliers
"""

#this function will be used to find outliers by finding any values that fall outside of the interquartile range

def find_outliers_IQR(df):

   q1=df.quantile(0.25)
   q3=df.quantile(0.75)

   IQR=q3-q1

   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

   return outliers


#when considering outliers in our dataset, it is important to understand that statistical outliers in any of these features are likely to be indicators of cancer.
#If we just calculate and remove outliers, we will remove many real datapoints that had unusually high values, even though the high values are real, and do in fact indicate malignancy.
#To counterract this, we consider outliers within the malginant subset, and outliers within the benign subset. This way, an outlier is no longer just an unusually high or low value, but it
#is an unusually high or low value considering that the datapoint was malignant or benign.
#this works especially well since there is a high correlation between every feature in our dataset.
print("-----------------malignant-------------------------------------------------------")
data_malignant = data2[data2["Class (2 for benign, 4 for malignant)"] == 1]
for i in data_malignant:
  print(i, "has", len(find_outliers_IQR(data_malignant[i])), "outliers")


print("-----------------benign-------------------------------------------------------")
data_benign = data2[data2["Class (2 for benign, 4 for malignant)"] == 0]

for i in data_benign:
  print(i, "has", len(find_outliers_IQR(data_benign[i])), "outliers")

"""# Removing Outliers

Since our
dataset is relatively small, it will be best to leave some of the outliers
untouched, and use imputation/clamp transformations on the others to minimize the amount of data lost. The malignant subset of data has relatively few outliers, so we will make no changes to this subset of the data.
"""

def clamp_transform(data, feature, lower, upper):
  return data[(data[feature] < upper) & (data[feature] > lower)]

#using clamp transformation to resolve outliers in uniformity of cell shape
shape_mean = data_benign["Uniformity of Cell Shape (1-10)"].mean()
print(shape_mean)
sns.boxplot(y = data_benign["Uniformity of Cell Shape (1-10)"])
#remove all values form dataset if they aren't between 0 and 3
data_benign = clamp_transform(data_benign, "Uniformity of Cell Shape (1-10)", 0, 3)


#using clamp transformation to resolve outliers in uniformity of cell size
size_mean = data_benign["Uniformity of Cell Size (1-10)"].mean()
print(size_mean)
sns.boxplot(y = data_benign["Uniformity of Cell Size (1-10)"])
#chose 3 as cutoff for clamp transformation
data_benign = clamp_transform(data_benign, "Uniformity of Cell Size (1-10)", 0,3)

#using clamp transformation to resolve outliers in single epithelial cell size

epi_mean = data_benign["Single Epithelial Cell Size (1-10)"].mean()

print(epi_mean)

data_benign = clamp_transform(data_benign, "Single Epithelial Cell Size (1-10)", 0,3)

frames = [data_malignant, data_benign]

data2_clean = pd.concat(frames)

data2_clean.info()

"""# **Model Selection**"""

#splitting data into training data and testing data

x_col_names =  ["Clump Thickness (1-10)", "Uniformity of Cell Size (1-10)", "Uniformity of Cell Shape (1-10)", "Marginal Adhesion (1-10)", "Single Epithelial Cell Size (1-10)", "Bare Nuclei (1-10)",
           "Bland Chromatin (1-10)", "Normal Nucleoli (1-10)", "Mitoses (1-10)"]
x = data2[x_col_names]
y = data2["Class (2 for benign, 4 for malignant)"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#one test/train split will represent the data with outliers removed, and the other will represent the data that still has outliers.
x2_col_names =  ["Clump Thickness (1-10)", "Uniformity of Cell Size (1-10)", "Uniformity of Cell Shape (1-10)", "Marginal Adhesion (1-10)", "Single Epithelial Cell Size (1-10)", "Bare Nuclei (1-10)",
           "Bland Chromatin (1-10)", "Normal Nucleoli (1-10)", "Mitoses (1-10)"]
x2 = data2_clean[x2_col_names]
y2 = data2_clean["Class (2 for benign, 4 for malignant)"]
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2)

"""# Logistic Regression

First we will initialize our sklearn logreg model. We will set the penalty to L2 so that every input feature is considered towards the logit function.
"""

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(penalty = 'l2')
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

logreg=LogisticRegression(penalty = 'l2')
logreg.fit(x2_train,y2_train)
y2_pred=logreg.predict(x2_test)

"""# Logistic Regression Model Performance and Results"""

from sklearn.metrics import accuracy_score

#accuracy of model with outliers still in dataset
score1 = accuracy_score(y_test,y_pred)
print("with outliers remaining, our model is ", score1*100, "% accurate")
#accuracy of model after removing outliers
score2 = accuracy_score(y2_test,y2_pred)
print("with outliers removed, our model is ", score2*100, "% accurate")

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted: Benign','Predicted:Malignant'],index=['Actual: Benign','Actual: Malignant'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

print("data with outliers:")
print("using logistic regression we get a log loss of", log_loss(y_test, y_pred))
print("using logistic regression we get a Mean Squared Error of ", mean_squared_error(y_test,y_pred))

print("data without outliers")
print("using logistic regression we get a log loss of", log_loss(y2_test, y2_pred))
print("using logistic regression we get a Mean Squared Error of ", mean_squared_error(y2_test,y2_pred))

"""## **K Nearest Neighbors**

Choosing optimal value for K
"""

minimum_error = 10000
final_k = 0
k_sum = 0
for i in range(50):
  x_col_names =  ["Clump Thickness (1-10)", "Uniformity of Cell Size (1-10)", "Uniformity of Cell Shape (1-10)", "Marginal Adhesion (1-10)", "Single Epithelial Cell Size (1-10)", "Bare Nuclei (1-10)",
           "Bland Chromatin (1-10)", "Normal Nucleoli (1-10)", "Mitoses (1-10)"]
  x = data2[x_col_names]
  y = data2["Class (2 for benign, 4 for malignant)"]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  for k in range(1,51,2):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    KNeighborsClassifier(k)
    y_pred = knn.predict(x_test)
    error = mean_squared_error(y_test, y_pred)
    if (error < minimum_error):
      minimum_error = error
      final_k = k
  k_sum = k_sum + final_k

final_k = k_sum //50
print(final_k)



#the code above will try several different k values for 50 unique
#test/train splits. For each test/train split, it will choose an
#optimal value for k by choosing the value that results in the
#lowest mean square error. It then chooses the average of all
#of the optimal k values for each test/train split, and
#chooses that as the overall optimal k value.

"""#K nearest Neighbors Model Performance and Results"""

from sklearn.neighbors import KNeighborsClassifier
#Implementing KNN on data2 with outliers remaining

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(x_train,y_train)
KNeighborsClassifier(7)
y_pred = knn.predict(x_test)
accuracy = knn.score(x_test, y_test)

print("using K Nearest Neighbors to predict whether a datapoint is benign or malignant, with 5 neighbors we get an accuracy of ", accuracy)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted: Benign','Predicted:Malignant'],index=['Actual: Benign','Actual: Malignant'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print("using K Nearest Neighbors we get a Mean Squared Error of ", mean_squared_error(y_test,y_pred))


#implementing KNN on data2 with outliers removed
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(x2_train,y2_train)
KNeighborsClassifier(7)
y2_pred = knn.predict(x2_test)
accuracy = knn.score(x2_test, y2_test)

print("using K Nearest Neighbors to predict whether a datapoint is benign or malignant, with 7 neighbors we get an accuracy of ", accuracy)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y2_test,y2_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted: Benign','Predicted:Malignant'],index=['Actual: Benign','Actual: Malignant'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print("using K Nearest Neighbors we get a Mean Squared Error of ", mean_squared_error(y2_test,y2_pred))