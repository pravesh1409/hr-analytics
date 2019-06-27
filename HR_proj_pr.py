import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',500)
dataset = pd.read_csv('train_HR.csv')
print('Dataset Loaded')
dataset.describe()
dataset.info()
#remove duplicates
dataset.drop_duplicates(keep=False,inplace=True)
#count null values
dataset.isnull().sum()
#education:2409
#previous_year_rating:4124
dataset.head(5)
#filling missing values
dataset['previous_year_rating'].fillna(3.0,inplace=True)
dataset['education'].describe()
dataset['education'].fillna('Bachelor\'s',inplace=True)
dataset.isnull().sum()
#No nulls#
#correlation
import seaborn as sns
corr=dataset.corr()
sns.heatmap(data=corr,square=True,annot=True,cbar=True)
#high correlation age: removing age column
del dataset['age']
#one hot encoding:to convert categorical data to numbers
dataset=pd.get_dummies(dataset,columns=['department','education','gender','recruitment_channel'])
#train test split
X =dataset.iloc[:,[2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]
Y=dataset.iloc[:,8]
#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3,random_state=50)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
###################################
#Logistic Regression 
classifier = LogisticRegression(random_state=1)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test) #To predict Y values
####################################
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
####################################
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
####################################
#KNN
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=4)  
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#Error rate 
error = []
# Calculating error for K values between 1 and 40
for i in range(1, 25):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
#plotting
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 26), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  
###################################

#################################### 
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#get accuracy
print('Accuracy of decision tree classifier: {:.2f}'.format(classifier.score(X_test, y_test)))
#performance metrics
from sklearn.metrics import precision_recall_fscore_support
all=precision_recall_fscore_support(y_test, y_pred, average='macro')
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)

#ACTUAL TEST DATA
test_data = pd.read_csv('test_HR.csv')

test_data.isnull().sum()
test_data.drop_duplicates(keep=False,inplace=True)
#filling missing values
test_data['previous_year_rating'].mode()
test_data['previous_year_rating'].fillna(3.0,inplace=True)
test_data['education'].describe()
test_data['education'].mode()
test_data['education'].fillna('Bachelor\'s',inplace=True)
test_data.isnull().sum() # to check

#Correlation matrix
corr=test_data.corr()
sns.heatmap(data=corr,square=True,annot=True,cbar=True)
#high correlation age: removing age column
del test_data['age']
del test_data['employee_id']
del test_data['region']
#one hot encoding:to convert categorical data to numbers
test_data=pd.get_dummies(test_data,columns=['department','education','gender','recruitment_channel'])
#transform
test_data = sc.fit_transform(test_data)
#To predict
results = classifier.predict(test_data)
#export
results=pd.DataFrame(results) 
results.to_csv("D:\Pravesh\Machine Learning\ML_capstone/dec_results.csv")