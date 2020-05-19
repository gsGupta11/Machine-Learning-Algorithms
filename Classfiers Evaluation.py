
"""
**Import Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""**Importing DataSet**<br>
Features are from coulumn 0 to last-1
 Dependent Vector is the last column
"""

dataset = pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

"""**Handle Missing Value if any**

**Encoding if reqd**

**Splitting into Test and Trainnning Datasets**
"""

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.25,random_state=0)

print(Xtrain)

print(Xtest)

print(Ytrain)

print(Ytest)

"""**Feature Scaling of Feature matrix**"""

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)

print(Xtrain)

"""**Trainning Different Models**"""

#Logictics Classification
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(Xtrain,Ytrain)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=10)
knc.fit(Xtrain,Ytrain)

#SVC Linear
from sklearn.svm import SVC
svcl = SVC(kernel="linear",random_state=0)
svcl.fit(Xtrain,Ytrain)

#SVC Kernel
svck = SVC(kernel="rbf",random_state=0)
svck.fit(Xtrain,Ytrain)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy",random_state=0)
dtc.fit(Xtrain,Ytrain)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20,criterion="entropy",random_state=0)
rfc.fit(Xtrain,Ytrain)

"""**Predicting a new o/p using each model;**"""

print("O/p by Logistics",lr.predict(ss.transform([[1018099,1,1,1,1,2,10,3,1,1]])))
print("O/p by KNN",knc.predict(ss.transform([[1018099,1,1,1,1,2,10,3,1,1]])))
print("O/p by SVC Linear",svcl.predict(ss.transform([[1018099,1,1,1,1,2,10,3,1,1]])))
print("O/p by SVC Kernel",svck.predict(ss.transform([[1018099,1,1,1,1,2,10,3,1,1]])))
print("O/p by Decision Tree",dtc.predict(ss.transform([[1018099,1,1,1,1,2,10,3,1,1]])))
print("O/p by Random Forest",rfc.predict(ss.transform([[1018099,1,1,1,1,2,10,3,1,1]])))

"""**Predicting Train Set O/p using each model**"""

ypredlr=lr.predict(Xtest)
ypredknn=knc.predict(Xtest)
ypredsvcl=svcl.predict(Xtest)
ypredsvck=svck.predict(Xtest)
ypreddt=dtc.predict(Xtest)
ypredrf=rfc.predict(Xtest)

"""**Evaluating the Best Model**"""

from sklearn.metrics import confusion_matrix,accuracy_score
clr =confusion_matrix(ypredlr,Ytest)
cknn=confusion_matrix(ypredknn,Ytest)
csvcl=confusion_matrix(ypredsvcl,Ytest)
csvck=confusion_matrix(ypredsvck,Ytest)
cdt=confusion_matrix(ypreddt,Ytest)
crf=confusion_matrix(ypredrf,Ytest)
arr=[clr,cknn,csvcl,csvck,cdt,crf]
alr = accuracy_score(ypredlr,Ytest)
aknn = accuracy_score(ypredknn,Ytest)
asvcl = accuracy_score(ypredsvcl,Ytest)
asvck = accuracy_score(ypredsvck,Ytest)
adt = accuracy_score(ypreddt,Ytest)
arf = accuracy_score(ypredrf,Ytest)

d={"Logistics":alr,"KNN":aknn,"SVC-Linear":asvcl,"SVC-NON Linear":asvck,"Decision Tree":adt,"Random Forest":arf}

print("Confusion Matrices and Accuracy Scores are:")
k=0
m=0
name = ""
for i in d:
  print(i)
  print(arr[k])
  print(d[i])
  if d[i]>m:
    m=d[i]
    name=i
  k+=1

print("The best suited Classification Model is: "+name+" with the accuracy of "+str(m*100) + " percent")
