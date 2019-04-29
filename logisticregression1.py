import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import math
dataset=pd.read_csv("heart.csv")
dataset.drop(['slope','sex','age','ca','oldpeak','fbs'],axis=1 ,inplace=True)
X=dataset.drop('target',axis=1)
Y=dataset["target"]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(Y_test,predictions)
print(classification_report(Y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,predictions)
print(confusion_matrix(Y_test,predictions))
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predictions)) 






