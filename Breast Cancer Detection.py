import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data=pd.read_csv('C:\machinelearning\data\data.csv')
data.drop(['id'],axis=1,inplace=True)
data['diagnosis']=(data['diagnosis']=='M').astype(int)
Processed_Data=data.drop(['diagnosis'],axis=1)
Parameters=['radius_mean','perimeter_mean', 'compactness_mean', 'concave points_mean', 'radius_worst','perimeter_worst', 'texture_worst','perimeter_se','radius_se','compactness_se','concave points_se','compactness_worst','concave points_worst', 'area_worst', 'concavity_mean']
Processed_Data=Processed_Data.drop(Parameters,axis=1)
def Limit(column):
    a,b=np.nanpercentile(column,[25, 75])
    diff=b-a
    up=b+1.5*diff
    low=a-1.5*diff
    return up,low
    for column in Processed_Data.columns:
        if Processed_Data[column].dtype!='object':
            up,low=Limit(Processed_Data[column])
            Processed_Data[column]=np.where((Processed_Data[column]>up) | (Processed_Data[column]<low),np.nan, Processed_Data[column])
    iterative_imputer
from sklearn.impute import KNNImputer
imputer=KNNImputer(n_neighbors=4)
Processed_Data.iloc[:, :]=imputer.fit_transform(Processed_Data)
Y=data['diagnosis']
X=Processed_Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=50)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=50)
lr.fit(x_train,y_train)
yPredict=lr.predict(x_test)
print('Accuracy: {}'.format(accuracy_score(y_test, yPredict)))
print('Recall: {}'.format(recall_score(y_test, yPredict)))
def CM(confusion):
    groups=['TN','FP','FN','TP']
    counts=['{0:0.0f}'.format(value)for value in confusion.flatten()]
    labels=np.asarray([f'{v1}\n{v2}' for v1, v2 in zip(groups,counts)]).reshape(2, 2)
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion,annot=labels,cmap='Blues',cbar=False, fmt='')
    plt.show()
CM(confusion_matrix(y_test, yPredict))