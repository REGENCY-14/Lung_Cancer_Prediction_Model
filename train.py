#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN












df=pd.read_csv('survey_lung_cancer.csv')
df



df.shape



df.duplicated().sum()



df=df.drop_duplicates()



df.isnull().sum()



df.info()



df.describe()




le = preprocessing.LabelEncoder()

columns_to_encode = [
    'GENDER', 'LUNG_CANCER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY',
    'CHEST PAIN'
]

for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])



df



df.info()



sns.countplot(x='LUNG_CANCER', data=df,)
plt.title('Target Distribution');



df['LUNG_CANCER'].value_counts()



def plot(col, df=df):
    return df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))



plot('GENDER')



df_new=df.drop(columns=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
df_new



cn=df_new.corr()
cn



cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap=cmap,annot=True, square=True)
plt.show()



kot = cn[cn>=.40]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Blues")



df_new['ANXYELFIN']=df_new['ANXIETY']*df_new['YELLOW_FINGERS']
df_new



X = df_new.drop('LUNG_CANCER', axis = 1)
y = df_new['LUNG_CANCER']



adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)



len(X)



X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)



lr_model=LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)



y_lr_pred= lr_model.predict(X_test)
y_lr_pred



lr_cr=classification_report(y_test, y_lr_pred)
print(lr_cr)



dt_model= DecisionTreeClassifier(criterion='entropy', random_state=0)  
dt_model.fit(X_train, y_train)



y_dt_pred= dt_model.predict(X_test)
y_dt_pred



dt_cr=classification_report(y_test, y_dt_pred)
print(dt_cr)



rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)



y_rf_pred= rf_model.predict(X_test)
y_rf_pred



rf_cr=classification_report(y_test, y_rf_pred)
print(rf_cr)




lr_accuracy = accuracy_score(y_test, y_lr_pred)
dt_accuracy = accuracy_score(y_test, y_dt_pred)
rf_accuracy = accuracy_score(y_test, y_rf_pred)

print(f'Logistic Regression Accuracy: {lr_accuracy}')
print(f'Decision Tree Accuracy: {dt_accuracy}')
print(f'Random Forest Accuracy: {rf_accuracy}')



best_model = rf_model

best_model.fit(X_train, y_train)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)


print("Final Model Evaluation on Training Set")
train_accuracy = accuracy_score(y_train, y_train_pred)
train_cr = classification_report(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")
print("Training Classification Report:")
print(train_cr)

print("Final Model Evaluation on Test Set")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_cr = classification_report(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")
print("Test Classification Report:")
print(test_cr)



import pickle



with open('best_rfmodel','wb') as file:
    pickle.dump(best_model,file)
    
print("Model saved as'best_rfmodel'")