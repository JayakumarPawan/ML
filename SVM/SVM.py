from sklearn.svm import SVC
import numpy as np
import pandas as pd
import collections as cc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# Importing the dataset
train = pd.read_csv('F:/ML_and_AI/TJ-Machine learning/SVM/training.csv')
test = pd.read_csv('F:/ML_and_AI/TJ-Machine learning/SVM/testing.csv')


#drop unnecesary columns
train = train.drop(['PassengerId','Parch','SibSp','Name','Cabin','Ticket'],axis = 1)
test = test.drop(['PassengerId','Parch','Name','SibSp','Cabin','Ticket'],axis = 1)

#seperate into X and y
Y = train.iloc[:,0]
X = train.drop('Survived',axis=1)


#Categorize data
from sklearn import preprocessing

X['Sex'] = preprocessing.LabelEncoder().fit_transform(X['Sex'])
test['Sex'] = preprocessing.LabelEncoder().fit_transform(test['Sex'])

X['Embarked'] = preprocessing.LabelEncoder().fit_transform(X['Embarked'])
test['Embarked'] = preprocessing.LabelEncoder().fit_transform(test['Embarked'])
# filling missing data
X['Income'].fillna(X['Income'].mean(), inplace = True)
X['Income'].values.reshape( -1,1)
test['Income'].fillna(test['Income'].median(), inplace = True)
test['Income'].values.reshape( -1,1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_test = StandardScaler()
test = sc_test.fit_transform(test)


C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range,)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
print('done')
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,verbose=10)
print('done')
grid.fit(X, Y)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
'''
with open("sampleSubmission.csv",'w') as files:
    files.write("id,solution\n")
    counter = 1
    answers = clf.predict(test)
    print(cc.Counter(answers)[1]/(cc.Counter(answers)[0]+cc.Counter(answers)[1]))
    for answer in answers:
        files.write(str(counter)+","+str(answer)+"\n")
        counter+=1'''
