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
train = train.drop(['PassengerId','Name','Cabin','Ticket'],axis = 1)
test = test.drop(['PassengerId','Name','Cabin','Ticket'],axis = 1)

#seperate into X and y
#Y = train.iloc[:560,0]
#X = train.drop('Survived',axis=1).iloc[:560,:]

Y = train.iloc[:560,0]
X = train.drop('Survived',axis=1).iloc[:560,:]

#y_test = np.array(train.iloc[560:,0])
#X_test = train.drop('Survived',axis=1).iloc[560:,:]

#Categorize data
from sklearn import preprocessing

X['Sex'] = preprocessing.LabelEncoder().fit_transform(X['Sex'])
test['Sex'] = preprocessing.LabelEncoder().fit_transform(test['Sex'])
#X_test['Sex'] = preprocessing.LabelEncoder().fit_transform(X_test['Sex'])

X['Embarked'] = preprocessing.LabelEncoder().fit_transform(X['Embarked'])
test['Embarked'] = preprocessing.LabelEncoder().fit_transform(test['Embarked'])
#X_test['Embarked'] = preprocessing.LabelEncoder().fit_transform(X_test['Embarked'])

# filling missing data
X['Income'].fillna(X['Income'].median(), inplace = True)
X['Income'].values.reshape( -1,1)

test['Income'].fillna(test['Income'].median(), inplace = True)
test['Income'].values.reshape( -1,1)

#X_test['Income'].fillna(test['Income'].median(), inplace = True)
#X_test['Income'].values.reshape( -1,1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#sc_diag = StandardScaler()
#X_test = sc_X.fit_transform(X_test)

sc_test = StandardScaler()
test = sc_test.fit_transform(test)

clf = SVC( C=100000.0, gamma=0.001)
clf.fit(X,Y)

#check score
'''
Once u found a good one comment out the X_test stuff
train_answers = clf.predict(X_test)
score = 0
for i in range(len(y_test)):
    if y_test[i] == train_answers[i]:
        score+=1
score/=len(y_test)
print(score)
'''

with open("sampleSubmission.csv",'w') as files:
    files.write("id,solution\n")
    counter = 1
    answers = clf.predict(test)
    print(cc.Counter(answers)[1]/(cc.Counter(answers)[0]+cc.Counter(answers)[1]))
    for answer in answers:
        files.write(str(counter)+","+str(answer)+"\n")
        counter+=1
