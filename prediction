#import libraries

# Import required libraries
import pandas as pd
import numpy as np 
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/trevorkarn/MLCamp2022/main/high_diamond_ranked_10min.csv')
df.columns
# df.head()

#KNN
features = ['blueWins','blueWardsPlaced', 'blueWardsDestroyed',
      'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
      'blueEliteMonsters', 'blueDragons', 'blueHeralds',
      'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
      'blueTotalExperience', 'blueTotalMinionsKilled',
      'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
      'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
      'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
      'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
      'redTotalGold', 'redAvgLevel', 'redTotalExperience',
      'redTotalMinionsKilled', 'redTotalJungleMinionsKilled','redCSPerMin', 'redGoldPerMin']

X = df[features].values
y = df["blueWins"].values
 
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=.7, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=1, random_state=7)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=1, random_state=42)
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

k_list = list(range(1,50,2))
cv_scores = []

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#This is the cross validation to find the best K number
# for k in k_list:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train1, y_train1, cv = StratifiedKFold(shuffle=True), scoring = 'accuracy')
#     cv_scores.append(scores.mean())

# plt.figure()
# plt.title('Performance of K Nearest Neighbors Algorithm')
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Accuracy Score')
# plt.plot(k_list, cv_scores)

# plt.show()
# best_k = k_list[cv_scores.index(max(cv_scores))]
# print(best_k)

df.head()

# df.blueWins.value_counts(normalize = True)

classifierNK1 = KNeighborsClassifier(n_neighbors=45)
classifierNK1.fit(X_train1, y_train1)

#testing
y_pred1 = classifierNK1.predict(X_test1)
accuracy_score(y_test1, y_pred1)

#training
y_train_pred1 = classifierNK1.predict(X_train1)
accuracy_score(y_train1, y_train_pred1)

target_column = ['blueWins'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

from sklearn.neural_network import MLPClassifier


lst = [[8,8,8],[5,5,5],[1,2,3],[50,50,50],[42, 21, 39]]
results = []
for case in lst:

  mlp = MLPClassifier(hidden_layer_sizes=(case[0],case[1],case[2]), activation='relu', solver='adam', max_iter=500)
  mlp.fit(X_train,y_train)

  predict_train = mlp.predict(X_train)
  predict_test = mlp.predict(X_test)

  from sklearn.metrics import classification_report,confusion_matrix
  print(confusion_matrix(y_train,predict_train))
  print(classification_report(y_train,predict_train))

  print(confusion_matrix(y_test,predict_test))
  print(classification_report(y_test,predict_test))

  #decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)
accuracy_score(y_pred, y_test)

classifier1a = tree.DecisionTreeClassifier(max_depth=7)
accDt1a = cross_val_score(classifier1a, X, y, cv=5)
accDt1as = sum(accDt1a)
accDt1as /= 5
print(accDt1as)
print()

classifier2a = tree.DecisionTreeClassifier(max_depth=8)
accDt2a = cross_val_score(classifier2a, X, y, cv=5)
accDt2as = sum(accDt2a)
accDt2as /= 5
print(accDt2as)
print()

classifier3a = tree.DecisionTreeClassifier(max_depth=9)
accDt3a = cross_val_score(classifier3a, X, y, cv=5)
accDt3as = sum(accDt3a)
accDt3as /= 5
print(accDt3as)
print()

from sklearn.ensemble import RandomForestClassifier# random forest
rf = RandomForestClassifier(max_depth=4)
rf = rf.fit(X_train1, y_train1)

rf_pred = rf.predict(X_test1)
accuracy_score(rf_pred, y_test1)

from sklearn.ensemble import RandomForestClassifier# random forest
rf = RandomForestClassifier(max_depth=2)
rf = rf.fit(X_train2, y_train2)

rf_pred = rf.predict(X_test2)
accuracy_score(rf_pred, y_test2)

from sklearn.ensemble import RandomForestClassifier# random forest
rf = RandomForestClassifier(max_depth=9)
rf = rf.fit(X_train3, y_train3)

rf_pred = rf.predict(X_test3)
accuracy_score(rf_pred, y_test3)

df = df.dropna()
df.describe(include="all")
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

features = ['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed',
       'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
       'blueEliteMonsters', 'blueDragons', 'blueHeralds',
       'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
       'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
       'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
       'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
       'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
       'redTotalGold', 'redAvgLevel', 'redTotalExperience',
       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff',
       'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin']

lol_X = df[features]
lol_y = df['blueWins']

lol_X_train, lol_X_test, lol_y_train, lol_y_test = train_test_split(lol_X, lol_y, test_size=0.25, random_state=42)

from sklearn import svm

# svc = svm.SVC(kernel='linear')
# svc = svc.fit(lol_X_train, lol_y_train)

# y_pred = svc.predict(lol_X_test)

from sklearn.metrics import confusion_matrix

# confusion_matrix(lol_y_test, y_pred)

from sklearn.metrics import accuracy_score

# accuracy_score(y_pred, lol_y_test)

# poly_svc = svm.SVC(kernel='poly')
# poly_svc = poly_svc.fit(lol_X_train, lol_y_train)

# y_pred = poly_svc.predict(lol_X_test)

# accuracy_score(y_pred, lol_y_test)

from sklearn import svm

classifier = svm.SVC(kernel='linear')

temp = cross_val_score(classifier, X, y, cv=5)

print(sum(temp)/len(temp))
