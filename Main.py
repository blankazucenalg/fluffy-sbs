import csv

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

earthquakes_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

Y_train = earthquakes_df["Dangerous"]

Y_test = test_df["Dangerous"]

# drop labels from training set, given that label is the dependant feature
# and we want to predict label given predictors or independent features
X_train = earthquakes_df.drop("Magnitude",axis=1)
X_train = earthquakes_df.drop("Dangerous",axis=1)

# Test set (droping rows with NULL values)
# make a copy of PassengerId
X_test  = test_df.drop("Magnitude",axis=1).dropna().copy()
X_test  = test_df.drop("Dangerous",axis=1).dropna().copy()

#lin = LinearRegression() #initialize regressor
lin = LogisticRegression()
lin.fit(X_train, Y_train) #fit training data

preds = lin.predict(X_test) #make prediction on X test set

results = open('results,csv','w')
results.write("Dangerous\n")
for p in preds:
    results.write(p.__str__()+"\n")
results.close()

print "Porcentaje de efectividad: "+lin.score(X_train, Y_train).__str__() + "%"