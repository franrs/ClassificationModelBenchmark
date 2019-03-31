import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Importing csv dataset into a numpy structure
X = np.genfromtxt('data/biodeg.csv', delimiter=";", skip_header=0, usecols=range(0, 41))    #1055x41
y = np.genfromtxt('data/biodeg.csv', delimiter=";", skip_header=0, usecols=-1, dtype=str)   #1055x1
classes = ["RB", "NRB"]   # Ready-Biodegradable, Not Ready-Biodegradable

# Distribution of classes
for i in classes:
    print("Number of " + i + " classes: " + str(sum(y==i)))    # RB: 356; NRB: 669

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, stratify=y)

# K Nearest Neighbors Classifier is going to be used
myKNN = KNeighborsClassifier()

# A param grid dictionary is created with the parameters to try in the cross-validation process
# TODO: Make it so that it doesn't cross-validate 'p' value for 'euclidean' and 'manhattan' metrics.
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan', 'minkowski'],
              'p': [2, 3, 4, 5, 6, 7, 8, 9]}

# KNN estimator, accuracy as scoring, and the data will be split into 10 chunks, thus, it will take 10 iterations
myGSCV = GridSearchCV(estimator=myKNN, param_grid=param_grid, scoring='accuracy',
                      cv=StratifiedKFold(n_splits=10, random_state=3),
                      verbose=2, n_jobs=-1)

# Training of the model
myGSCV.fit(X_train, y_train)

# prediction (using best_estimator_ by default)
y_pred = myGSCV.predict(X_test)

# Results
print("\nBest Estimator:\n" + str(myGSCV.best_estimator_))  # best estimator
print("\nParameters of best estimator:\n" + str(myGSCV.best_params_))   # parameters of the best estimator
print("\nTraining score: " + str(myGSCV.best_score_))  # training score for achieved with the best estimator
print("Test score: " + str(myGSCV.score(X_test, y_test)))  # test score

# Print ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y, myGSCV.score, pos_label=2)
