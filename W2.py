#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#%%
# Importing csv dataset into a numpy structure
X = np.genfromtxt('data/biodeg.csv', delimiter=";", skip_header=0, usecols=range(0, 41))    # 1055x41
y = np.genfromtxt('data/biodeg.csv', delimiter=";", skip_header=0, usecols=-1, dtype=str)   # 1055x1
classes = ["RB", "NRB"]   # Ready-Biodegradable, Not Ready-Biodegradable

#%%
# Distribution of classes
for i in classes:
    print("Number of " + i + " classes: " + str(sum(y==i)))    # RB: 356; NRB: 669

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, stratify=y)

#%%
# A dictionary with classifier and their param grid
classifiers = {
    KNeighborsClassifier(): {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5, 6]
    },    # 7, 8, 9
    DecisionTreeClassifier(): {
        'max_depth': [None, 3, 5, 7],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [5, 7, 10, 15]
    },
    MLPClassifier(): {
        'hidden_layer_sizes': [(100,), (10,), (10, 10, 10,), (5, 10, 20, 10, 5,)],
        'activation': ['tanh', 'relu']
    },
    RandomForestClassifier(): {
        'n_estimators': [5, 10, 20, 40, 80, 160, 200, 500, 1000]
    },
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=None)): {
        'n_estimators': [30, 50, 100]
    },
    SVC(): {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.001, 0.0001]
    },
    GaussianNB(): {
    }
}

for (estimator, param_grid) in classifiers.items():
    myGSCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy',
                          cv=StratifiedKFold(n_splits=10, random_state=3),
                          n_jobs=-1)
    myGSCV.fit(X_train, y_train)
    y_pred = myGSCV.predict(X_test)

    # Results
    print("\n" + str(estimator))
    print("Best parameters:\t" + str(myGSCV.best_params_))  # parameters of the best estimator
    print("Training score:\t" + str(myGSCV.best_score_))  # training score for achieved with the best estimator
    print("Test score:\t" + str(myGSCV.score(X_test, y_test)))  # test score

#%%
# TODO:
#   - Save score in a dict (scores = {estimator: test_score, ...}) and sort it by test_score
#   - Test with other scoring systems
#   - Plot confusion matrix (actually ROC? since it's a binary problem)
#   - Test other estimators
#   - Research and document parameters election, as well as stratification process
#   - Evaluation of statistical significancy of models in terms of performance (use: wilcoxon / boxplot)
#   - Representation of Decision Regions (use pca to reduce dimensionality to 2,
#     also refer to https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html)
#   - Research and apply other techniques, such us: no lineal features selection, sample selections,
#     weight balancing (useful since the dataset is unbalanced
