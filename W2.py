#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from pprint import pprint

#%%
# Importing csv dataset into a numpy structure
# QSAR DATASET: https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation

X = np.genfromtxt('data/biodeg.csv', delimiter=";", skip_header=0, usecols=range(0, 41))    # 1055x41
y = np.genfromtxt('data/biodeg.csv', delimiter=";", skip_header=0, usecols=-1, dtype=str)   # 1055x1
classes = np.unique(y)   # "RB": Ready-Biodegradable, "NRB": Not Ready-Biodegradable
print(classes)

# Distribution of classes
for i in classes:
    print("Number of " + i + " classes: " + str(sum(y==i)))    # RB: 356; NRB: 669

# replace "NRB" with 0, and "RB" with 1
with np.nditer(y, op_flags=['readwrite']) as it:
    for i in it:
        map_v = 0
        for c in classes:
            if i == c:
                i[...] = map_v
                break
            map_v +=1
y=y.astype(int)

#%%
# preprocessing: ( x - mean(x) / std(x) )
X = StandardScaler().fit_transform(X)

# Split dataet into test and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13, stratify=y)

#%% PCA in order to reduce to 2 components
# Note that this PCA will be only used to visualizate the whole DATASET
# Later in cross-validation, a PCA must be performed for each fold.
myPCA = PCA(n_components=2)
Xp_train = myPCA.fit_transform(X_train)
Xp_test = myPCA.transform(X_test) # applying SAME transformation to test data
Xp = np.concatenate((Xp_train, Xp_test))

#%%
# A dictionary with classifier and their param grid
# Research parameters selection...
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

#%% plotting dataset
figure = plt.figure(figsize=(36, 18))
h = 0.01 # mesh step size
i = 1

x_min, x_max = Xp[:, 0].min() - .5, Xp[:, 0].max() + .5
y_min, y_max = Xp[:, 1].min() - .5, Xp[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(2, (len(classifiers) + 1)/2, 1) # +1 because one subplot is for input data
ax.set_title("Input data")

# Plot the training points
ax.scatter(Xp_train[:, 0], Xp_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
# Plot the testing points
ax.scatter(Xp_test[:, 0], Xp_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

#%%
# Trying all classifiers (cross-validating each one)
test_scores = {}

for (estimator, param_grid) in classifiers.items():
    myGSCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy',
                          cv=StratifiedKFold(n_splits=10, random_state=3),
                          n_jobs=-1)
    myGSCV.fit(X_train, y_train)
    y_pred = myGSCV.predict(X_test)

    # save test score of the classifier
    test_score = myGSCV.score(X_test, y_test)
    estimator_name = str(estimator).split("(")[0]
    test_scores.update({estimator_name: test_score})

    # Results
    print("\n" + estimator_name)
    print("Best parameters:\t" + str(myGSCV.best_params_))  # parameters of the best estimator
    print("Training score:\t" + str(test_score))  # training score for achieved with the best estimator
    print("Test score:\t" + str(myGSCV.score(X_test, y_test)))  # test score

    # Plotting (Using dimension reduced variables Xp)

    # Xp = np.concatenate((Xp_train, Xp_test))
    myGSCV.fit(Xp_train, y_train, class_weight=class_weights) # fit for PCA
    yp_pred = myGSCV.predict(Xp_test) # prediction for PCA

    i += 1
    ax = plt.subplot(2, (len(classifiers) + 1)/2, i)
    if hasattr(myGSCV, "decision_function"):
        Z = myGSCV.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = myGSCV.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

    # Plot the training points
    ax.scatter(Xp_train[:, 0], Xp_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(Xp_test[:, 0], Xp_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(estimator_name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % test_score).lstrip('0'),
            size=15, horizontalalignment='right')

#%%
plt.tight_layout()
plt.show()

#%%
# Best classifiers based on accuracy score:
scores_list = [(k, test_scores[k]) for k in sorted(test_scores, key=test_scores.get, reverse=True)]
print("\nTest Accuracy Scores Ranking per Classifier:")
pprint(scores_list)

#%%
# TODO:
#   - Test with other scoring systems (ROC->AUC, Precission Recall)
#   - Plot confusion matrix
#   - Test other estimators
#   - Research and document parameters election, as well as stratification process
#   - Evaluation of statistical significancy of models in terms of performance (use: wilcoxon, boxplot)
#   - Research and apply other techniques, such us: no lineal features selection, sample selections,
#     weight balancing (useful since the dataset is unbalanced)

#%% NOTES
# PCA only for visualization...
# Add with a note that if scores and accuracy are measured for pca model, pca should be implemented in each fold
# con el kfold estamos probando tambien diferentes folds de datos, por eso hay que hacer cada vez el pca en
# fold, pero a la hora de representarlo, realmente es Xp total lo que representamos y para lo que son las
# regiones, es usar el predictor conseguido con el kfold y yasta.