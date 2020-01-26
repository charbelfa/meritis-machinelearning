# Exercice 1:
import pandas as pd
from numpy import argmax
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt

# I
# 1.
from sklearn.model_selection import train_test_split, KFold, cross_val_score

diabetes_dataset = datasets.load_diabetes()

# 2.
cols = 'age,sex,bmi,map,tc,dl,hdl,tch,ltg,glu'.split(',')
diabetes_X_df = pd.DataFrame(diabetes_dataset.data, columns=cols)
X = diabetes_dataset.data
y = diabetes_dataset.target
print(X.shape, y.shape)

# Train et test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Train: ", x_train.shape, y_train.shape)
print("Test: ", x_test.shape, y_test.shape)

# 3.
model_lr = linear_model.LinearRegression().fit(x_train, y_train)

# 4.
# Plot
y_prediction = model_lr.predict(x_test)
plt.scatter(y_test, y_prediction)
plt.xlabel('Valeurs observées')
plt.ylabel('Valeurs Prédites')
plt.show()
# Idealement on veut avoir une ligne y=x (linéaire)

score = model_lr.score(x_test, y_test)
print(score)

# 5.

# II

k_fold_cv = KFold(n_splits=10)
fold_scores = []
models = []
loop = 1
for train_index, test_index in k_fold_cv.split(x_train):
    x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    model_lr_temp = linear_model.LinearRegression().fit(x_train_fold, y_train_fold)
    models.append(model_lr_temp)

    score_temp = model_lr_temp.score(x_test_fold,y_test_fold)
    print(f"Loop {loop}: {score_temp}")

    fold_scores.append(score_temp)
    loop += 1


print(fold_scores)
max_score_index = argmax(fold_scores)

# test
print("modele avec max score: ", max_score_index)
model_lr_max_score = models[max_score_index]

print("General Test result: ", model_lr_max_score.score(x_test, y_test))

# Autre méthode:
val_scores = cross_val_score(linear_model.LinearRegression(), x_train, y_train, cv=k_fold_cv)
print('Val Scores using cross_val_score: ', val_scores)