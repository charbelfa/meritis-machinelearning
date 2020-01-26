# Exercice 2

# I

# 1.
# a
import numpy as np
import pandas as pd
from sklearn import linear_model, svm
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

rain_dataset = pd.read_csv("weatherAUS.csv")
print(rain_dataset.head())
print(rain_dataset.columns)
print(f"Shape: {rain_dataset.shape}")

# b.
rain_dataset = rain_dataset.dropna(how='any')
print(rain_dataset.shape)

# c.
rain_dataset = rain_dataset.drop(
                columns=['Date', 'Sunshine', 'RISK_MM', 'Location', 'Cloud9am', 'Cloud3pm', 'Evaporation'],
                  axis=1)
print(rain_dataset.shape)

# d.
rain_dataset['RainTomorrow'].replace({'Yes': 1,
                                        'No': 0
                                      },
                                     inplace=True)
rain_dataset['RainToday'].replace({'Yes': 1,
                                     'No': 0
                                   },
                                  inplace=True)

# e.
rain_dataset = pd.get_dummies(rain_dataset,
                              columns=['WindGustDir','WindDir9am','WindDir3pm'],
                              drop_first=True)

print(rain_dataset.columns)
print(rain_dataset.shape)

# f.
y = rain_dataset['RainTomorrow']
x = rain_dataset.drop(labels=['RainTomorrow'],
                      axis=1
                      )
print(f'X Shape: {x.shape}')
print(f'y Shape: {y.shape}')

# #####
# 2.
x = StandardScaler().fit_transform(x)
y = y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_test.shape)

# II
# 1.
k_fold_cv = KFold(n_splits=10)
svm_scores = []
lr_scores = []

svm_models = []
lr_models = []

loop = 1
for train_index, test_index in k_fold_cv.split(x_train):

    x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    print(f'Loop {loop}')
    # creer lr model
    lr_model_temp = linear_model.LogisticRegression().fit(x_train_fold, y_train_fold)
    # Sauvegarder modele
    lr_models.append(lr_model_temp)
    # Predire score et le stocker
    lr_score = lr_model_temp.score(x_test_fold, y_test_fold)
    lr_scores.append(lr_score)

    # SVM
    # creer svm model
    svm_model = svm.SVC().fit(x_train_fold, y_train_fold)
    svm_models.append(svm_model)
    # predire score et le stocker
    svm_score = svm_model.score(x_test_fold, y_test_fold)
    svm_scores.append(svm_score)

    print(f"Loop {loop}: LR Score: {lr_score}, SVM score: {svm_score}")
    loop += 1


# Determiner meilleure modele

score_svm_moyenne = np.mean(svm_scores)
score_lr_moyenne = np.mean(lr_scores)

print(f'Moyenne: SVM: {score_svm_moyenne}, LR: {score_lr_moyenne}')


if score_svm_moyenne > score_lr_moyenne:
    # Gagnant svm:
    model_avec_max_score_index = np.argmax(svm_scores)
    print(f"Gagnant: SVM modèle de fold {model_avec_max_score_index+1}")
    model_final = svm_models[model_avec_max_score_index]

else:
    # gagnant lr:

    model_avec_max_score_index = np.argmax(lr_scores)
    print(f"Gagnant: Logistic Regression modèle de fold {model_avec_max_score_index+1}")
    model_final = lr_models[model_avec_max_score_index]

# Test
print(f'Final Test Result: {model_final.score(x_test,y_test)}')
