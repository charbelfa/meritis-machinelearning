import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split


maladie_dataset = pd.read_csv('framingham_heart_disease.csv')
maladie_dataset.rename(
    columns={'male': 'sex'}
)

maladie_dataset.dropna(axis=0, inplace=True)
print(maladie_dataset['totChol'])

# Separer les donnees en X et Y
inputs = maladie_dataset.drop('TenYearCHD', 1)
inputs = inputs.drop('currentSmoker', 1)
labels = maladie_dataset['TenYearCHD']

features = inputs.columns

# Question 3: Fonction plotting

def histo(df, features, rows, cols):
    figure = plt.figure(figsize=(25,25))

    for indice, feature in enumerate(features):
        subplot = figure.add_subplot(rows, cols, indice+1)
        df[feature].hist(
            bins=20,
            ax=subplot,
            facecolor='skyblue'
        )
        subplot.set_title(feature, color='Green')

    figure.tight_layout()
    plt.show()

histo(maladie_dataset, maladie_dataset.columns, 5,4)


# Learning Model
x_train, x_test, y_train, y_test = train_test_split(
    inputs, labels, test_size=0.2, random_state=0, stratify=labels
)

reg_log = LogisticRegression(solver='liblinear')
reg_log.fit(x_train, y_train)


# Test
y_test_prediction = reg_log.predict(x_test)
y_test_prediciton_quant = reg_log.predict_proba(x_test)[:, 1]
print(y_test_prediciton_quant)


######################
# Score

print('Precision: ', accuracy_score(y_test, y_test_prediction))
matrice_confusion = confusion_matrix(y_test, y_test_prediction)
print(matrice_confusion)

sensibilite = matrice_confusion[0, 0] / (matrice_confusion[0, 0] + matrice_confusion[1, 0])
print('Sensibilite: ', sensibilite)

specificite = matrice_confusion[1, 1] / (matrice_confusion[1, 1] + matrice_confusion[0, 1])
print('Specificite: ', specificite)

fp, tp, thresholds = roc_curve(y_test, y_test_prediciton_quant)
print('AUC: ', auc(fp, tp))

figure, ax = plt.subplots()
ax.plot(fp,tp)
ax.plot(
    [0,1],
    [0,1],
    transform = ax.transAxes,
    ls='--',
    c='.3'
)

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])

plt.xlabel('Taux FP  (1-Specificite)')
plt.ylabel('Taux TP: Sensibilite')
plt.grid(True)
#plt.show()

