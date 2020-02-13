# TP Détection d'anomalie
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import precision
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import math

# I
# ACP:
# 2.
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('creditcard.csv')
# Print shape and form
print(dataset.head())
print(dataset.shape)
print(dataset.columns)


# II
# Visualiser les features anomalie v/s normaux


def tracer_courbes():
    # Ayant 31 features, il faudra pouvoir tout les visualiser
    for index, feature in enumerate(dataset.columns):
        plt.figure()

        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        # Plotter 2 courbes sur le même axe: une pour classe=0, autre classe = 1

        # Courbe normaux:
        sns.distplot(
            dataset[feature][dataset['Class'] == 0],
            color='g',
            label='Cas normal (Pas de Fraud)'
        )
        # Courbe Fraud
        sns.distplot(
            dataset[feature][dataset['Class'] == 1],
            color='r',
            label='Cas Fraud'
        )

        plt.legend()

        print(f'Tracer {feature}')
        # save plot
        plt.savefig(f"{feature}.png")


# tracer_courbes()

print('Dropper les colonnes inutiles')
# Dropper les colonnes inutiles
dataset.drop(
    labels=[
        'Time',
        'V8',
        'V13',
        'V15',
        'V20',
        'V22',
        'V23',
        'V24',
        'V25',
        'V26',
        'V27',
        'V28',
    ],
    inplace=True,
    axis=1
)
print(f'Nouvelle structure: {dataset.shape}')
print(f'Colonnes: {dataset.columns}')
#######

# III
# 1. Repartition des donnees en fraud et normaux
normal_dataset = dataset[dataset['Class'] == 0]
fraud_dataset = dataset[dataset['Class'] == 1]

# De la normal dataset: 60% train, 20% CV, 20% Test
normal_train, normal_test = train_test_split(normal_dataset, test_size=0.4)
normal_cv, normal_test = train_test_split(normal_test, test_size=0.5)

# 50 % de la base fraud pour les CV et 50% pour les tests
fraud_cv, fraud_test = train_test_split(fraud_dataset, test_size=0.5)

# normal: normal_train, normal_cv, normal_test
# fraud: fraud_cv, fraud_test

# Train seulement avec les donnees normaux
train_dataset_X = normal_train.drop(
    labels='Class',
    axis=1
)

# base CV
cv_dataset_X = pd.concat([normal_cv, fraud_cv])
cv_dataset_y = cv_dataset_X['Class']

cv_dataset_X.drop(
    labels='Class',
    axis=1,
    inplace=True
)

# base tests
test_dataset_X = pd.concat([normal_test, fraud_test])
test_dataset_y = test_dataset_X['Class']

test_dataset_X.drop(
    labels='Class',
    axis=1,
    inplace=True
)

# 2.

standard_scaler = StandardScaler()

# Fit transform car donnees d'apprentissage
train_dataset_X = standard_scaler.fit_transform(train_dataset_X)
cv_dataset_X = standard_scaler.transform(cv_dataset_X)
test_dataset_X = standard_scaler.transform(test_dataset_X)


# 3.
def calculer_gaussienne(dataset):
    # Moyenne de chaque feature (exemple, on calcule mu des X1, mu des X2, etc.) ce qui va retourner une liste de
    # moyenne
    moyenne = np.mean(
        dataset,
        axis=0
    )

    # Même chose pour écart-type
    ecart_type = np.std(
        dataset,
        axis=0,

    )
    print(ecart_type)
    return moyenne, ecart_type


# Calculer la liste des moyennes mu1, mu2, ... mu_n et sigma_1,...sigma_n sur la train set
moyennes, ecart_types = calculer_gaussienne(train_dataset_X)


# Construire p(X)
def proba_x(data, mu_list, sigma_list):
    proba = 0

    # Les exp peuvent être transformés en addition car e^a . e^b = e^(a+b)
    fonction_expo = 0
    for x_j, mu_j, sigma_j in zip(data, mu_list, sigma_list):
        fonction_expo += -((x_j-mu_j)**2/(2*sigma_j**2))
    #print(fonction_expo)


    # [1/(sqrt(2*pi)*sigma_j)]*1/(sqrt(2*pi)*sigma_j)]*... = 1/( (sqrt(2*pi)^n) * (sigma_1*sigma_2*...*sigma_n))
    # n étant le nombre de features (size of X)
    denom_sigma = 1
    for sigma in sigma_list:
        denom_sigma *= sigma

    fraction = 1 / (math.sqrt(2 * math.pi) ** len(mu_list) * denom_sigma)

    return fraction * math.exp(fonction_expo)


def proba_x(data, mu_list, sigma_list):
    proba = 1
    for x_j, mu_j, sigma_j in zip(data, mu_list, sigma_list):
        expo = math.exp((((-x_j) + mu_j) ** 2 / (-2 * sigma_j ** 2)))
        proba *= (1 / (math.sqrt(2 * math.pi) * sigma_j) * expo)
    return proba


# IV

# 1. La classification n'est pas une bonne technique à considérer ici car
# les données ne sont pas équilibriment réparties entre 0 et 1 (Normale et anormale)
# On a beaucoup plus de données normales alors que 0.17% anormales.

# 2.

def get_predicted_label(data, mu_list, sigma_list, epsilon):
    return 0 if proba_x(data, mu_list, sigma_list) > epsilon else 1


epsilon_list = np.arange(1e-15, 1e-5, 5e-10)

print(epsilon_list)
print(len(epsilon_list))

y_predicted = [proba_x(x_cv, moyennes, ecart_types) for x_cv in cv_dataset_X]

#for proba
for epsilon in epsilon_list:
    y_predicted_labels = [0 if y>epsilon else 1 for y in y_predicted]
    print(accuracy_score(cv_dataset_y, y_predicted_labels))
