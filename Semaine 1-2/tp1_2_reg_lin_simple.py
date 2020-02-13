import pandas as pd
import numpy as np
import random

###########
# Partie I
# I. Implementation sans lib
#########
# a. Importation et repartition des donnees

gpa_dataset = pd.read_csv('gpa dataset.csv')
print(gpa_dataset.head())

# Entrees/features
x = gpa_dataset.iloc[:,:-1].values[:,0:]
labels = gpa_dataset.iloc[:,1:].values

# Separer en 80% train et 20% Test

# générer des indices aleatoires
rows, cols = gpa_dataset.shape
indices_aleatoires = random.sample(
    range(rows),
    rows
)
i = int(rows * 0.8)

# Normalisation des X: (Mean Normalization)
def normaliser_x(x):
    max_val = x.max()
    min_val = x.min()
    avg_val = x.mean()

    normalized_data_list = []
    for i, val in enumerate(x[:,0]):
         normalized_data_list.append(
             [(val - avg_val) / (max_val - min_val)]
         )

    return np.array(normalized_data_list)


x = normaliser_x(x)

# Separation des X en train et test
x_train = x[indices_aleatoires[:i]]
x_test = x[indices_aleatoires[i:]]
y_train = labels[indices_aleatoires[:i]]
y_test = labels[indices_aleatoires[i:]]


# b. Fonction de coût
def fonction_cout(x, y, theta):
    prediction = np.dot(x, theta)
    return ((prediction - y[:,0]) ** 2).mean() / 2


# c. Apprentissage desc de gradient.
# Fonction lineaire simple: Y = theta0 + theta1.X

# Garder historique des thetas et couts pour chaque itertation d'apprentissage
theta_0 = []
theta_1 = []
couts = []

# Commencer par des thetas aléatoires
thetas = np.random.rand(2)
print(f'Thetas Initiaux: {thetas}')


# Rajouter des "1" dans la colonne des X0 (pour éviter de supprimer theta0)
input_train = np.column_stack(
    (
        np.ones(len(x_train)),
        x_train
    )
)

taux_apprentissage = 0.001

for i in range(10000):
    # Produit scalaire
    prediction = np.dot(input_train, thetas)
    theta_0_temp = thetas[0] - taux_apprentissage * (prediction - y_train[:,0]).mean()  #Theta_0
    theta_1_temp = thetas[1] - taux_apprentissage * ((prediction - y_train[:,0]) * input_train[:,1]).mean()

    # Mettre a jour les thetas
    thetas = np.array([
        theta_0_temp,
        theta_1_temp
    ])

    # Garder historique
    theta_0.append(theta_0_temp)
    theta_1.append(theta_1_temp)

    couts.append(fonction_cout(input_train, y_train, thetas))

    if i % 100 == 0:
        print(f"Iteration: {i + 1}, Cout = {couts[-1]}, theta = {thetas}")

print(f"theta0 = {theta_0[-1]}, theta1 = {theta_1[-1]}, Cout={couts[-1]}")


# e. Testing: comparer predicted vs test labels
input_test = np.column_stack(
    (
        np.ones(len(x_test)),
        x_test
    )
)

y_test_prediction = np.dot(input_test, thetas)

resultats = np.column_stack(
    (
        y_test_prediction,
        y_test
    )
)

print("Y_Test Prediction; Y_Test_Label")
print("--------------------------------")
print(resultats)


###############
## PARTIE II: Utilisation de Scikit learn###
###############
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("Utilisation de Scikit Learn")

input_data = gpa_dataset.iloc[:,:-1].values
y = gpa_dataset.iloc[:,1:].values

x_train, x_test, y_train, y_test = train_test_split(input_data,y,test_size = 0.2, random_state=0)
reg_lineaire = LinearRegression()
reg_lineaire.fit(x_train, y_train)

y_pred = reg_lineaire.predict(x_test)

for i in range(len(y_test)):
    print(f'{y_pred[i]}; {y_test[i]}')

print(f"Score {r2_score(y_test,y_pred)}")