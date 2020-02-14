import matplotlib.pyplot

import pandas as pd
import numpy as np
import scipy.special
#
train_set = pd.read_csv('mnist_train.csv')
test_set = pd.read_csv('mnist_test.csv')
print(train_set.head())
print(f"Train: {len(train_set)}")
print(f"Test: {len(test_set)}")


# Visualisation
mnist_train = open('mnist_train.csv')
train_list = mnist_train.readlines()
mnist_train.close()

mnist_test = open('mnist_test.csv')
test_list = mnist_test.readlines()
mnist_test.close()

# exemple_valeur = training_list[1].split(',')
# image = np.asfarray(exemple_valeur[1:]).reshape((28, 28))
# print(image)

# x = matplotlib.pyplot.imshow(img, cmap='Greys', interpolation=None)


## Reseaux de Neurones
class reseaux_de_neurones:

    def __init__(self, nb_inputs, nb_caches, nb_output, taux_apprentissage):
        self.nodes_in = nb_inputs
        self.nodes_caches = nb_caches
        self.nodes_out = nb_output
        self.taux_apprentissage = taux_apprentissage
        self.fonction_activation = lambda x: scipy.special.expit(x)

        self.weight_caches_entree = np.random.normal(
            0.0,
            pow(self.nodes_caches, -0.5),
            (self.nodes_caches, self.nodes_in)
        )
        self.weight_cache_sortie = np.random.normal(
            0.0,
            pow(self.nodes_caches, -0.5),
            (self.nodes_out, self.nodes_caches)
        )

    def prediction(self, inputs_list):
        # Convertir input list of 2D et la transposer
        inputs = np.array(inputs_list, ndmin=2).T

        # Calculer signal en entree a la couche cachee
        in_caches = np.dot(self.weight_caches_entree, inputs)
        out_caches = self.fonction_activation(in_caches)

        # Entree et sortie finale
        in_final = np.dot(self.weight_cache_sortie, out_caches)
        out_final = self.fonction_activation(in_final)

        return out_final

    def train(self, inputs_list, targets_list):
        entree = np.array(inputs_list, ndmin=2).T
        sortie = np.array(targets_list, ndmin=2).T

        # Calculer le signal dentree dans la couche cachee
        in_caches =  np.dot(self.weight_caches_entree, entree)
        out_caches = self.fonction_activation(in_caches)

        # Calculer signal dans la couche de sortie
        in_final = np.dot(self.weight_cache_sortie, out_caches)
        out_final = self.fonction_activation(in_final)

        ## Backpropagation
        # Erreur est la différence entre valeur observee et predite
        erreur_sortie = sortie - out_final
        erreur_caches = np.dot(self.weight_cache_sortie.T, erreur_sortie)

        self.weight_cache_sortie += self.taux_apprentissage * np.dot(
            (erreur_sortie * out_final * (1 - out_final)), np.transpose(out_caches)
        )

        self.weight_caches_entree += self.taux_apprentissage * np.dot(
            (erreur_caches * out_caches * (1 - out_caches)), np.transpose(entree)
        )
        #print(f"Poids: W_Cache_Entree: {self.weight_caches_entree}, W_Cache_Sortie: {self.weight_cache_sortie}")


##########
inodes = 784
hnodes = 300
onodes = 10

learning_rate = 0.1

reseau = reseaux_de_neurones(inodes, hnodes, onodes, learning_rate)

print(reseau.weight_caches_entree.shape)
print(reseau.weight_cache_sortie.shape)

epochs = 10
for epoch in range(epochs):

    # boucler sur tous les records:
    print(f"Epoch {epoch}")

    for entry in train_list[1:]:

        all_values = entry.split(',')

        # Ecart et Shifter les inputs quand on a un pixel noir (0) pour eviter de perdre la valeur du poids associé
        entrees = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # Creer target values
        labels_sortie = np.zeros(onodes) + 0.01
        labels_sortie[int(all_values[0])] = 0.99

        reseau.train(entrees, labels_sortie)


# Scoreboard

score = []
for entry in test_list[1:]:
    all_values = entry.split(',')

    # observation:
    valeur_observee = int(all_values[0])

    entrees = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = reseau.prediction(entrees)

    # indice de la plus grande valeur du vecteur represente la valeur predite.
    valeur_predite = np.argmax(outputs)

    if valeur_predite == valeur_observee:
        score.append(1)
    else:
        score.append(0)


score_array = np.asfarray(score)

print("Performance: ", score_array.sum()/score_array.size)


# Visualisation