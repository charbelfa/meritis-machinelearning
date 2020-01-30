import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Etape 1: Importer le fichier csv contenant l'ensemble des données.
startup_data = pd.read_csv(r'50_Startups.csv')
print(startup_data.columns)

# Verifier que nos données ne contiennet pas de bruits:
print(
    startup_data
        .isnull()
        .sum()
        .sort_values(ascending=False)
)

# Etape 2: Séparer nos données en features et labels (Sortie)
data = startup_data.iloc[:, :-1].values
labels = startup_data.iloc[:, -1].values

print(f'Entree Data: {data}')
print(f'Sortie Labels: {labels}')

# Etape 3: Pre-traitement
# Encoder la colonne state
data[:, 3] = LabelEncoder().fit_transform(data[:, 3])

# Utiliser la technique one hot encoder pour encoder nos donnees
data = OneHotEncoder(categorical_features=[3]).fit_transform(data).toarray()
data = data[:, 1:]

# Etape 4: Apprentissage de modele
# Séparer nos données en training et Test Sets (20% Test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Definir modele et apprentissage (Regression lineaire multivariee)
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Test et score
# On va utiliser la technique R2 Score pour evaluer notre modele
y_hat = linear_regression.predict(X_test)
print(
    r2_score(
        y_test,
        y_hat,
        multioutput='uniform_average'
    )
)
