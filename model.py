import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Charger les données
file_path = "Loan_Data.csv"
df = pd.read_csv(file_path)

# Afficher les premières lignes et les informations générales
print("\nPremières lignes du dataset :")
print(df.head())

# Analyse exploratoire (EDA) 

print("\nInformations générales :")
print(df.info())

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nStatistiques descriptives :")
print(df.describe())

print("\nRépartition du target :")
print(df['default'].value_counts(normalize=True))

# Répartition des défauts
sns.countplot(x='default', data=df)
plt.title('Répartition des défauts de paiement')
plt.xlabel('Default')
plt.ylabel('Nombre de clients')
plt.show()

# Visualisation du déséquilibre des classes dans la variable 'default'
sns.countplot(x='default', data=df, palette='Blues')
plt.title('Déséquilibre des classes : Nombre de clients avec et sans défaut')
plt.xlabel('Default')
plt.ylabel('Nombre de clients')
plt.xticks([0, 1], ['Sans défaut', 'Avec défaut'])
plt.show()

# Visualisation du déséquilibre des classes avec un graphique circulaire
class_counts = df['default'].value_counts()
class_labels = ['Sans défaut', 'Avec défaut']

plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Proportion des clients avec et sans défaut')
plt.axis('equal')  # Pour que le graphique soit un cercle parfait
plt.show()


# Histogramme de la variable "income"
sns.histplot(df['income'], kde=True, color='blue')
plt.title('Distribution du revenu')
plt.xlabel('Revenu')
plt.ylabel('Fréquence')
plt.show()

# Histogramme de la variable "fico_score"
sns.histplot(df['fico_score'], kde=True, color='green')
plt.title('Distribution du FICO Score')
plt.xlabel('FICO Score')
plt.ylabel('Fréquence')
plt.show()

# Scatter plot entre "income" et "fico_score"
plt.figure(figsize=(8,6))
sns.scatterplot(x='income', y='fico_score', hue='default', data=df)
plt.title('Relation entre le revenu et le FICO Score')
plt.xlabel('Revenu')
plt.ylabel('FICO Score')
plt.show()

# Pairplot des variables
sns.pairplot(df, hue='default', palette='coolwarm')
plt.title('Pairplot des variables')
plt.show()

# Violin plot pour "fico_score" et "default"
sns.violinplot(x='default', y='fico_score', data=df, palette='coolwarm')
plt.title('Distribution du FICO Score par défaut')
plt.show()

# Corrélation générale
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 10}, linewidths=0.5)
plt.title('Matrice de corrélation améliorée')
plt.show()

# Suppression de l'ID client
df_cleaned = df.drop(columns=["customer_id"])

# Séparer les features et la cible
X = df_cleaned.drop(columns=["default"])
y = df_cleaned["default"]

# Division en train et test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation des variables continues
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Vérification des dimensions après prétraitement
X_train_scaled.shape, X_test_scaled.shape

# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train_scaled, y_train)

# Prédire les valeurs sur l'ensemble de test
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modèle sauvegardé sous 'model.pkl' !")

