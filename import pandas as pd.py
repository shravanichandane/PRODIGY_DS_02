#importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#loading dataset

df = pd.read_csv(r'C:\Users\shrav\Downloads\archive.zip') #copy path of the file
print(df.head())


#cheak missing values

print(df.isnull().sum())


#Handle missing values

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].value_counts().index[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)


#Exploratory Data Analysis

#Explore relationships between variables

sns.scatterplot(x='Age', y='Fare', data=df)
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

sns.boxplot(x='Pclass', y='Fare', data=df)
plt.show()

print(df.groupby('Pclass')['Survived'].mean())
print(df.groupby('Pclass')['Fare'].mean())
print(df.groupby(pd.cut(df['Age'], bins=4))['Survived'].mean())


#Create new features

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 16, 32, 64, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])


#Explore relationships between variables

sns.countplot(x='AgeGroup', hue='Survived', data=df)
plt.show()

sns.boxplot(x='AgeGroup', y='Fare', data=df)
plt.show()

print(df.groupby('AgeGroup')['Survived'].mean())
print(df.groupby('AgeGroup')['Fare'].mean())

