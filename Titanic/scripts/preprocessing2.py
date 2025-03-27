import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Obrobka2:
    def __init__(self):
        path = 'data/test.csv' 
        train_data = pd.read_csv(path)
        

        # Mount of samples
        # print(f"{train_data.shape} \n\n")

        # Deleting useless features
        train_data = train_data.drop(columns=['Name','Ticket'])

        # Checking which columns have empty values
        # print(f"{train_data.isnull().sum()} \n\n") 

        # Because of most samples in the Cabin are missing we decide to delete column 'Cabin'ArithmeticError
        train_data = train_data.drop(columns=['Cabin'])

        # Filling Age columns with median
        train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

        # Filling Embarked with the mode (dominanta)
        train_data['Embarked'].fillna(train_data['Embarked'].mode(), inplace=True) 

        # Mapping Sex
        train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})

        train_data = train_data.drop(columns=['PassengerId'])
        

        # Using one-hot encoding on Embarked
        train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True) 
        """
            It creates new binary columns for each unique value in the categorical columns
            In other words I create columns with names: C, Q, S (in this case) and binary values inside of them 
            
            drop first: Prevents multicollinearity (a situation where one feature can be predicted from others, which can cause issues in some machine learning models).
            A passenger who boarded at Cherbourg (C) will have 0 in both Embarked_Q and Embarked_S.
            A passenger who boarded at Queenstown (Q) will have 1 in Embarked_Q and 0 in Embarked_S.
            A passenger who boarded at Southampton (S) will have 0 in Embarked_Q and 1 in Embarked_S."
        """

        # Checking if there are so anomaly in the all features i.e. spelling mistake


        self.train_float_data = train_data.astype(float)
        """
        
    
        # Splitting dataset for training and testing 
        X = train_float_data.drop(columns=['Survived'])
        y = train_float_data['Survived']

        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

        # Normalizing Fare but always after spliting datasets, because of data leackage 
        self.X_train['Fare'] = (self.X_train['Fare'] - self.X_train['Fare'].mean()) / self.X_train['Fare'].std() # Syntax: (value_of_sample - mean_of_samples) / samples standard deviation
        self.X_test['Fare'] = (self.X_test['Fare'] - self.X_test['Fare'].mean()) / self.X_test['Fare'].std() # Syntax: (value_of_sample - mean_of_samples) / samples standard deviation

        #print(X_train)

        # print(train_data)
        """
        # print('\n\n')"
        
    def zwrot(self):
        return(self.train_float_data)
        