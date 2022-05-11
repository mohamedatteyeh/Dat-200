import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

training_data = pd.read_csv('train.csv', index_col= 0 ) # Training Data
test_data = pd.read_csv('test.csv',index_col = 0) # Test Data

#____________________________________________________________________
# The submission model, with chosen parameters from the Evaluation.

X = training_data.iloc[:,:-1].copy()
y = training_data.iloc[:,-1].copy()

forest_xtrain = X
forest_ytrain = y
forest_test = test_data
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=163,
                                random_state= 1,
                                n_jobs=-1)
forest.fit(forest_xtrain, forest_ytrain)

# The pridiction and the submission file to Kaggle
forest_target_predictions = forest.predict(forest_test)
output = pd.DataFrame({'index': forest_test.index,'target': forest_target_predictions})
output.to_csv('submission_forest',index=False)


