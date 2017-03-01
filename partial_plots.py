#some stuff often useful...
from __future__ import division
from ggplot import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


#read data set
conv_rates_df = pd.read_csv('data/conversion_data.csv')

#Python needs to create dummy variables, can't have categorical variable with many levels
conv_rates_df['country'] = conv_rates_df['country'].map(lambda x: 'DE' if x == 'Germany' else x);
target = conv_rates_df.pop('converted')
df_transformed = pd.get_dummies(conv_rates_df)

#randomize order
df_transformed.iloc[np.random.permutation(len(df_transformed))].head()

#build boosting. Only boosting has partial plots in Python 
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X_1 = df_transformed
y_1 = target

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.30, random_state=0)

#keep estimators as 1 for now if you want to just play around. 
#Once you build the actual model, best number is often between 50 and 200. 
model_1 = GradientBoostingClassifier(n_estimators=1)
y_pred_1 = model_1.fit(X_train_1, y_train_1).predict(X_test_1)


##This is the main point. After you build the model the code below will plot all partial plots

from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence


#example for the most important variable (just need to change features_enum = [2] to get others)
#Often useful to have a for loop going through all variables and then plotting them.
#Then, in the challenge solution, you can focus on the plots that (1) are most important and (2) agree with your overall story
features_enum = [1]
fig, axs = plot_partial_dependence( model_1, 
                                    X_train_1, 
                                    features_enum, 
                                    feature_names=df_transformed.columns,
                                    n_jobs=3, grid_resolution=100)
plt.show()
