import pickle
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, SelectFromModel



df = pd.read_csv('heart_cleaned.csv').drop(columns=['Unnamed: 0'])

numerical_pipeline = make_pipeline(
    KNNImputer(n_neighbors=2),  # Impute missing values
    StandardScaler()                 # Scale features
)

# For categorical features: impute missing values, then one-hot encode
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),  # Impute missing values
    OneHotEncoder(handle_unknown="ignore")    # One-hot encode features
)

# Combine numerical and categorical pipelines
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
preproc_baseline = make_column_transformer(
    (numerical_pipeline, numerical_features),
    (categorical_pipeline, make_column_selector(dtype_include='object')),
    remainder='passthrough'
)

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']
preproc_selector_multi = SelectFromModel(
    DecisionTreeClassifier(),
    threshold = "median", # drop all multivariate features lower than the median correlation
)

preproc_multi = make_pipeline(
    preproc_baseline,
    preproc_selector_multi
)
preproc_multi.fit(X,y)

joblib.dump(preproc_multi,'models/preprocessor.joblib')
