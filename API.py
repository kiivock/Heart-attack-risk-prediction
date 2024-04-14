from fastapi import FastAPI
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}


@app.get("/predict")
def predict(
        Age: int,
        Sex: str,
        ChestPainType  : str,
        RestingBP :int,
        Cholesterol : int,
        FastingBS :int,
        RestingECG: str,
        MaxHR: int,
        ExerciseAngina : str,
        Oldpeak : int,
        ST_Slope : str
    ):
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    data_dict = {
    'Age': [Age],
    'Sex': [Sex],
    'ChestPainType': [ChestPainType],
    'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'FastingBS': [FastingBS],
    'RestingECG': [RestingECG],
    'MaxHR': [MaxHR],
    'ExerciseAngina': [ExerciseAngina],
    'Oldpeak': [Oldpeak],
    'ST_Slope': [ST_Slope]
    }

    X_pred = pd.DataFrame(data_dict, index=[0])

    preprocessor = joblib.load('models/preprocessor.joblib')
    preprocessed_data = pd.DataFrame(preprocessor.transform(X_pred))

    model=joblib.load('models/harp_model.joblib')
    prediction=model.predict(preprocessed_data)

    return {'result': prediction[0]}
