import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import RandomOverSampler
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import requests
from io import StringIO

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*test scores are non-finite.*", category=UserWarning, module='sklearn.model_selection._search')

# Streamlit app title
st.title("Total Knee Size Predictor")

@st.cache_data(show_spinner=False)
def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text))
    data.columns = data.columns.str.strip().str.lower()  # Ensure columns are in lower case without spaces
    data['age_height_interaction'] = data['age'] * data['height']
    data['height_log'] = np.log1p(data['height'])
    return data

@st.cache_data(show_spinner=False)
def train_and_scale_models(data, features):
    X = data[features].values
    y_tibia = data['tibia used'].values
    y_femur = data['femur used'].values

    scaler_tibia = StandardScaler().fit(X)
    X_scaled_tibia = scaler_tibia.transform(X)
    tibia_xgb = XGBRegressor(n_estimators=100, random_state=1)
    tibia_gbr = GradientBoostingRegressor(n_estimators=100, random_state=1)
    tibia_ridge = Ridge()

    scaler_femur = StandardScaler().fit(X)
    X_scaled_femur = scaler_femur.transform(X)
    femur_xgb = XGBRegressor(n_estimators=100, random_state=1)
    femur_gbr = GradientBoostingRegressor(n_estimators=100, random_state=1)
    femur_ridge = Ridge()

    tibia_stack = StackingRegressor(estimators=[('xgb', tibia_xgb), ('gbr', tibia_gbr)], final_estimator=XGBRegressor(), cv=5)
    femur_stack = StackingRegressor(estimators=[('xgb', femur_xgb), ('gbr', femur_gbr)], final_estimator=XGBRegressor(), cv=5)

    # Fit models
    tibia_xgb.fit(X_scaled_tibia, y_tibia)
    tibia_gbr.fit(X_scaled_tibia, y_tibia)
    tibia_ridge.fit(X_scaled_tibia, y_tibia)
    tibia_stack.fit(X_scaled_tibia, y_tibia)

    femur_xgb.fit(X_scaled_femur, y_femur)
    femur_gbr.fit(X_scaled_femur, y_femur)
    femur_ridge.fit(X_scaled_femur, y_femur)
    femur_stack.fit(X_scaled_femur, y_femur)

    return {
        'tibia': {'xgb': tibia_xgb, 'gbr': tibia_gbr, 'ridge': tibia_ridge, 'stack': tibia_stack, 'scaler': scaler_tibia},
        'femur': {'xgb': femur_xgb, 'gbr': femur_gbr, 'ridge': femur_ridge, 'stack': femur_stack, 'scaler': scaler_femur}
    }

class TibiaFemurPredictor:
    def __init__(self):
        self.models = None
        self.data = None
        self.prediction_df = None
        self.metrics_df = None

    def oversample_minority_group(self, data):
        if 'sex' not in data.columns:
            st.error("The dataset does not contain the required 'sex' column.")
            return data
        if 0 in data['sex'].values and 1 in data['sex'].values:
            X = data.drop('sex', axis=1)
            y = data['sex']
            ros = RandomOverSampler(random_state=1)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            return pd.concat([X_resampled, y_resampled], axis=1)
        else:
            st.warning("Both male and female samples are required for oversampling.")
            return data

    def train_models(self):
        self.data = self.oversample_minority_group(self.data)
        self.models = train_and_scale_models(self.data, ['height_log', 'age_height_interaction', 'sex'])
        st.session_state['models'] = self.models
        st.success("Models trained and scalers initialized.")

    def predict(self, age, height, sex_val, model_type):
        if not self.models:
            st.error("Models are not trained yet.")
            return

        X_new = np.array([[np.log1p(height), age * height, sex_val]])
        tibia_scaler = self.models['tibia']['scaler']
        femur_scaler = self.models['femur']['scaler']
        X_new_scaled_tibia = tibia_scaler.transform(X_new)
        X_new_scaled_femur = femur_scaler.transform(X_new)

        preds_tibia = self.models['tibia'][model_type].predict(X_new_scaled_tibia)
        preds_femur = self.models['femur'][model_type].predict(X_new_scaled_femur)

        prediction_data = {
            "Model": [model_type.upper()],
            "Predicted Femur": [round(preds_femur[0], 1)],
            "Predicted Tibia": [round(preds_tibia[0], 1)]
        }

        prediction_df = pd.DataFrame(prediction_data)
        self.prediction_df = prediction_df

        st.table(prediction_df)

    def calculate_metrics(self, X, y, bone, model_type):
        model = self.models[bone][model_type]
        preds = model.predict(X)

        residuals = y - preds
        mae = mean_absolute_error(y, preds)
        residuals_kurtosis = kurtosis(residuals)

        metrics = {
            'model': model_type,
            'r2_score': r2_score(y, preds),
            'rmse': mean_squared_error(y, preds, squared=False),
            'mse': mean_squared_error(y, preds),
            'mae': mae,
            'mape': np.mean(np.abs((y - preds) / y)) * 100,
            'kurtosis': residuals_kurtosis,
            'residuals': residuals
        }

        return metrics

    def display_metrics(self):
        features = ['height_log', 'age_height_interaction', 'sex']
        X_tibia = self.data[features].values
        y_tibia = self.data['tibia used'].values
        X_femur = self.data[features].values
        y_femur = self.data['femur used'].values

        X_tibia_scaled = self.models['tibia']['scaler'].transform(X_tibia)
        X_femur_scaled = self.models['femur']['scaler'].transform(X_femur)

        tibia_metrics_xgb = self.calculate_metrics(X_tibia_scaled, y_tibia, 'tibia', 'xgb')
        tibia_metrics_gbr = self.calculate_metrics(X_tibia_scaled, y_tibia, 'tibia', 'gbr')
        tibia_metrics_ridge = self.calculate_metrics(X_tibia_scaled, y_tibia, 'tibia', 'ridge')
        tibia_metrics_stack = self.calculate_metrics(X_tibia_scaled, y_tibia, 'tibia', 'stack')
        femur_metrics_xgb = self.calculate_metrics(X_femur_scaled, y_femur, 'femur', 'xgb')
        femur_metrics_gbr = self.calculate_metrics(X_femur_scaled, y_femur, 'femur', 'gbr')
        femur_metrics_ridge = self.calculate_metrics(X_femur_scaled, y_femur, 'femur', 'ridge')
        femur_metrics_stack = self.calculate_metrics(X_femur_scaled, y_femur, 'femur', 'stack')

        metrics_data = {
            'Metric': ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis'],
            'Tibia XGB': [tibia_metrics_xgb[key] for key in ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis']],
            'Tibia GBR': [tibia_metrics_gbr[key] for key in ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis']],
            'Tibia Ridge': [tibia_metrics_ridge[key] for key in ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis']],
            'Tibia Stack': [tibia_metrics_stack[key] for key in ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis']],
            'Femur XGB': [femur_metrics_xgb[key] for key in ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis']],
            'Femur GBR': [femur_metrics_gbr[key] for key in ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis']],
            'Femur Ridge': [femur_metrics_ridge[key] for key in ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis']],
            'Femur Stack': [femur_metrics_stack[key] for key in ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis']]
        }

        df_metrics = pd.DataFrame(metrics_data)
        self.metrics_df = df_metrics

        st.table(df_metrics)

def main():
    predictor = TibiaFemurPredictor()

    # Dictionary of available CSV files and their URLs
    csv_files = {
        "Data Central Florida": "https://raw.githubusercontent.com/Msmbusiness/Knee/main/data%20central%20florida.csv",
        "Data Midwest": "https://raw.githubusercontent.com/Msmbusiness/Knee/main/data%20midwest.csv"
    }

    default_file = "Data Central Florida"
    selected_file = st.selectbox("Select a CSV file", list(csv_files.keys()), index=list(csv_files.keys()).index(default_file))
    if selected_file:
        file_url = csv_files[selected_file]
        predictor.data = load_data_from_url(file_url)
        st.success(f"Data file '{selected_file}' loaded successfully.")

        if st.button("Train Models"):
            predictor.train_models()

        if 'models' in st.session_state:
            predictor.models = st.session_state['models']
            age = st.slider("Age:", min_value=55, max_value=85, value=65)
            height = st.slider("Height (inches):", min_value=60, max_value=76, value=65)
            sex = st.selectbox("Sex:", ["Female", "Male"])
            sex_val = 0 if sex == "Female" else 1

            if st.button("Predict"):
                model_type = st.selectbox("Select Model", ["xgb", "gbr", "ridge", "stack"])
                predictor.predict(age, height, sex_val, model_type)

            if st.button("Display Metrics"):
                predictor.display_metrics()

    st.write("""
        **Disclaimer:** This application is for educational purposes only. It is not intended to diagnose, provide medical advice, or offer recommendations. The predictions made by this application are not validated and should be used for research purposes only.
    """)

if __name__ == "__main__":
    main()


