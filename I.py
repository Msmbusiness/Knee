import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from bayes_opt import BayesianOptimization
import requests
from io import StringIO
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_squared_error

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*test scores are non-finite.*", category=UserWarning, module='sklearn.model_selection._search')

# Dictionary for femur sizes
femur_sizes = {
    1: (55.6, 59),
    2: (58.3, 62),
    3: (60.8, 65),
    4: (63.8, 68),
    5: (66.4, 71),
    6: (69.3, 74),
    7: (72.2, 77),
    8: (75.3, 80),
    9: (78.3, 83)
}

@st.cache_data
def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text))
    data.columns = data.columns.str.strip()
    data['age_height_interaction'] = data['age'] * data['height']
    data['height_log'] = np.log1p(data['height'])
    return data

@st.cache_resource
def train_and_scale_models(data, features):
    def bayesian_optimization_xgb(X, y):
        def xgb_evaluate(n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda):
            model = XGBRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=1
            )
            model.fit(X, y)
            return -mean_squared_error(y, model.predict(X))

        xgb_bo = BayesianOptimization(
            f=xgb_evaluate,
            pbounds={
                'n_estimators': (50, 200),
                'max_depth': (3, 7),
                'learning_rate': (0.01, 0.2),
                'reg_alpha': (0, 1),
                'reg_lambda': (0, 1)
            },
            random_state=1
        )
        xgb_bo.maximize(init_points=5, n_iter=25)
        return xgb_bo.max['params']

    def bayesian_optimization_gbr(X, y):
        def gbr_evaluate(n_estimators, max_depth, learning_rate, alpha):
            model = GradientBoostingRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                alpha=alpha,
                random_state=1
            )
            model.fit(X, y)
            return -mean_squared_error(y, model.predict(X))

        gbr_bo = BayesianOptimization(
            f=gbr_evaluate,
            pbounds={
                'n_estimators': (50, 200),
                'max_depth': (3, 7),
                'learning_rate': (0.01, 0.2),
                'alpha': (1e-6, 0.99)
            },
            random_state=1
        )
        gbr_bo.maximize(init_points=5, n_iter=25)
        return gbr_bo.max['params']

    st.write("Training models...")
    X = data[features].values
    y_tibia = data['tibia used'].values
    y_femur = data['femur used'].values

    scaler_tibia = StandardScaler().fit(X)
    X_scaled_tibia = scaler_tibia.transform(X)
    scaler_femur = StandardScaler().fit(X)
    X_scaled_femur = scaler_femur.transform(X)

    xgb_params_tibia = bayesian_optimization_xgb(X_scaled_tibia, y_tibia)
    gbr_params_tibia = bayesian_optimization_gbr(X_scaled_tibia, y_tibia)
    xgb_params_femur = bayesian_optimization_xgb(X_scaled_femur, y_femur)
    gbr_params_femur = bayesian_optimization_gbr(X_scaled_femur, y_femur)

    xgb_params_tibia['n_estimators'] = int(xgb_params_tibia['n_estimators'])
    xgb_params_tibia['max_depth'] = int(xgb_params_tibia['max_depth'])
    gbr_params_tibia['n_estimators'] = int(gbr_params_tibia['n_estimators'])
    gbr_params_tibia['max_depth'] = int(gbr_params_tibia['max_depth'])
    xgb_params_femur['n_estimators'] = int(xgb_params_femur['n_estimators'])
    xgb_params_femur['max_depth'] = int(xgb_params_femur['max_depth'])
    gbr_params_femur['n_estimators'] = int(gbr_params_femur['n_estimators'])
    gbr_params_femur['max_depth'] = int(gbr_params_femur['max_depth'])

    tibia_xgb = XGBRegressor(**xgb_params_tibia, random_state=1)
    tibia_gbr = GradientBoostingRegressor(**gbr_params_tibia, random_state=1)
    femur_xgb = XGBRegressor(**xgb_params_femur, random_state=1)
    femur_gbr = GradientBoostingRegressor(**gbr_params_femur, random_state=1)

    tibia_stack = StackingRegressor(estimators=[('xgb', tibia_xgb), ('gbr', tibia_gbr)], final_estimator=XGBRegressor(), cv=5)
    femur_stack = StackingRegressor(estimators=[('xgb', femur_xgb), ('gbr', femur_gbr)], final_estimator=XGBRegressor(), cv=5)

    tibia_xgb.fit(X_scaled_tibia, y_tibia)
    tibia_gbr.fit(X_scaled_tibia, y_tibia)
    tibia_stack.fit(X_scaled_tibia, y_tibia)

    femur_xgb.fit(X_scaled_femur, y_femur)
    femur_gbr.fit(X_scaled_femur, y_femur)
    femur_stack.fit(X_scaled_femur, y_femur)

    return {
        'tibia': {'xgb': tibia_xgb, 'gbr': tibia_gbr, 'stack': tibia_stack, 'scaler': scaler_tibia},
        'femur': {'xgb': femur_xgb, 'gbr': femur_gbr, 'stack': femur_stack, 'scaler': scaler_femur}
    }

class TibiaFemurPredictor:
    def __init__(self):
        self.models = None
        self.data = None
        self.prediction_df = pd.DataFrame({
            "Model": ["XGB", "GBR", "Stack", "Average", "Preferred"],
            "Predicted Femur": [np.nan]*5,
            "Predicted Tibia": [np.nan]*5
        })

    def oversample_minority_group(self, data):
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

    def predict(self, age, height, sex_val):
        if 'models' not in st.session_state or st.session_state['models'] is None:
            st.warning("Models are not trained yet.")
            return

        self.models = st.session_state['models']
        X_new = np.array([[np.log1p(height), age * height, sex_val]])
        tibia_scaler = self.models['tibia']['scaler']
        femur_scaler = self.models['femur']['scaler']
        X_new_scaled_tibia = tibia_scaler.transform(X_new)
        X_new_scaled_femur = femur_scaler.transform(X_new)

        preds_tibia_xgb = self.models['tibia']['xgb'].predict(X_new_scaled_tibia)
        preds_tibia_gbr = self.models['tibia']['gbr'].predict(X_new_scaled_tibia)
        preds_tibia_stack = self.models['tibia']['stack'].predict(X_new_scaled_tibia)
        preds_femur_xgb = self.models['femur']['xgb'].predict(X_new_scaled_femur)
        preds_femur_gbr = self.models['femur']['gbr'].predict(X_new_scaled_femur)
        preds_femur_stack = self.models['femur']['stack'].predict(X_new_scaled_femur)

        tibia_avg = round((preds_tibia_xgb[0] + preds_tibia_gbr[0] + preds_tibia_stack[0]) / 3, 1)
        femur_avg = round((preds_femur_xgb[0] + preds_femur_gbr[0] + preds_femur_stack[0]) / 3, 1)

        # Update the session state with new predictions
        st.session_state['preferred_femur'] = round(femur_avg)
        st.session_state['preferred_tibia'] = round(tibia_avg)

        self.prediction_df["Predicted Femur"] = [
            round(preds_femur_xgb[0], 1),
            round(preds_femur_gbr[0], 1),
            round(preds_femur_stack[0], 1),
            round(femur_avg, 1),
            round(femur_avg, 1)
        ]
        self.prediction_df["Predicted Tibia"] = [
            round(preds_tibia_xgb[0], 1),
            round(preds_tibia_gbr[0], 1),
            round(preds_tibia_stack[0], 1),
            round(tibia_avg, 1),
            round(tibia_avg, 1)
        ]

        st.table(self.prediction_df)

        femur_df = pd.DataFrame(femur_sizes).T
        femur_df.columns = ["A", "B"]
        femur_df.index.name = "Femur Size"
        femur_df.index = femur_df.index.astype(int)
        femur_df = femur_df.reset_index()

        tibia_df = pd.DataFrame({
            'Tibial size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'Anterior-Posterior': [40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 56.0, 60.0, 64.0],
            'Medial-Lateral': [61.0, 64.0, 67.0, 70.0, 73.0, 76.0, 81.0, 86.0, 90.0]
        })
        tibia_df.set_index('Tibial size', inplace=True)
        tibia_df = tibia_df.reset_index()

        # Combine the femur and tibia dataframes
        combined_df = pd.merge(femur_df, tibia_df, left_on='Femur Size', right_on='Tibial size', how='outer')
        combined_df = combined_df.rename(columns={
            'A': 'Femur Size A',
            'B': 'Femur Size B',
            'Tibial size': 'Tibial Size',
            'Anterior-Posterior': 'Tibial Anterior-Posterior',
            'Medial-Lateral': 'Tibial Medial-Lateral'
        })

        return combined_df.round(1)

def main():
    st.title("Total Knee Implant Size Predictor")
    st.markdown("<h6 style='text-align: center;'>Michael Messieh MD</h6>", unsafe_allow_html=True)

    predictor = TibiaFemurPredictor()

    csv_files = {
        "Data Central Florida": "https://raw.githubusercontent.com/Msmbusiness/Knee/main/data%20central%20florida.csv",
        "Data Midwest": "https://raw.githubusercontent.com/Msmbusiness/Knee/main/data%20midwest.csv"
    }

    selected_file = st.selectbox("Select file", list(csv_files.keys()))

    if selected_file:
        file_url = csv_files[selected_file]
        predictor.data = load_data_from_url(file_url)
        st.success(f"Data file '{selected_file}' loaded successfully.")
        if st.button("Train Models"):
            predictor.train_models()

    age = st.number_input("Age", min_value=55, max_value=85, value=70)
    height = st.number_input("Height (inches)", min_value=60, max_value=76, value=69)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex_val = 0 if sex == "Female" else 1

    if st.button("Predict"):
        combined_df = predictor.predict(age, height, sex_val)

        def highlight_row(s):
            femur_color = 'background-color: yellow'
            tibia_color = 'background-color: pink'
            styles = [''] * len(s)
            if 'preferred_femur' in st.session_state and s['Femur Size'] == st.session_state['preferred_femur']:
                styles[0] = femur_color  # Assuming 'Femur Size' is the first column
            if 'preferred_tibia' in st.session_state and s['Tibial Size'] == st.session_state['preferred_tibia']:
                styles[3] = tibia_color  # Assuming 'Tibial Size' is the fourth column
            return styles

        st.dataframe(combined_df.style.apply(highlight_row, axis=1))

    st.markdown("""
        **Disclaimer:** This application is for educational purposes only. It is not intended to diagnose, provide medical advice, or offer recommendations. The predictions made by this application are not validated and should be used for research purposes only.
    """)

if __name__ == "__main__":
    main()







