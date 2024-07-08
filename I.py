import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*test scores are non-finite.*", category=UserWarning, module='sklearn.model_selection._search')

# Streamlit app title
st.title("Tibia Femur Predictor")

# Dictionary for femur sizes
femur_sizes = {
    1: (55.6, 59),
    2: (58.3, 62),
    3: (60.8, 65),
    4: (63.8, 68),
    5: (66.4, 71),
    6: (69.3, 74),
    7: (72.2, 77),
    8: (75.3, 80)
}

@st.cache_data(show_spinner=False)
def load_data(file):
    data = pd.read_csv(file)
    data.columns = data.columns.str.strip()
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
    tibia_xgb = XGBRegressor(n_estimators=100, random_state=1).fit(X_scaled_tibia, y_tibia)
    tibia_gbr = GradientBoostingRegressor(n_estimators=100, random_state=1).fit(X_scaled_tibia, y_tibia)

    scaler_femur = StandardScaler().fit(X)
    X_scaled_femur = scaler_femur.transform(X)
    femur_xgb = XGBRegressor(n_estimators=100, random_state=1).fit(X_scaled_femur, y_femur)
    femur_gbr = GradientBoostingRegressor(n_estimators=100, random_state=1).fit(X_scaled_femur, y_femur)

    return {
        'tibia': {'xgb': tibia_xgb, 'gbr': tibia_gbr, 'scaler': scaler_tibia},
        'femur': {'xgb': femur_xgb, 'gbr': femur_gbr, 'scaler': scaler_femur}
    }

class TibiaFemurPredictor:
    def __init__(self):
        self.models = None
        self.data = None

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

    def plot_learning_curve(self, estimator, title, X, y, ax, color):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        ax.plot(train_sizes, train_scores_mean, 'o-', color=color, label=f"Training score {title}")
        ax.plot(train_sizes, test_scores_mean, 'o-', color=color, label=f"Cross-validation score {title}")
        ax.set_title(title)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.legend(loc="best")
        ax.grid()

    def plot_superimposed_learning_curves(self):
        X = self.data[['height_log', 'age_height_interaction', 'sex']].values
        y_tibia = self.data['tibia used'].values
        y_femur = self.data['femur used'].values

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        self.plot_learning_curve(self.models['tibia']['xgb'], "Tibia XGB", X, y_tibia, axes[0], 'blue')
        self.plot_learning_curve(self.models['tibia']['gbr'], "Tibia GBR", X, y_tibia, axes[0], 'green')
        self.plot_learning_curve(self.models['femur']['xgb'], "Femur XGB", X, y_femur, axes[1], 'blue')
        self.plot_learning_curve(self.models['femur']['gbr'], "Femur GBR", X, y_femur, axes[1], 'green')

        st.pyplot(fig)

    def predict(self, age, height, sex, model_type):
        if not self.models:
            st.error("Models are not trained yet.")
            return

        X_new = np.array([[np.log1p(height), age * height, sex]])
        tibia_scaler = self.models['tibia']['scaler']
        femur_scaler = self.models['femur']['scaler']
        X_new_scaled_tibia = tibia_scaler.transform(X_new)
        X_new_scaled_femur = femur_scaler.transform(X_new)

        preds_tibia_xgb = self.models['tibia']['xgb'].predict(X_new_scaled_tibia)
        preds_tibia_gbr = self.models['tibia']['gbr'].predict(X_new_scaled_tibia)
        preds_femur_xgb = self.models['femur']['xgb'].predict(X_new_scaled_femur)
        preds_femur_gbr = self.models['femur']['gbr'].predict(X_new_scaled_femur)

        predicted_tibia_xgb = round(preds_tibia_xgb[0], 1)
        predicted_tibia_gbr = round(preds_tibia_gbr[0], 1)
        predicted_femur_xgb = round(preds_femur_xgb[0], 1)
        predicted_femur_gbr = round(preds_femur_gbr[0], 1)

        st.write(f"Predicted Optimotion Tibia Used with XGB: {predicted_tibia_xgb}")
        st.write(f"Predicted Optimotion Tibia Used with GBR: {predicted_tibia_gbr}")
        st.write(f"Predicted Optimotion Femur Used with XGB: {predicted_femur_xgb}")
        st.write(f"Predicted Optimotion Femur Used with GBR: {predicted_femur_gbr}")

        if model_type == "xgb" and predicted_femur_xgb > 8.5:
            st.error("Predict size 9 femur", icon="🚨")

        femur_df = pd.DataFrame(femur_sizes).T
        femur_df.columns = ["A", "B"]
        femur_df = femur_df.round(1)
        femur_df.index.name = "Size"
        femur_df.index = femur_df.index.astype(int)
        femur_df = femur_df.reset_index()

        if predicted_femur_gbr <= 8.8:
            femur_size = min(max(predicted_femur_gbr, 1), 8)  # Clamp between 1 and 8

            def highlight_row(s):
                return ['background-color: yellow' if s['Size'] == femur_size else '' for _ in s]

            st.table(femur_df.style.apply(highlight_row, axis=1))
        else:
            st.table(femur_df)

def main():
    predictor = TibiaFemurPredictor()

    uploaded_file = st.file_uploader("Upload Data File", type="csv")
    if uploaded_file is not None:
        predictor.data = load_data(uploaded_file)
        st.success("Data file loaded successfully.")

        if st.button("Train Models"):
            predictor.train_models()

        if 'models' in st.session_state:
            predictor.models = st.session_state['models']
            age = st.number_input("Age:", min_value=0, max_value=120, value=30)
            height = st.number_input("Height (inches):", min_value=0.0, max_value=100.0, value=65.0)
            sex = st.selectbox("Sex:", ["Female", "Male"])
            sex_val = 0 if sex == "Female" else 1
            model_type = st.selectbox("Model:", ["xgb", "gbr"])

            if st.button("Predict"):
                predictor.predict(age, height, sex_val, model_type)

            if st.button("Evaluate Models"):
                predictor.plot_superimposed_learning_curves()

    st.write("""
        **Disclaimer:** This application is for educational purposes only. It is not intended to diagnose, provide medical advice, or offer recommendations. The predictions made by this application are not validated and should be used for research purposes only.
    """)

if __name__ == "__main__":
    main()


