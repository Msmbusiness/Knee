import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from imblearn.over_sampling import RandomOverSampler
import streamlit as st

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*test scores are non-finite.*", category=UserWarning, module='sklearn.model_selection._search')

# Femur size dictionary
femur_size_dict = {
    1: {"AP": 55.6, "ML": 59.0},
    2: {"AP": 58.3, "ML": 62.0},
    3: {"AP": 60.8, "ML": 65.0},
    4: {"AP": 63.8, "ML": 68.0},
    5: {"AP": 66.4, "ML": 71.0},
    6: {"AP": 69.3, "ML": 74.0},
    7: {"AP": 72.2, "ML": 77.0},
    8: {"AP": 75.3, "ML": 80.0}
}

class TibiaFemurPredictor:
    def __init__(self):
        self.models = {
            'tibia': {'xgb': None, 'gbr': None, 'scaler': None},
            'femur': {'xgb': None, 'gbr': None, 'scaler': None}
        }
        self.data_file_path = None
        self.data = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data.columns = self.data.columns.str.strip()
        self.data['age_height_interaction'] = self.data['age'] * self.data['height']
        self.data['height_log'] = np.log1p(self.data['height'])
        st.write("Data loaded successfully:", self.data.head())

    def oversample_minority_group(self):
        if 0 in self.data['sex'].values and 1 in self.data['sex'].values:
            X = self.data.drop('sex', axis=1)
            y = self.data['sex']

            ros = RandomOverSampler(random_state=1)
            X_resampled, y_resampled = ros.fit_resample(X, y)

            self.data = pd.concat([X_resampled, y_resampled], axis=1)
            st.write("Data after oversampling:", self.data.head())
        else:
            st.warning("Both male and female samples are required for oversampling.")

    def train_and_scale_models(self, df, features):
        if len(df) < 2:
            st.warning("Not enough data to train models.")
            return

        X = df[features].values
        y_tibia = df['tibia used'].values
        y_femur = df['femur used'].values

        scaler_tibia = StandardScaler().fit(X)
        X_scaled_tibia = scaler_tibia.transform(X)
        self.models['tibia']['scaler'] = scaler_tibia

        self.models['tibia']['xgb'] = XGBRegressor(n_estimators=100, random_state=1).fit(X_scaled_tibia, y_tibia)
        self.models['tibia']['gbr'] = GradientBoostingRegressor(n_estimators=100, random_state=1).fit(X_scaled_tibia, y_tibia)

        scaler_femur = StandardScaler().fit(X)
        X_scaled_femur = scaler_femur.transform(X)
        self.models['femur']['scaler'] = scaler_femur

        self.models['femur']['xgb'] = XGBRegressor(n_estimators=100, random_state=1).fit(X_scaled_femur, y_femur)
        self.models['femur']['gbr'] = GradientBoostingRegressor(n_estimators=100, random_state=1).fit(X_scaled_femur, y_femur)

    def train_models(self):
        self.oversample_minority_group()
        self.train_and_scale_models(self.data, ['height_log', 'age_height_interaction', 'sex'])
        st.write("Models and scalers after training:", self.models)

    def evaluate_models(self, features):
        if not self.models['tibia']['scaler'] or not self.models['femur']['scaler']:
            st.error("Scalers are not initialized. Please train the models first.")
            return

        X_tibia = self.data[features].values
        y_tibia = self.data['tibia used'].values
        X_femur = self.data[features].values
        y_femur = self.data['femur used'].values

        X_tibia_scaled = self.models['tibia']['scaler'].transform(X_tibia)
        X_femur_scaled = self.models['femur']['scaler'].transform(X_femur)

        tibia_metrics_xgb = self.calculate_metrics(X_tibia_scaled, y_tibia, 'tibia', 'xgb')
        tibia_metrics_gbr = self.calculate_metrics(X_tibia_scaled, y_tibia, 'tibia', 'gbr')
        femur_metrics_xgb = self.calculate_metrics(X_femur_scaled, y_femur, 'femur', 'xgb')
        femur_metrics_gbr = self.calculate_metrics(X_femur_scaled, y_femur, 'femur', 'gbr')

        st.write("Tibia Metrics (XGB)", tibia_metrics_xgb)
        st.write("Tibia Metrics (GBR)", tibia_metrics_gbr)
        st.write("Femur Metrics (XGB)", femur_metrics_xgb)
        st.write("Femur Metrics (GBR)", femur_metrics_gbr)

        # Plot learning curves
        self.plot_superimposed_learning_curves(self.models['tibia']['xgb'], self.models['tibia']['gbr'], "Tibia", X_tibia_scaled, y_tibia)
        self.plot_superimposed_learning_curves(self.models['femur']['xgb'], self.models['femur']['gbr'], "Femur", X_femur_scaled, y_femur)

    def calculate_metrics(self, X, y, bone, model_type):
        model = self.models[bone][model_type]
        preds = model.predict(X)

        mae = mean_absolute_error(y, preds)

        metrics = {
            'model': model_type,
            'r2_score': r2_score(y, preds),
            'mae': mae,
        }

        return metrics

    def plot_superimposed_learning_curves(self, model1, model2, title, X, y, ylim=None, cv=5, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(f"{title} Learning Curves")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        
        train_sizes, train_scores_1, test_scores_1 = learning_curve(model1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_sizes, train_scores_2, test_scores_2 = learning_curve(model2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

        train_scores_mean_1 = np.mean(train_scores_1, axis=1)
        train_scores_std_1 = np.std(train_scores_1, axis=1)
        test_scores_mean_1 = np.mean(test_scores_1, axis=1)
        test_scores_std_1 = np.std(test_scores_1, axis=1)

        train_scores_mean_2 = np.mean(train_scores_2, axis=1)
        train_scores_std_2 = np.std(train_scores_2, axis=1)
        test_scores_mean_2 = np.mean(test_scores_2, axis=1)
        test_scores_std_2 = np.std(test_scores_2, axis=1)

        plt.fill_between(train_sizes, train_scores_mean_1 - train_scores_std_1,
                         train_scores_mean_1 + train_scores_std_1, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean_1 - test_scores_std_1,
                         test_scores_mean_1 + test_scores_std_1, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean_1, 'o-', color="r", label=f"Training score {title} XGB")
        plt.plot(train_sizes, test_scores_mean_1, 'o-', color="g", label=f"Cross-validation score {title} XGB")

        plt.fill_between(train_sizes, train_scores_mean_2 - train_scores_std_2,
                         train_scores_mean_2 + train_scores_std_2, alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean_2 - train_scores_std_2,
                         test_scores_mean_2 + test_scores_std_2, alpha=0.1, color="y")
        plt.plot(train_sizes, train_scores_mean_2, 'o-', color="b", label=f"Training score {title} GBR")
        plt.plot(train_sizes, test_scores_mean_2, 'o-', color="y", label=f"Cross-validation score {title} GBR")

        plt.legend(loc="best")
        st.pyplot(plt)

    def predict(self, age, height, sex, model_type):
        if self.models['tibia']['scaler'] is None or self.models['femur']['scaler'] is None:
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

        st.write(f"Predicted Optimotion Tibia Used with XGB: {preds_tibia_xgb[0]:.1f}")
        st.write(f"Predicted Optimotion Tibia Used with GBR: {preds_tibia_gbr[0]:.1f}")
        st.write(f"Predicted Optimotion Femur Used with XGB: {preds_femur_xgb[0]:.1f}")
        st.write(f"Predicted Optimotion Femur Used with GBR: {preds_femur_gbr[0]:.1f}")

        femur_size = int(np.round(preds_femur_gbr[0]))
        if femur_size > 8.5:
            st.error("Warning: Predicted femur size is greater than 8", icon="🚨")

        femur_df = pd.DataFrame([
            {"Size": size, "Anterior-Posterior (A)": details["AP"], "Medial-Lateral (B)": details["ML"]}
            for size, details in femur_size_dict.items()
        ])

        def highlight_row(s):
            return ['background-color: yellow' if s['Size'] == femur_size else '' for _ in s]

        st.dataframe(femur_df.style.apply(highlight_row, axis=1).format({
            "Anterior-Posterior (A)": "{:.1f}",
            "Medial-Lateral (B)": "{:.1f}"
        }))

def main():
    # Check if the predictor is already in session_state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = TibiaFemurPredictor()

    predictor = st.session_state.predictor

    st.title("Tibia Femur Predictor")

    data_file = st.file_uploader("Upload Data File", type=["csv"])
    if data_file is not None:
        predictor.load_data(data_file)
        st.success("Data file loaded successfully.")

    if st.button("Train Models"):
        predictor.train_models()
        st.success("Models trained successfully.")

    if st.button("Evaluate Models"):
        predictor.evaluate_models(['height_log', 'age_height_interaction', 'sex'])

    age = st.number_input("Age:", min_value=0, max_value=120)
    height = st.number_input("Height (inches):", min_value=0.0)
    sex = st.selectbox("Sex:", ["Female", "Male"])
    sex_val = 0 if sex == "Female" else 1
    model_type = st.selectbox("Model:", ["xgb", "gbr"])

    if st.button("Predict"):
        predictor.predict(age, height, sex_val, model_type)

    st.markdown("---")
    st.markdown(
        "### Disclaimer\n"
        "This tool is intended for educational purposes only. It is not intended to diagnose, provide medical advice, or make treatment recommendations. "
        "The predictions made by this tool should not be used as a substitute for professional medical judgment. Use this tool to contribute to further research and understanding. "
        "Predictions are subject to validation and should be interpreted with caution."
    )

if __name__ == "__main__":
    main()
