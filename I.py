import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, probplot
from sklearn.linear_model import LinearRegression
import requests
from io import StringIO

# Streamlit app title
st.title("Total Knee Size Predictor")

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

    scaler_femur = StandardScaler().fit(X)
    X_scaled_femur = scaler_femur.transform(X)
    femur_xgb = XGBRegressor(n_estimators=100, random_state=1)
    femur_gbr = GradientBoostingRegressor(n_estimators=100, random_state=1)

    # Fit models
    tibia_xgb.fit(X_scaled_tibia, y_tibia)
    tibia_gbr.fit(X_scaled_tibia, y_tibia)

    femur_xgb.fit(X_scaled_femur, y_femur)
    femur_gbr.fit(X_scaled_femur, y_femur)

    return {
        'tibia': {'xgb': tibia_xgb, 'gbr': tibia_gbr, 'scaler': scaler_tibia},
        'femur': {'xgb': femur_xgb, 'gbr': femur_gbr, 'scaler': scaler_femur}
    }

class TibiaFemurPredictor:
    def __init__(self):
        self.models = None
        self.data = None
        self.prediction_df = None
        self.learning_curve_fig = None
        self.residuals_hist_fig = None
        self.qq_plot_fig = None
        self.height_vs_pred_fig = None

    def train_models(self):
        self.models = train_and_scale_models(self.data, ['height_log', 'age_height_interaction', 'sex'])
        st.session_state['models'] = self.models
        st.success("Models trained and scalers initialized.")

    def plot_learning_curve(self, estimator, title, X, y, ax, color):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))
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
        self.plot_learning_curve(self.models['tibia']['gbr'], "Tibia GBR", X, y_tibia, axes[0], 'red')
        self.plot_learning_curve(self.models['femur']['xgb'], "Femur XGB", X, y_femur, axes[1], 'blue')
        self.plot_learning_curve(self.models['femur']['gbr'], "Femur GBR", X, y_femur, axes[1], 'red')

        self.learning_curve_fig = fig
        st.pyplot(fig)

    
    def predict(self, age, height, sex_val):
        if not self.models:
            st.error("Models are not trained yet.")
            return

        X_new = np.array([[np.log1p(height), age * height, sex_val]])
        tibia_scaler = self.models['tibia']['scaler']
        femur_scaler = self.models['femur']['scaler']
        X_new_scaled_tibia = tibia_scaler.transform(X_new)
        X_new_scaled_femur = femur_scaler.transform(X_new)

        tibia_preds_xgb = self.models['tibia']['xgb'].predict(X_new_scaled_tibia)
        tibia_preds_gbr = self.models['tibia']['gbr'].predict(X_new_scaled_tibia)
        femur_preds_xgb = self.models['femur']['xgb'].predict(X_new_scaled_femur)
        femur_preds_gbr = self.models['femur']['gbr'].predict(X_new_scaled_femur)

        prediction_data = {
            "Model": ["XGB", "GBR"],
            "Predicted Femur": [round(femur_preds_xgb[0], 1), round(femur_preds_gbr[0], 1)],
            "Predicted Tibia": [round(tibia_preds_xgb[0], 1), round(tibia_preds_gbr[0], 1)]
        }

        prediction_df = pd.DataFrame(prediction_data)
        self.prediction_df = prediction_df

        st.table(prediction_df)

        avg_femur_pred = round(np.mean([femur_preds_xgb[0], femur_preds_gbr[0]]))
        femur_df = pd.DataFrame(femur_sizes).T
        femur_df.columns = ["A", "B"]
        femur_df.index.name = "Size"
        femur_df.index = femur_df.index.astype(int)
        femur_df = femur_df.reset_index()

        def highlight_row(s):
            return ['background-color: yellow' if s['Size'] == avg_femur_pred else '' for _ in s.index]

        st.table(femur_df.style.apply(highlight_row, axis=1))
    def plot_height_vs_bone_size(self):
        heights = np.arange(60, 76, 1)
        tibia_preds_male = {'xgb': [], 'gbr': []}
        femur_preds_male = {'xgb': [], 'gbr': []}
        tibia_preds_female = {'xgb': [], 'gbr': []}
        femur_preds_female = {'xgb': [], 'gbr': []}

        for height in heights:
            for sex_val, tibia_preds, femur_preds in zip([1, 0], [tibia_preds_male, tibia_preds_female], [femur_preds_male, femur_preds_female]):
                X_new = np.array([[np.log1p(height), self.data['age'].mean() * height, sex_val]])
                X_new_scaled_tibia = self.models['tibia']['scaler'].transform(X_new)
                X_new_scaled_femur = self.models['femur']['scaler'].transform(X_new)

                tibia_preds['xgb'].append(self.models['tibia']['xgb'].predict(X_new_scaled_tibia)[0])
                tibia_preds['gbr'].append(self.models['tibia']['gbr'].predict(X_new_scaled_tibia)[0])
                femur_preds['xgb'].append(self.models['femur']['xgb'].predict(X_new_scaled_femur)[0])
                femur_preds['gbr'].append(self.models['femur']['gbr'].predict(X_new_scaled_femur)[0])

        fig, ax = plt.subplots(figsize=(10, 6))
        show_tibia_xgb = st.checkbox('Tibia XGB', value=True)
        show_tibia_gbr = st.checkbox('Tibia GBR', value=True)
        show_femur_xgb = st.checkbox('Femur XGB', value=True)
        show_femur_gbr = st.checkbox('Femur GBR', value=True)

        if show_tibia_xgb:
            ax.plot(heights, tibia_preds_male['xgb'], label='Tibia XGB (Males)', color='blue', linestyle='--')
            ax.plot(heights, tibia_preds_female['xgb'], label='Tibia XGB (Females)', color='blue')
        if show_tibia_gbr:
            ax.plot(heights, tibia_preds_male['gbr'], label='Tibia GBR (Males)', color='green', linestyle='--')
            ax.plot(heights, tibia_preds_female['gbr'], label='Tibia GBR (Females)', color='green')
        if show_femur_xgb:
            ax.plot(heights, femur_preds_male['xgb'], label='Femur XGB (Males)', color='red', linestyle='--')
            ax.plot(heights, femur_preds_female['xgb'], label='Femur XGB (Females)', color='red')
        if show_femur_gbr:
            ax.plot(heights, femur_preds_male['gbr'], label='Femur GBR (Males)', color='purple', linestyle='--')
            ax.plot(heights, femur_preds_female['gbr'], label='Femur GBR (Females)', color='purple')

        ax.set_xlabel('Height (inches)')
        ax.set_ylabel('Predicted Size')
        ax.set_title('Height vs Predicted Size (Males and Females)')
        ax.legend()
        self.height_vs_pred_fig = fig
        st.pyplot(fig)

    def plot_odds_ratio(self):
        heights = np.arange(64, 76, 1)
        odds_ratios = []

        for height in heights:
            X_new = np.array([[np.log1p(height), self.data['age'].mean() * height, 1]])  # Males only
            X_new_scaled_tibia = self.models['tibia']['scaler'].transform(X_new)
            X_new_scaled_femur = self.models['femur']['scaler'].transform(X_new)

            tibia_pred = self.models['tibia']['gbr'].predict(X_new_scaled_tibia)[0]
            femur_pred = self.models['femur']['gbr'].predict(X_new_scaled_femur)[0]
            odds_ratio = femur_pred - tibia_pred
            odds_ratios.append(odds_ratio)

        heights = np.array(heights).reshape(-1, 1)
        regression_line = LinearRegression().fit(heights, odds_ratios).predict(heights)

        plt.figure(figsize=(10, 6))
        plt.plot(heights, odds_ratios, 'o', label='Difference (Femur Used - Tibia Used)', color='green')
        plt.plot(heights, regression_line, '-', label='Regression Line', color='red')
        plt.xlabel('Height (inches)')
        plt.ylabel('Difference')
        plt.title('Height vs Difference in Femur Used and Tibia Used (Males Only)')
        plt.legend()
        st.pyplot(plt)

    def evaluate_models(self):
        features = ['height_log', 'age_height_interaction', 'sex']
        X_tibia = self.data[features].values
        y_tibia = self.data['tibia used'].values
        X_femur = self.data[features].values
        y_femur = self.data['femur used'].values

        X_tibia_scaled = self.models['tibia']['scaler'].transform(X_tibia)
        X_femur_scaled = self.models['femur']['scaler'].transform(X_femur)

        metrics = {
            'r2_score': [],
            'adjusted_r2_score': [],
            'mae': [],
            'mape': [],
            'skewness': [],
            'kurtosis': []
        }

        model_types = ['xgb', 'gbr']
        bones = ['tibia', 'femur']

        for model_type in model_types:
            for bone in bones:
                preds = self.models[bone][model_type].predict(self.models[bone]['scaler'].transform(self.data[features].values))
                y_true = self.data[f'{bone} used'].values
                metrics['r2_score'].append(r2_score(y_true, preds))
                metrics['adjusted_r2_score'].append(1 - (1 - r2_score(y_true, preds)) * (len(y_true) - 1) / (len(y_true) - X_tibia.shape[1] - 1))
                metrics['mae'].append(mean_absolute_error(y_true, preds))
                metrics['mape'].append(np.mean(np.abs((y_true - preds) / y_true)) * 100)
                metrics['skewness'].append(skew(y_true - preds))
                metrics['kurtosis'].append(kurtosis(y_true - preds))

        metrics_df = pd.DataFrame(metrics, index=pd.MultiIndex.from_product([bones, model_types], names=['Bone', 'Model']))
        st.table(metrics_df.T.applymap(lambda x: round(x, 4)))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, (bone, ax) in enumerate(zip(bones, axes.flatten())):
            for j, model_type in enumerate(model_types):
                residuals = self.data[f'{bone} used'] - self.models[bone][model_type].predict(self.models[bone]['scaler'].transform(self.data[features].values))
                ax.hist(residuals, bins=20, alpha=0.5, label=f'{bone.capitalize()} {model_type.upper()}')
                ax.set_title(f'{bone.capitalize()} Residuals Histogram')
                ax.legend()

        self.residuals_hist_fig = fig
        st.pyplot(fig)

    def plot_qq_plots(self):
        features = ['height_log', 'age_height_interaction', 'sex']
        X_tibia = self.data[features].values
        y_tibia = self.data['tibia used'].values
        X_femur = self.data[features].values
        y_femur = self.data['femur used'].values

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('XGB QQ Plots')

        for i, (bone, ax) in enumerate(zip(['tibia', 'femur'], axes.flatten())):
            residuals = self.data[f'{bone} used'] - self.models[bone]['xgb'].predict(self.models[bone]['scaler'].transform(self.data[features].values))
            probplot(residuals, dist="norm", plot=ax)
            ax.set_title(f'{bone.capitalize()} XGB QQ Plot')

        self.qq_plot_fig = fig
        st.pyplot(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('GBR QQ Plots')

        for i, (bone, ax) in enumerate(zip(['tibia', 'femur'], axes.flatten())):
            residuals = self.data[f'{bone} used'] - self.models[bone]['gbr'].predict(self.models[bone]['scaler'].transform(self.data[features].values))
            probplot(residuals, dist="norm", plot=ax)
            ax.set_title(f'{bone.capitalize()} GBR QQ Plot')

        self.qq_plot_fig = fig
        st.pyplot(fig)


    def fit_regression_line(self, sex_val, age):
        heights = np.linspace(60, 76, 100)
        femur_pred_xgb = []
        femur_pred_gbr = []

        for height in heights:
            X_new = np.array([[np.log1p(height), age * height, sex_val]])
            X_new_scaled_femur = self.models['femur']['scaler'].transform(X_new)

            femur_pred_xgb.append(self.models['femur']['xgb'].predict(X_new_scaled_femur)[0])
            femur_pred_gbr.append(self.models['femur']['gbr'].predict(X_new_scaled_femur)[0])

        model_type = st.selectbox("Select Model for Regression Line:", ["xgb", "gbr"])
        if model_type == "xgb":
            femur_preds = femur_pred_xgb
        else:
            femur_preds = femur_pred_gbr

        femur_reg = LinearRegression().fit(heights.reshape(-1, 1), femur_preds)
        femur_reg_preds = femur_reg.predict(heights.reshape(-1, 1))

        femur_mae = mean_absolute_error(femur_preds, femur_reg_preds)

        st.write(f"Femur {model_type.upper()} MAE (Regression Line): {femur_mae:.4f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(heights, femur_preds, 'o', label=f'Femur {model_type.upper()} Predictions', color='blue')
        ax.plot(heights, femur_reg_preds, '-', label=f'Femur {model_type.upper()} Regression Line', color='blue')

        ax.set_xlabel('Height (inches)')
        ax.set_ylabel('Femur Size')
        ax.set_title(f'Regression Line Fit to Femur Size Predictions ({model_type.upper()})')
        ax.legend()
        st.pyplot(fig)

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
            height = st.slider("Height (inches):", min_value=59, max_value=75, value=65)
            sex = st.selectbox("Sex:", ["Male", "Female"])
            sex_val = 1 if sex == "Male" else 0

            if st.button("Predict"):
                predictor.predict(age, height, sex_val)

            if st.checkbox("Plot Height vs Bone Size (Males and Females)"):
                predictor.plot_height_vs_bone_size()

            if st.checkbox("Plot Difference (Femur Used - Tibia Used) (Males Only)"):
                predictor.plot_odds_ratio()

            if st.checkbox("Training CV Validation Plots"):
                predictor.plot_superimposed_learning_curves()

            if st.checkbox("Evaluate Models"):
                predictor.evaluate_models()

            if st.checkbox("Plot QQ Plots"):
                predictor.plot_qq_plots()

            if st.checkbox("Fit Models with Regression Line"):
                predictor.fit_regression_line(sex_val, age)

    st.write("""
        **Disclaimer:** This application is for educational purposes only. It is not intended to diagnose, provide medical advice, or offer recommendations. The predictions made by this application are not validated and should be used for research purposes only.
    """)

if __name__ == "__main__":
    main()
