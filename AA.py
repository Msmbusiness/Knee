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
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import kurtosis
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import requests
from io import StringIO
import tempfile
from fpdf import FPDF

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*test scores are non-finite.*", category=UserWarning, module='sklearn.model_selection._search')

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
        self.height_vs_pred_fig = None
        self.learning_curve_fig = None
        self.qq_plot_fig = None

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

        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        self.plot_learning_curve(self.models['tibia']['xgb'], "Tibia XGB", X, y_tibia, axes[0, 0], 'blue')
        self.plot_learning_curve(self.models['tibia']['gbr'], "Tibia GBR", X, y_tibia, axes[0, 1], 'red')
        self.plot_learning_curve(self.models['tibia']['ridge'], "Tibia Ridge", X, y_tibia, axes[1, 0], 'green')
        self.plot_learning_curve(self.models['tibia']['stack'], "Tibia Stack", X, y_tibia, axes[1, 1], 'purple')
        self.plot_learning_curve(self.models['femur']['xgb'], "Femur XGB", X, y_femur, axes[2, 0], 'blue')
        self.plot_learning_curve(self.models['femur']['gbr'], "Femur GBR", X, y_femur, axes[2, 1], 'red')
        self.plot_learning_curve(self.models['femur']['ridge'], "Femur Ridge", X, y_femur, axes[3, 0], 'green')
        self.plot_learning_curve(self.models['femur']['stack'], "Femur Stack", X, y_femur, axes[3, 1], 'purple')

        self.learning_curve_fig = fig
        st.pyplot(fig)

    def predict(self, age, height, sex_val, model_type):
        if not self.models:
            st.error("Models are not trained yet.")
            return

        X_new = np.array([[np.log1p(height), age * height, sex_val]])
        tibia_scaler = self.models['tibia']['scaler']
        femur_scaler = self.models['femur']['scaler']
        X_new_scaled_tibia = tibia_scaler.transform(X_new)
        X_new_scaled_femur = femur_scaler.transform(X_new)

        preds_tibia_xgb = self.models['tibia']['xgb'].predict(X_new_scaled_tibia)
        preds_tibia_gbr = self.models['tibia']['gbr'].predict(X_new_scaled_tibia)
        preds_tibia_ridge = self.models['tibia']['ridge'].predict(X_new_scaled_tibia)
        preds_tibia_stack = self.models['tibia']['stack'].predict(X_new_scaled_tibia)
        preds_femur_xgb = self.models['femur']['xgb'].predict(X_new_scaled_femur)
        preds_femur_gbr = self.models['femur']['gbr'].predict(X_new_scaled_femur)
        preds_femur_ridge = self.models['femur']['ridge'].predict(X_new_scaled_femur)
        preds_femur_stack = self.models['femur']['stack'].predict(X_new_scaled_femur)

        # Linear regression line prediction for GBR model
        heights = np.linspace(60, 76, 100)
        tibia_pred_gbr = [self.models['tibia']['gbr'].predict(tibia_scaler.transform(np.array([[np.log1p(h), age * h, sex_val]])))[0] for h in heights]
        femur_pred_gbr = [self.models['femur']['gbr'].predict(femur_scaler.transform(np.array([[np.log1p(h), age * h, sex_val]])))[0] for h in heights]

        tibia_reg = LinearRegression().fit(heights.reshape(-1, 1), tibia_pred_gbr)
        femur_reg = LinearRegression().fit(heights.reshape(-1, 1), femur_pred_gbr)

        tibia_reg_pred = tibia_reg.predict(np.array([[height]]))[0]
        femur_reg_pred = femur_reg.predict(np.array([[height]]))[0]

        prediction_data = {
            "Model": ["XGB", "GBR", "GBR with Reg Line", "Ridge", "Stack"],
            "Predicted Femur": [round(preds_femur_xgb[0], 1), round(preds_femur_gbr[0], 1), round(femur_reg_pred, 1), round(preds_femur_ridge[0], 1), round(preds_femur_stack[0], 1)],
            "Predicted Tibia": [round(preds_tibia_xgb[0], 1), round(preds_tibia_gbr[0], 1), round(tibia_reg_pred, 1), round(preds_tibia_ridge[0], 1), round(preds_tibia_stack[0], 1)]
        }

        prediction_df = pd.DataFrame(prediction_data)
        self.prediction_df = prediction_df

        st.table(prediction_df)

        # Highlight the row based on the rounded value of the GBR predicted femur size
        femur_df = pd.DataFrame(femur_sizes).T
        femur_df.columns = ["A", "B"]
        femur_df.index.name = "Size"
        femur_df.index = femur_df.index.astype(int)
        femur_df = femur_df.reset_index()

        def highlight_row(s):
            return ['background-color: yellow' if s['Size'] == round(preds_femur_gbr[0]) else '' for _ in s.index]

        st.table(femur_df.style.apply(highlight_row, axis=1))

    def fit_regression_line(self, sex_val, age):
        heights = np.linspace(60, 76, 100)
        tibia_pred_gbr = []
        femur_pred_gbr = []

        for height in heights:
            X_new = np.array([[np.log1p(height), age * height, sex_val]])
            X_new_scaled_tibia = self.models['tibia']['scaler'].transform(X_new)
            X_new_scaled_femur = self.models['femur']['scaler'].transform(X_new)

            tibia_pred_gbr.append(self.models['tibia']['gbr'].predict(X_new_scaled_tibia)[0])
            femur_pred_gbr.append(self.models['femur']['gbr'].predict(X_new_scaled_femur)[0])

        # Fit linear regression lines
        tibia_reg = LinearRegression().fit(heights.reshape(-1, 1), tibia_pred_gbr)
        femur_reg = LinearRegression().fit(heights.reshape(-1, 1), femur_pred_gbr)

        # Calculate metrics and residuals for the linear regression lines
        tibia_reg_preds = tibia_reg.predict(heights.reshape(-1, 1))
        femur_reg_preds = femur_reg.predict(heights.reshape(-1, 1))

        tibia_mae = mean_absolute_error(tibia_pred_gbr, tibia_reg_preds)
        femur_mae = mean_absolute_error(femur_pred_gbr, femur_reg_preds)

        st.write(f"Tibia MAE (Regression Line): {tibia_mae:.4f}")
        st.write(f"Femur MAE (Regression Line): {femur_mae:.4f}")

        fig, ax = plt.subplots()
        ax.plot(heights, tibia_pred_gbr, label='Tibia GBR', color='green')
        ax.plot(heights, tibia_reg_preds, label='Tibia GBR Regression Line', linestyle='--', color='green')
        ax.plot(heights, femur_pred_gbr, label='Femur GBR', color='blue')
        ax.plot(heights, femur_reg_preds, label='Femur GBR Regression Line', linestyle='--', color='blue')

        ax.set_xlabel('Height (inches)')
        ax.set_ylabel('Predicted Size')
        ax.set_title(f'Height vs Predicted Size with Regression Line ({"Female" if sex_val == 0 else "Male"})')
        ax.legend()

        self.height_vs_pred_fig = fig
        st.pyplot(fig)

    def plot_height_vs_predicted_size(self, sex_val, age):
        heights = np.linspace(60, 76, 100)
        tibia_pred_xgb = []
        tibia_pred_gbr = []
        tibia_pred_ridge = []
        tibia_pred_stack = []
        femur_pred_xgb = []
        femur_pred_gbr = []
        femur_pred_ridge = []
        femur_pred_stack = []

        for height in heights:
            X_new = np.array([[np.log1p(height), age * height, sex_val]])
            X_new_scaled_tibia = self.models['tibia']['scaler'].transform(X_new)
            X_new_scaled_femur = self.models['femur']['scaler'].transform(X_new)

            tibia_pred_xgb.append(self.models['tibia']['xgb'].predict(X_new_scaled_tibia)[0])
            tibia_pred_gbr.append(self.models['tibia']['gbr'].predict(X_new_scaled_tibia)[0])
            tibia_pred_ridge.append(self.models['tibia']['ridge'].predict(X_new_scaled_tibia)[0])
            tibia_pred_stack.append(self.models['tibia']['stack'].predict(X_new_scaled_tibia)[0])
            femur_pred_xgb.append(self.models['femur']['xgb'].predict(X_new_scaled_femur)[0])
            femur_pred_gbr.append(self.models['femur']['gbr'].predict(X_new_scaled_femur)[0])
            femur_pred_ridge.append(self.models['femur']['ridge'].predict(X_new_scaled_femur)[0])
            femur_pred_stack.append(self.models['femur']['stack'].predict(X_new_scaled_femur)[0])

        fig, ax = plt.subplots()
        show_tibia_xgb = st.checkbox('Tibia XGB', value=True)
        show_tibia_gbr = st.checkbox('Tibia GBR', value=True)
        show_tibia_ridge = st.checkbox('Tibia Ridge', value=True)
        show_tibia_stack = st.checkbox('Tibia Stack', value=True)
        show_femur_xgb = st.checkbox('Femur XGB', value=True)
        show_femur_gbr = st.checkbox('Femur GBR', value=True)
        show_femur_ridge = st.checkbox('Femur Ridge', value=True)
        show_femur_stack = st.checkbox('Femur Stack', value=True)
        smooth_method = st.selectbox('Select smoothing method:', ('None', 'Spline', 'Moving Average'))

        if smooth_method == 'Spline':
            heights_smooth = np.linspace(heights.min(), heights.max(), 300)
            if show_tibia_xgb:
                tibia_smooth = make_interp_spline(heights, tibia_pred_xgb)(heights_smooth)
                ax.plot(heights_smooth, tibia_smooth, color='blue', label='Tibia XGB')
            if show_tibia_gbr:
                tibia_smooth = make_interp_spline(heights, tibia_pred_gbr)(heights_smooth)
                ax.plot(heights_smooth, tibia_smooth, color='green', label='Tibia GBR')
            if show_tibia_ridge:
                tibia_smooth = make_interp_spline(heights, tibia_pred_ridge)(heights_smooth)
                ax.plot(heights_smooth, tibia_smooth, color='orange', label='Tibia Ridge')
            if show_tibia_stack:
                tibia_smooth = make_interp_spline(heights, tibia_pred_stack)(heights_smooth)
                ax.plot(heights_smooth, tibia_smooth, color='purple', label='Tibia Stack')
            if show_femur_xgb:
                femur_smooth = make_interp_spline(heights, femur_pred_xgb)(heights_smooth)
                ax.plot(heights_smooth, femur_smooth, color='red', label='Femur XGB')
            if show_femur_gbr:
                femur_smooth = make_interp_spline(heights, femur_pred_gbr)(heights_smooth)
                ax.plot(heights_smooth, femur_smooth, color='green', label='Femur GBR')
            if show_femur_ridge:
                femur_smooth = make_interp_spline(heights, femur_pred_ridge)(heights_smooth)
                ax.plot(heights_smooth, femur_smooth, color='orange', label='Femur Ridge')
            if show_femur_stack:
                femur_smooth = make_interp_spline(heights, femur_pred_stack)(heights_smooth)
                ax.plot(heights_smooth, femur_smooth, color='purple', label='Femur Stack')
        elif smooth_method == 'Moving Average':
            window_size = 5
            if show_tibia_xgb:
                tibia_ma = pd.Series(tibia_pred_xgb).rolling(window=window_size).mean().dropna().values
                ax.plot(heights[window_size-1:], tibia_ma, color='blue', label='Tibia XGB (MA)')
            if show_tibia_gbr:
                tibia_ma = pd.Series(tibia_pred_gbr).rolling(window=window_size).mean().dropna().values
                ax.plot(heights[window_size-1:], tibia_ma, color='green', label='Tibia GBR (MA)')
            if show_tibia_ridge:
                tibia_ma = pd.Series(tibia_pred_ridge).rolling(window=window_size).mean().dropna().values
                ax.plot(heights[window_size-1:], tibia_ma, color='orange', label='Tibia Ridge (MA)')
            if show_tibia_stack:
                tibia_ma = pd.Series(tibia_pred_stack).rolling(window=window_size).mean().dropna().values
                ax.plot(heights[window_size-1:], tibia_ma, color='purple', label='Tibia Stack (MA)')
            if show_femur_xgb:
                femur_ma = pd.Series(femur_pred_xgb).rolling(window=window_size).mean().dropna().values
                ax.plot(heights[window_size-1:], femur_ma, color='red', label='Femur XGB (MA)')
            if show_femur_gbr:
                femur_ma = pd.Series(femur_pred_gbr).rolling(window=window_size).mean().dropna().values
                ax.plot(heights[window_size-1:], femur_ma, color='green', label='Femur GBR (MA)')
            if show_femur_ridge:
                femur_ma = pd.Series(femur_pred_ridge).rolling(window=window_size).mean().dropna().values
                ax.plot(heights[window_size-1:], femur_ma, color='orange', label='Femur Ridge (MA)')
            if show_femur_stack:
                femur_ma = pd.Series(femur_pred_stack).rolling(window=window_size).mean().dropna().values
                ax.plot(heights[window_size-1:], femur_ma, color='purple', label='Femur Stack (MA)')
        else:
            if show_tibia_xgb:
                ax.plot(heights, tibia_pred_xgb, color='blue', label='Tibia XGB')
            if show_tibia_gbr:
                ax.plot(heights, tibia_pred_gbr, color='green', label='Tibia GBR')
            if show_tibia_ridge:
                ax.plot(heights, tibia_pred_ridge, color='orange', label='Tibia Ridge')
            if show_tibia_stack:
                ax.plot(heights, tibia_pred_stack, color='purple', label='Tibia Stack')
            if show_femur_xgb:
                ax.plot(heights, femur_pred_xgb, color='red', label='Femur XGB')
            if show_femur_gbr:
                ax.plot(heights, femur_pred_gbr, color='green', label='Femur GBR')
            if show_femur_ridge:
                ax.plot(heights, femur_pred_ridge, color='orange', label='Femur Ridge')
            if show_femur_stack:
                ax.plot(heights, femur_pred_stack, color='purple', label='Femur Stack')

        ax.set_xlabel('Height (inches)')
        ax.set_ylabel('Predicted Size')
        ax.set_title(f'Height vs Predicted Size ({"Female" if sex_val == 0 else "Male"})')
        ax.legend()

        self.height_vs_pred_fig = fig
        st.pyplot(fig)

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

    def display_interactive_table(self, tibia_metrics_xgb, tibia_metrics_gbr, tibia_metrics_ridge, tibia_metrics_stack, femur_metrics_xgb, femur_metrics_gbr, femur_metrics_ridge, femur_metrics_stack):
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

    def evaluate_models(self):
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

        self.display_interactive_table(tibia_metrics_xgb, tibia_metrics_gbr, tibia_metrics_ridge, tibia_metrics_stack, femur_metrics_xgb, femur_metrics_gbr, femur_metrics_ridge, femur_metrics_stack)

    def plot_qq_plots(self):
        features = ['height_log', 'age_height_interaction', 'sex']
        X_tibia = self.data[features].values
        y_tibia = self.data['tibia used'].values
        X_femur = self.data[features].values
        y_femur = self.data['femur used'].values

        X_tibia_scaled = self.models['tibia']['scaler'].transform(X_tibia)
        X_femur_scaled = self.models['femur']['scaler'].transform(X_femur)

        residuals = {
            'tibia_xgb': y_tibia - self.models['tibia']['xgb'].predict(X_tibia_scaled),
            'tibia_gbr': y_tibia - self.models['tibia']['gbr'].predict(X_tibia_scaled),
            'tibia_ridge': y_tibia - self.models['tibia']['ridge'].predict(X_tibia_scaled),
            'tibia_stack': y_tibia - self.models['tibia']['stack'].predict(X_tibia_scaled),
            'femur_xgb': y_femur - self.models['femur']['xgb'].predict(X_femur_scaled),
            'femur_gbr': y_femur - self.models['femur']['gbr'].predict(X_femur_scaled),
            'femur_ridge': y_femur - self.models['femur']['ridge'].predict(X_femur_scaled),
            'femur_stack': y_femur - self.models['femur']['stack'].predict(X_femur_scaled),
        }

        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle('QQ Plots for Residuals')

        for i, (key, res) in enumerate(residuals.items()):
            ax = axes[i // 2, i % 2]
            stats.probplot(res, dist="norm", plot=ax)
            ax.set_title(key.replace('_', ' ').title())

        self.qq_plot_fig = fig
        st.pyplot(fig)

    def calculate_residual_tests(self):
        features = ['height_log', 'age_height_interaction', 'sex']
        X_tibia = self.data[features].values
        y_tibia = self.data['tibia used'].values
        X_femur = self.data[features].values
        y_femur = self.data['femur used'].values

        X_tibia_scaled = self.models['tibia']['scaler'].transform(X_tibia)
        X_femur_scaled = self.models['femur']['scaler'].transform(X_femur)

        residuals = {
            'tibia_xgb': y_tibia - self.models['tibia']['xgb'].predict(X_tibia_scaled),
            'tibia_gbr': y_tibia - self.models['tibia']['gbr'].predict(X_tibia_scaled),
            'tibia_ridge': y_tibia - self.models['tibia']['ridge'].predict(X_tibia_scaled),
            'tibia_stack': y_tibia - self.models['tibia']['stack'].predict(X_tibia_scaled),
            'femur_xgb': y_femur - self.models['femur']['xgb'].predict(X_femur_scaled),
            'femur_gbr': y_femur - self.models['femur']['gbr'].predict(X_femur_scaled),
            'femur_ridge': y_femur - self.models['femur']['ridge'].predict(X_femur_scaled),
            'femur_stack': y_femur - self.models['femur']['stack'].predict(X_femur_scaled),
        }

        residual_tests_data = {
            'Model': [],
            'T-score': [],
            'P-value': [],
            'Wilcoxon P-value': []
        }

        model_pairs = [
            ('tibia_xgb', 'tibia_gbr'),
            ('tibia_xgb', 'tibia_ridge'),
            ('tibia_gbr', 'tibia_ridge'),
            ('tibia_gbr', 'tibia_stack'),
            ('femur_xgb', 'femur_gbr'),
            ('femur_xgb', 'femur_ridge'),
            ('femur_gbr', 'femur_ridge'),
            ('femur_gbr', 'femur_stack')
        ]

        for (model1, model2) in model_pairs:
            t_score, p_value = ttest_rel(residuals[model1], residuals[model2])
            _, wilcoxon_p_value = wilcoxon(residuals[model1], residuals[model2])

            residual_tests_data['Model'].append(f'{model1} vs {model2}')
            residual_tests_data['T-score'].append(round(t_score, 4))
            residual_tests_data['P-value'].append(round(p_value, 4))
            residual_tests_data['Wilcoxon P-value'].append(round(wilcoxon_p_value, 4))

        df_residual_tests = pd.DataFrame(residual_tests_data)
        st.table(df_residual_tests)

    def save_outputs_to_pdf(self):
        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Total Knee Size Predictor Results", ln=True, align="C")

        # Prediction Table
        if self.prediction_df is not None:
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt="Predictions", ln=True, align="L")
            pdf.set_font("Arial", size=8)
            for i in range(len(self.prediction_df)):
                row = self.prediction_df.iloc[i]
                pdf.cell(200, 10, txt=str(row.values), ln=True, align="L")

        # Metrics Table
        if self.metrics_df is not None:
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt="Metrics", ln=True, align="L")
            pdf.set_font("Arial", size=8)
            for i in range(len(self.metrics_df)):
                row = self.metrics_df.iloc[i]
                pdf.cell(200, 10, txt=str(row.values), ln=True, align="L")

        # Height vs Predicted Size Plot
        if self.height_vs_pred_fig is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                self.height_vs_pred_fig.savefig(tmpfile.name, format='png')
                pdf.add_page()
                pdf.image(tmpfile.name, x=10, y=10, w=190)
                tmpfile.close()

        # Learning Curves Plot
        if self.learning_curve_fig is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                self.learning_curve_fig.savefig(tmpfile.name, format='png')
                pdf.add_page()
                pdf.image(tmpfile.name, x=10, y=10, w=190)
                tmpfile.close()

        # QQ Plots
        if self.qq_plot_fig is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                self.qq_plot_fig.savefig(tmpfile.name, format='png')
                pdf.add_page()
                pdf.image(tmpfile.name, x=10, y=10, w=190)
                tmpfile.close()

        pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(pdf_output.name)
        st.success("PDF saved successfully!")
        st.download_button("Download PDF", data=open(pdf_output.name, "rb").read(), file_name="output.pdf")

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

            predictor.plot_height_vs_predicted_size(sex_val, age)

            if st.button("Predict"):
                model_type = st.selectbox("Select Model", ["xgb", "gbr", "ridge", "stack"])
                predictor.predict(age, height, sex_val, model_type)

            if st.button("Training CV Validation Plots"):
                predictor.plot_superimposed_learning_curves()

            if st.button("QQ Plots for Residuals"):
                predictor.plot_qq_plots()

            if st.button("Evaluate Models"):
                predictor.evaluate_models()

            if st.button("Calculate Residual Tests"):
                predictor.calculate_residual_tests()

            if st.button("Fit Models with Regression Line"):
                predictor.fit_regression_line(sex_val, age)

            if st.button("Save Outputs to PDF"):
                predictor.save_outputs_to_pdf()

    st.write("""
        **Disclaimer:** This application is for educational purposes only. It is not intended to diagnose, provide medical advice, or offer recommendations. The predictions made by this application are not validated and should be used for research purposes only.
    """)

if __name__ == "__main__":
    main()
