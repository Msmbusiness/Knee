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
from matplotlib.backends.backend_pdf import PdfPages
import tkinter as tk
from tkinter import filedialog, messagebox

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*test scores are non-finite.*", category=UserWarning, module='sklearn.model_selection._search')

class TibiaFemurPredictor:
    def __init__(self):
        self.models = {
            'tibia': {'xgb': None, 'gbr': None, 'scaler': None},
            'femur': {'xgb': None, 'gbr': None, 'scaler': None}
        }
        self.data_file_path = None
        self.data = None

    def upload_data(self):
        self.data_file_path = filedialog.askopenfilename()
        if self.data_file_path:
            self.load_data(self.data_file_path)
            messagebox.showinfo("Info", "Data file loaded successfully.")

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data.columns = self.data.columns.str.strip()
        self.data['age_height_interaction'] = self.data['age'] * self.data['height']
        self.data['height_log'] = np.log1p(self.data['height'])

    def oversample_minority_group(self):
        if 0 in self.data['sex'].values and 1 in self.data['sex'].values:
            X = self.data.drop('sex', axis=1)
            y = self.data['sex']

            ros = RandomOverSampler(random_state=1)
            X_resampled, y_resampled = ros.fit_resample(X, y)

            self.data = pd.concat([X_resampled, y_resampled], axis=1)
        else:
            messagebox.showwarning("Warning", "Both male and female samples are required for oversampling.")

    def train_and_scale_models(self, df, features):
        if len(df) < 2:
            messagebox.showwarning("Warning", "Not enough data to train models.")
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

    def evaluate_models(self, features):
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

        self.display_interactive_table(tibia_metrics_xgb, "Tibia Metrics (XGB)")
        self.display_interactive_table(tibia_metrics_gbr, "Tibia Metrics (GBR)")
        self.display_interactive_table(femur_metrics_xgb, "Femur Metrics (XGB)")
        self.display_interactive_table(femur_metrics_gbr, "Femur Metrics (GBR)")

        # Plot histograms of residuals
        self.plot_residuals_histogram(tibia_metrics_xgb['residuals'], tibia_metrics_gbr['residuals'], femur_metrics_xgb['residuals'], femur_metrics_gbr['residuals'])

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

    def display_interactive_table(self, metrics, title):
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title(title)

        columns = ('Metric', 'Value')
        tree = ttk.Treeview(root, columns=columns, show='headings')

        tree.heading('Metric', text='Metric')
        tree.heading('Value', text='Value')

        data = { "Metric": ['r2_score', 'rmse', 'mse', 'mae', 'mape', 'kurtosis'],
                 "Value": [f"{metrics['r2_score']:.2f}", f"{metrics['rmse']:.2f}", f"{metrics['mse']:.2f}",
                           f"{metrics['mae']:.2f}", f"{metrics['mape']:.2f}", f"{metrics['kurtosis']:.2f}"] }

        for metric, value in zip(data['Metric'], data['Value']):
            tree.insert('', tk.END, values=(metric, value))

        tree.pack()
        root.mainloop()

    def plot_residuals_histogram(self, tibia_residuals_xgb, tibia_residuals_gbr, femur_residuals_xgb, femur_residuals_gbr):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].hist(tibia_residuals_xgb, bins=20, color='blue', alpha=0.7)
        axes[0, 0].set_title("Residuals Histogram (XGB) - Tibia")
        axes[0, 0].set_xlabel("Residuals")
        axes[0, 0].set_ylabel("Frequency")

        axes[0, 1].hist(tibia_residuals_gbr, bins=20, color='green', alpha=0.7)
        axes[0, 1].set_title("Residuals Histogram (GBR) - Tibia")
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")

        axes[1, 0].hist(femur_residuals_xgb, bins=20, color='blue', alpha=0.7)
        axes[1, 0].set_title("Residuals Histogram (XGB) - Femur")
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")

        axes[1, 1].hist(femur_residuals_gbr, bins=20, color='green', alpha=0.7)
        axes[1, 1].set_title("Residuals Histogram (GBR) - Femur")
        axes[1, 1].set_xlabel("Residuals")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), ax=None, color=None):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.grid()

        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color=color)
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color=color)
        ax.plot(train_sizes, train_scores_mean, 'o-', color=color,
                label=f"Training score {title}")
        ax.plot(train_sizes, test_scores_mean, 'o-', color=color,
                label=f"Cross-validation score {title}")

    def plot_superimposed_learning_curves(self):
        if self.data is None:
            print("Error: Data not loaded or processed correctly.")
            return

        if len(self.data) < 2:
            print("Not enough data to plot learning curves.")
            return

        X_tibia = self.data[['height_log', 'age_height_interaction', 'sex']].values
        y_tibia = self.data['tibia used'].values
        X_femur = self.data[['height_log', 'age_height_interaction', 'sex']].values
        y_femur = self.data['femur used'].values

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        self.plot_learning_curve(self.models['tibia']['xgb'], "Tibia XGB", X_tibia, y_tibia, ax=axes[0], color='blue')
        self.plot_learning_curve(self.models['tibia']['gbr'], "Tibia GBR", X_tibia, y_tibia, ax=axes[0], color='green')

        self.plot_learning_curve(self.models['femur']['xgb'], "Femur XGB", X_femur, y_femur, ax=axes[1], color='blue')
        self.plot_learning_curve(self.models['femur']['gbr'], "Femur GBR", X_femur, y_femur, ax=axes[1], color='green')

        axes[0].legend(loc="best")
        axes[1].legend(loc="best")

        plt.tight_layout()
        plt.show()

    def save_superimposed_learning_curves(self):
        if self.data is None:
            print("Error: Data not loaded or processed correctly.")
            return

        if len(self.data) < 2:
            print("Not enough data to plot learning curves.")
            return

        X_tibia = self.data[['height_log', 'age_height_interaction', 'sex']].values
        y_tibia = self.data['tibia used'].values
        X_femur = self.data[['height_log', 'age_height_interaction', 'sex']].values
        y_femur = self.data['femur used'].values

        with PdfPages('superimposed_learning_curves.pdf') as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            self.plot_learning_curve(self.models['tibia']['xgb'], "Tibia XGB", X_tibia, y_tibia, ax=axes[0], color='blue')
            self.plot_learning_curve(self.models['tibia']['gbr'], "Tibia GBR", X_tibia, y_tibia, ax=axes[0], color='green')

            self.plot_learning_curve(self.models['femur']['xgb'], "Femur XGB", X_femur, y_femur, ax=axes[1], color='blue')
            self.plot_learning_curve(self.models['femur']['gbr'], "Femur GBR", X_femur, y_femur, ax=axes[1], color='green')

            axes[0].legend(loc="best")
            axes[1].legend(loc="best")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        messagebox.showinfo("Info", "Learning curves saved to PDF.")

    def predict(self, age, height, sex, model_type):
        if self.models['tibia']['scaler'] is None or self.models['femur']['scaler'] is None:
            messagebox.showerror("Error", "Models are not trained yet.")
            return

        X_new = np.array([[np.log1p(height), age * height, sex]])
        tibia_scaler = self.models['tibia']['scaler']
        femur_scaler = self.models['femur']['scaler']
        X_new_scaled = tibia_scaler.transform(X_new)

        preds_tibia = self.models['tibia'][model_type].predict(X_new_scaled)
        preds_femur = self.models['femur'][model_type].predict(X_new_scaled)

        result_text_tibia = (f"Predicted Tibia Used with {model_type.upper()}: {preds_tibia[0]:.1f} (MAE: {self.calculate_metrics(X_new_scaled, preds_tibia, 'tibia', model_type)['mae']:.2f})")
        result_text_femur = (f"Predicted Femur Used with {model_type.upper()}: {preds_femur[0]:.1f} (MAE: {self.calculate_metrics(X_new_scaled, preds_femur, 'femur', model_type)['mae']:.2f})")

        messagebox.showinfo("Prediction", f"{result_text_tibia}\n{result_text_femur}")

def main():
    predictor = TibiaFemurPredictor()

    root = tk.Tk()
    root.title("Tibia Femur Predictor")

    upload_button = tk.Button(root, text="Upload Data File", command=predictor.upload_data)
    upload_button.pack()

    train_button = tk.Button(root, text="Train Models", command=predictor.train_models)
    train_button.pack()

    evaluate_button = tk.Button(root, text="Evaluate Models", command=lambda: predictor.evaluate_models(['height_log', 'age_height_interaction', 'sex']))
    evaluate_button.pack()

    age_label = tk.Label(root, text="Age:")
    age_label.pack()
    age_entry = tk.Entry(root)
    age_entry.pack()

    height_label = tk.Label(root, text="Height (inches):")
    height_label.pack()
    height_entry = tk.Entry(root)
    height_entry.pack()

    sex_label = tk.Label(root, text="Sex:")
    sex_label.pack()
    sex_var = tk.StringVar(root)
    sex_var.set("Female")
    sex_option = tk.OptionMenu(root, sex_var, "Female", "Male")
    sex_option.pack()

    model_label = tk.Label(root, text="Model:")
    model_label.pack()
    model_var = tk.StringVar(root)
    model_var.set("xgb")
    model_option = tk.OptionMenu(root, model_var, "xgb", "gbr")
    model_option.pack()

    def predict_callback():
        age = float(age_entry.get())
        height = float(height_entry.get())
        sex = 0 if sex_var.get() == "Female" else 1
        model_type = model_var.get()
        predictor.predict(age, height, sex, model_type)

    predict_button = tk.Button(root, text="Predict", command=predict_callback)
    predict_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
