import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
import warnings
import requests
from io import StringIO
import tempfile
from fpdf import FPDF

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    gbr_tibia = GradientBoostingRegressor(n_estimators=100, random_state=1)
    gbr_femur = GradientBoostingRegressor(n_estimators=100, random_state=1)
    xgb_tibia = XGBRegressor(n_estimators=100, random_state=1)
    xgb_femur = XGBRegressor(n_estimators=100, random_state=1)
    
    gbr_tibia.fit(X_scaled, y_tibia)
    gbr_femur.fit(X_scaled, y_femur)
    xgb_tibia.fit(X_scaled, y_tibia)
    xgb_femur.fit(X_scaled, y_femur)
    
    return {
        'tibia': {'gbr': gbr_tibia, 'xgb': xgb_tibia, 'scaler': scaler},
        'femur': {'gbr': gbr_femur, 'xgb': xgb_femur, 'scaler': scaler}
    }

class TibiaFemurPredictor:
    def __init__(self):
        self.models = None
        self.data = None
        self.prediction_df = None

    def train_models(self):
        self.models = train_and_scale_models(self.data, ['height_log', 'age_height_interaction', 'sex'])
        st.session_state['models'] = self.models
        st.success("Models trained and scalers initialized.")

    def predict(self, age, height, sex_val):
        if not self.models:
            st.error("Models are not trained yet.")
            return

        X_new = np.array([[np.log1p(height), age * height, sex_val]])
        scaler = self.models['tibia']['scaler']
        X_new_scaled = scaler.transform(X_new)
        
        pred_tibia_gbr = self.models['tibia']['gbr'].predict(X_new_scaled)[0]
        pred_femur_gbr = self.models['femur']['gbr'].predict(X_new_scaled)[0]
        pred_tibia_xgb = self.models['tibia']['xgb'].predict(X_new_scaled)[0]
        pred_femur_xgb = self.models['femur']['xgb'].predict(X_new_scaled)[0]

        # Linear regression line prediction for GBR model
        heights = np.linspace(60, 76, 100)
        tibia_pred_gbr = [self.models['tibia']['gbr'].predict(scaler.transform(np.array([[np.log1p(h), age * h, sex_val]])))[0] for h in heights]
        femur_pred_gbr = [self.models['femur']['gbr'].predict(scaler.transform(np.array([[np.log1p(h), age * h, sex_val]])))[0] for h in heights]

        tibia_reg = LinearRegression().fit(heights.reshape(-1, 1), tibia_pred_gbr)
        femur_reg = LinearRegression().fit(heights.reshape(-1, 1), femur_pred_gbr)

        tibia_reg_pred = tibia_reg.predict(np.array([[height]]))[0]
        femur_reg_pred = femur_reg.predict(np.array([[height]]))[0]

        prediction_data = {
            "Model": ["GBR", "GBR with Reg Line", "XGB"],
            "Predicted Femur": [round(pred_femur_gbr, 1), round(femur_reg_pred, 1), round(pred_femur_xgb, 1)],
            "Predicted Tibia": [round(pred_tibia_gbr, 1), round(tibia_reg_pred, 1), round(pred_tibia_xgb, 1)]
        }

        prediction_df = pd.DataFrame(prediction_data)
        self.prediction_df = prediction_df
        st.session_state['prediction_df'] = prediction_df

    def display_prediction(self):
        if self.prediction_df is not None:
            st.table(self.prediction_df)

            # Highlight the row based on the rounded value of the GBR predicted femur size
            femur_df = pd.DataFrame(femur_sizes).T
            femur_df.columns = ["A", "B"]
            femur_df.index.name = "Size"
            femur_df.index = femur_df.index.astype(int)
            femur_df = femur_df.reset_index()

            def highlight_row(s):
                return ['background-color: yellow' if s['Size'] == round(self.prediction_df.loc[0, "Predicted Femur"]) else '' for _ in s.index]

            st.table(femur_df.style.apply(highlight_row, axis=1))

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

            if st.button("Predict"):
                predictor.predict(age, height, sex_val)
                predictor.display_prediction()

            if st.button("Save Outputs to PDF"):
                predictor.save_outputs_to_pdf()

    st.write("""
        **Disclaimer:** This application is for educational purposes only. It is not intended to diagnose, provide medical advice, or offer recommendations. The predictions made by this application are not validated and should be used for research purposes only.
    """)

if __name__ == "__main__":
    main()
