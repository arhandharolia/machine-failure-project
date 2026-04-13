# 🏭 Machine Failure Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)

## 📌 Objective
This is a Capstone project meant to predict machine failures in advance using industrial sensor data. By reliably predicting failures before they happen, we can drastically reduce machine downtime, save on maintenance costs, and manage resources safely.

## 🛠 Features
1. **Machine Learning Pipeline**: Data preprocessing, feature scaling, and predictive modeling using Logistic Regression and K-Nearest Neighbors (KNN). The final KNN model yielded over a **97% accuracy score**.
2. **Streamlit Web Application**: An interactive dark-mode dashboard mapping to professional "Google Stitch" UI aesthetics. It features:
   - **Authentication Portal**: Simulated robust login screen.
   - **Machine Health Dashboard**: Centralized metrics showing probability of failure.
   - **Sensor Data Analytics**: Live tracking for RPM, Torque, and Temperatures.
   - **Alerts & History Log**: Categorized operational history.

## 📁 Project Structure
- `app.py`: The Main Streamlit Application serving the dashboards.
- `notebook.ipynb`: Jupyter Notebook detailing data cleaning, scaling, and model training.
- `data/`: Contains the `predictive_maintenance.csv` dataset.
- `model.pkl` & `scaler.pkl`: The serialized Machine Learning artifacts.
- `requirements.txt`: Project package dependencies list.

## 🚀 Setup & Installation
Follow these steps to run the Streamlit application on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arhandharolia/machine-failure-project.git
   cd machine-failure-project
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   *Note: If `streamlit` is not recognized, run `python -m streamlit run app.py` instead.*

## 📊 Dataset Details
The predictive maintenance dataset contains key industrial sensor features:
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- **Target Label**: Failure (0 = Healthy, 1 = Machine Failure)
