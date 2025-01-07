# End-to-End Churn Modelling Using ANN

This project demonstrates an end-to-end machine learning pipeline for customer churn prediction using an Artificial Neural Network (ANN). The workflow includes data preprocessing, model training, hyperparameter tuning, evaluation, and deployment. TensorBoard has been integrated to monitor and visualize the training process.

## Project Structure

The repository is structured as follows:

```
END TO END CHURN MODELLING
├── logs/                  # Directory containing logs generated during training.
├── output/                # Directory for saving outputs, such as model predictions.
├── regressionlogs/        # Logs related to regression experiments.
├── TensorBoard/           # TensorBoard logs and visualization data.
├── app.py                 # Flask application for model deployment.
├── Churn_Modelling.csv    # Dataset used for training and testing the model.
├── experiments.ipynb      # Jupyter notebook for exploratory data analysis and experiments.
├── hyperparametertuningann.ipynb  # Notebook for hyperparameter tuning of the ANN model.
├── label_encoder_gender.pkl       # Pickle file for gender label encoder.
├── model.h5               # Trained ANN model saved in HDF5 format.
├── onehot_encoder_geo.pkl # Pickle file for geographical one-hot encoder.
├── prediction.ipynb       # Notebook for generating predictions with the trained model.
├── regression_model.h5    # Regression model saved in HDF5 format.
├── requirements.txt       # Python dependencies for the project.
├── salaryregression.ipynb # Notebook for salary regression experiments.
├── scaler.pkl             # Pickle file for scaling numerical features.
```

## Key Features

1. **Data Preprocessing**
   - Categorical data encoded using one-hot and label encoding.
   - Numerical data scaled using a scaler for efficient training.

2. **Model Training**
   - ANN model implemented for predicting customer churn.
   - Regression models for related tasks, including salary prediction.

3. **Hyperparameter Tuning**
   - `hyperparametertuningann.ipynb` contains experiments for optimizing model performance.

4. **Model Deployment**
   - `app.py` contains the Flask-based web application for serving predictions.

5. **Visualization with TensorBoard**
   - TensorBoard integrated for visualizing training metrics like loss and accuracy.
   - Logs available in the `TensorBoard/` directory.

## Requirements

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**:
   - Use the `experiments.ipynb` to clean and preprocess the dataset (`Churn_Modelling.csv`).

2. **Model Training**:
   - Train the ANN model using `hyperparametertuningann.ipynb` or implement regression models via `salaryregression.ipynb`.

3. **Model Deployment**:
   - Run the Flask app:
     ```bash
     python app.py
     ```
   - Access the application at `http://localhost:5000`.

4. **TensorBoard Visualization**:
   - Start TensorBoard:
     ```bash
     tensorboard --logdir=TensorBoard/
     ```
   - Open `http://localhost:6006` in your browser to visualize the metrics.

## Dataset

The `Churn_Modelling.csv` file contains the dataset used for this project. It includes customer information such as demographics, account details, and churn status.

## Outputs

- Saved models: `model.h5` and `regression_model.h5`.
- Encoders and scaler saved as `.pkl` files for reuse.
- Logs for training and experiments.

