Fraud Detection System
This project is a machine learning-based fraud detection system built to identify fraudulent credit card transactions. The core of the system is a Random Forest Classifier model, deployed as a web application using the Flask framework. The solution is specifically designed to handle the challenge of class imbalance, which is a common and critical issue in real-world fraud datasets.

Problem Solved
Fraud datasets are highly imbalanced, with a very small number of fraudulent transactions compared to legitimate ones. A standard machine learning model often fails to identify fraud (low recall), even if its overall accuracy is high. This system addresses this by using SMOTE (Synthetic Minority Over-sampling Technique) on the training data. This balances the dataset, allowing the model to learn the patterns of fraudulent behavior more effectively, thereby maximizing its ability to catch fraud.

Project Features
Machine Learning Model: A robust Random Forest Classifier is used for accurate and fast predictions.

Imbalance Handling: The model is trained using SMOTE to significantly improve its recall score.

Web Application: A user-friendly web interface is built with Flask for real-time transaction prediction and easy interaction.

Performance Evaluation: A dedicated endpoint provides a visual confusion matrix and a detailed classification report to assess the model's performance on the test dataset.

Data Pipeline: The project includes a pre-processing script to convert the original .pkl data files into the .csv format used for training and testing.

Model Persistence: The trained model and data encoders are saved using joblib for efficient re-loading and faster deployment.

How to Run
Clone the repository:

Bash

git clone <your-repo-link>
cd <your-project-folder>
Set up a virtual environment (recommended):

Bash

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Bash

pip install Flask pandas scikit-learn seaborn matplotlib imbalanced-learn joblib numpy

Prepare the dataset:
Place your .csv data files into a folder named formatteddataset/ in the project's root directory. If you have the original .pkl files, you can use the provided conversion script at the top of app.py to generate the .csv files.

Run the Flask application:

Bash

python app.py
Access the application:
Open your web browser and navigate to http://127.0.0.1:5000/. You can use the main page to enter transaction details for a prediction or visit /confusion to view the model's performance metrics.
