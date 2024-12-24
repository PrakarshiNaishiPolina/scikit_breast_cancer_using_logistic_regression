Breast Cancer Classification with Logistic Regression
This project demonstrates the process of training and evaluating a machine learning model using the breast cancer dataset. The model is trained using Logistic Regression to predict whether a tumor is malignant or benign.
Dataset
The project uses the Breast Cancer Wisconsin dataset, which contains 30 features computed from digitized images of fine needle aspirates (FNAs) of breast cancer biopsies. The goal is to predict whether the tumor is malignant (1) or benign (0).

Features: 30 numeric features describing various attributes like radius, texture, smoothness, etc.
Target: A binary classification target (0 = benign, 1 = malignant).
Steps
Loading the Dataset: The dataset is loaded using load_breast_cancer() from sklearn.datasets.

Splitting the Data: The dataset is divided into training and testing sets using train_test_split(). 70% of the data is used for training and 30% for testing.

Data Preprocessing: The features are standardized using StandardScaler(). This ensures all the features are on a similar scale, which helps improve model performance.

Model Training: Logistic Regression is used to train the model on the standardized training data.

Making Predictions: The trained model is used to make predictions on the test set.

Model Evaluation: The performance of the model is evaluated using accuracy score and a detailed classification report.
