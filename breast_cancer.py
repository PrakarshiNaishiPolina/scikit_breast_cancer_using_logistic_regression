from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

# load the dataset

data=load_breast_cancer()
print(data.keys())
X= data.data
y=data.target

# split the dataset into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# standardize the data = preprocessing the data

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# initialize the models
log_reg=LogisticRegression(random_state=42)

# train the models
log_reg.fit(X_train,y_train)

# make predictions

y_pred=log_reg.predict(X_test)

# evaluate the models

print("Logistic Regression Accuracy: ",accuracy_score(y_test,y_pred))
print("\nLogistic Regression Classification Report:\n",classification_report(y_test,y_pred))