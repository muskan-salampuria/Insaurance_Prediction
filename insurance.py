"""
Insurance Data using 
Logistic Regression 
"""
import pandas as pd
from matplotlib import pyplot as plt

#Fetching Data set
df=pd.read_csv("insurance_data.csv")

#plotting features vs result
plt.scatter(df.age,df.bought_insurance, marker="*",color="Red")

#Now splitting test train data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(df[["age"]],df.bought_insurance,test_size=0.1)

#training model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

#Predicting Values
model.predict(X_test)

#Checking the accuracy of the model
print(model.score(X_test,y_test))



