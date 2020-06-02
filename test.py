import tensorflow
import keras
import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style




data = pd.read_csv("student-mat.csv", sep=";")
#print(data)
data = data[["G1", "G2", "G3","studytime","failures","absences", "freetime"]]
print(data)
predict = "G3"
x=np.array(data.drop([predict], 1))
y=np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
linear =linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print(accuracy)

with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
print("Coefficient: \n", linear.coef_)
print("Intercept:\n", linear.intercept_)
predictions = linear.predict(x_test)
print('predicted grade 3, grade 1, grade 2, studytime, failures, absences, freetime, actual grade 3')
for x in range(len(predictions)):
    print(predictions[x],x_test[x], y_test[x])