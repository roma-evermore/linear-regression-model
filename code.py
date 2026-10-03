#X minimum and maximum are very different. Hence predictions are wrong. If you want them to be realistic change minimum and maximum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#dowload&preparedata
#data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv("C:\\Users\\romak\\OneDrive\\Документы\\Downloads\\gdp_and_life_satisfaction_2020.csv")
X = lifesat[["GDP Per Capita (USD)"]].values
y = lifesat[["Life Satisfaction (0-10)"]].values
#visualising
lifesat.plot(kind='scatter', grid=True,
             x="GDP Per Capita (USD)", y="Life Satisfaction (0-10)")
plt.axis([936, 175814, 4.3, 7.8])
plt.show()
#selection of a Model
model = LinearRegression()
#TrainModel
model.fit(X,y)
#Prediction
X_new = [[89312]]
prediction = model.predict(X_new)

print("GDP Per Capita:", X_new[0][0])
print("Predicted Life Satisfaction: ", prediction[0][0])
print("Coeffience: ", model.coef_[0][0])
print("Model Intercept: ", model.intercept_[0])
