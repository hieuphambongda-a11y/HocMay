import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('BostonHousing.csv')

X = data.drop('medv',axis=1)
y = data['medv']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)

