import pandas as pd 
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# load csv file
df = pd.read_csv("Cellphone.csv")
#printing data 
print(df.head())
#select independent variables
X=df[["weight","resoloution","internal mem","cpu core","ram","battery"]]
y=df["Price"]

#split into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=50)

#Feature Scaling kini ne bola tha karne ke liye
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#instantiate model
classifier =RandomForestClassifier()

#model fit
classifier.fit(X_train, y_train)

#picle file stage 2 karni hai pata ni kaise

pickle.dump(classifier, open("model.pkl","wb"))