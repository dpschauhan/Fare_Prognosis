import numpy as np 
from flask import Flask,request,jsonify,render_template
import pickle
# app bnana hai 
app= Flask(__name__)

#load model
model = pickle.load(open("model.pkl","rb"))



@app.route("/hello")
def bn():
    return "HEllo its me"

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features= [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text = "The Mobile Price Could be Approximately {}".format(prediction))

if __name__=="__main__":
    app.run(debug=True)
