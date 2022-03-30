import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

df=pd.read_csv("Training Data.csv")
leMarried= LabelEncoder()
leHouse= LabelEncoder()
leCar= LabelEncoder()
leProfession= LabelEncoder()

df['Married/Single']=leMarried.fit(df['Married/Single'])
df['House_Ownership']=leHouse.fit(df['House_Ownership'])
df['Car_Ownership']=leCar.fit(df['Car_Ownership'])
df['Profession']=leProfession.fit(df['Profession'])



# Create flask app
app = Flask(__name__)
CORS(app)
model = pickle.load(open("m1.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    int_features = [x for x in request.form.values()]

    Income, Age,Experience,MarriedSingle, House_Ownership,Car_Ownership, Profession, CURRENT_JOB_YRS, CURRENT_HOUSE_YRS=int_features
    data = {'Income':[Income],'Age':[Age],'Experience':[Experience],'Married/Single':[MarriedSingle], 'House_Ownership':[House_Ownership], 'Car_Ownership':[Car_Ownership], 'Profession':[Profession],'CURRENT_JOB_YRS':[CURRENT_JOB_YRS],'CURRENT_HOUSE_YRS':[CURRENT_HOUSE_YRS]}
    # Create the pandas DataFrame
    dfM = pd.DataFrame(data)
    dfM['Married/Single']=leMarried.transform(dfM['Married/Single'])
    dfM['House_Ownership']=leHouse.transform(dfM['House_Ownership'])
    dfM['Car_Ownership']=leCar.transform(dfM['Car_Ownership'])
    dfM['Profession']=leProfession.transform(dfM['Profession'])

    prediction = model.predict(dfM)
    if (prediction[0]==0):
        res= {
            "Status":"Approved"
        }

    if (prediction[0]==1):
        res= {
            "Status":"Rejected"
        }
    # return render_template("index.html", prediction_text = res)
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)     