import pandas as pd
from sklearn.linear_model import PoissonRegressor
import joblib

# Simple starter dataset
data = {
    "sot":[5,4,6,3,7],
    "bc":[2,1,3,1,4],
    "bcm":[1,1,1,0,2],
    "gpg":[1.5,1.2,2.0,0.8,2.5],
    "pos":[55,48,60,45,62],
    "offsides":[2,1,3,1,2],
    "fouls":[10,12,9,14,8],
    "opp_con_pg":[1.2,1.5,1.0,1.8,0.9],
    "opp_cs":[8,6,10,5,12],
    "form":[1.1,1.0,1.2,0.9,1.3],
    "inj":[0.1,0.2,0.05,0.3,0.0],
    "h2h":[0.1,-0.1,0.2,-0.2,0.0],
    "home":[1,0,1,0,1],
    "goals":[2,1,3,0,4]
}

df = pd.DataFrame(data)

X = df.drop(columns=["goals"])
y = df["goals"]

model = PoissonRegressor()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("✅ model.pkl created")
