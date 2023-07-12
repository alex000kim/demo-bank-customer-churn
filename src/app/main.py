import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from jsonschema import validate
from utils.load_params import load_params

app = FastAPI()
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

params = load_params(params_path='params.yaml')
model_path = params.train.model_path
feat_cols = params.base.feat_cols
model = load(filename=model_path)

schema = {
    "type" : "array",
    "items":
        {
    "type": "object",
    "properties" :
                {
        "CreditScore" : {"type" : "number"},
        "Age" : {"type" : "number"},
        "Tenure" : {"type" : "number"},
        "Balance" : {"type" : "number"},
        "NumOfProducts" : {"type" : "number"},
        "HasCrCard" : {"type" : "number"},
        "IsActiveMember" : {"type" : "number"},
        "EstimatedSalary" : {"type" : "number"}
                },
        }
}

@app.post("/predict")
async def predict(info : Request):
    json_list = await info.json()
    validate(instance=json_list, schema=schema)
    input_data = pd.DataFrame(json_list)
    probs = model.predict_proba(input_data)[:,0]
    probs = probs.tolist()
    return probs
