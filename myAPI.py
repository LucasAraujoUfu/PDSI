from fastapi import FastAPI
import json
import pandas as pd
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer as Bc

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

knn = None

knn = KNeighborsClassifier(n_neighbors=3)
data = Bc()
knn.fit(data.data, data.target)

@app.get('/train')
def train():
    knn.fit(data.data, data.target)


@app.get('/consult')
def consult(aluno: str):
    if knn == None:
        return '-1'
    al = json.loads(aluno)
    return str(knn.predict([al['notas']])[0])


@app.get('/consultAll')
def consAll(dataset:str):
    if knn == None:
        return '-1'
    url = dataset
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    s = {}
    try:
        df = pd.read_csv(path)
        X = np.array(df)
        l = []
        for i in X:
            a = knn.predict([i[1:]])[0]
            l.append(a)
        s["result"]=l
        k = str(s)
        k = k.replace("'", '"')
        return k
    except Exception:
        s["result"] = ["Erro inesperado"]
        k = str(s)
        k = k.replace("'", '"')
        return k

