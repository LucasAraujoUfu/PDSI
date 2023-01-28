from fastapi import FastAPI
import json

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

knn = KNeighborsClassifier(n_neighbors=3)
data = Bc()
knn.fit(data.data, data.target)


@app.get('/consult')
def consult(aluno: str):
    al = json.loads(aluno)
    return str(knn.predict([al['notas']])[0])


