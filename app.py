from flask import Flask, jsonify
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score


app = Flask(__name__)

datos = pd.read_csv("unitsData.csv")
x = datos.drop(columns=['Won'])
y = datos['Won']
#NB Model
# Create a Naive Bayes object
NB_model = GaussianNB()
#Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)
#Training the model
NB_model.fit(x_train, y_train)

#Evaluation
y_pred = NB_model.predict(x_test)
#Check performance of model
score = precision_score(y_test, y_pred, average='micro')

@app.route('/')
def home():
    return "Está funcionando"

@app.route('/positions')
def getPositions():
    import random

    n = 15
    won=0
    while(won==0):
        listaUnidades = []
        cont = 0
        while(n!=cont):
            if(cont<n):
                temp= random.randint(1,3)
                listaUnidades.append(temp)
                cont+=temp
            else:
                listaUnidades=[]
                cont=0

        tempList = []
        for i in range(0,66):
            tempList.append(0)

        for tipo in listaUnidades:
            sw=False
            posR = random.randint(0, 65)
            while(sw==False):
                if(tempList[posR]==0):
                    tempList[posR]=tipo
                    sw=True
                else:
                    if(posR<65):
                        posR+=1
                    else:
                        posR = random.randint(0, 65)
        
        won=NB_model.predict([tempList])[0]
    return tempList


if __name__ == '__main__':
    app.run(port=4000)