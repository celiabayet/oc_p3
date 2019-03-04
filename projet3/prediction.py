from flask import Flask
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import request
import json


#Importing data
movie =  pd.read_csv("movie",index_col=None, sep="\t", encoding = 'utf-8')
data =  pd.read_csv("data",index_col=None, sep="\t", encoding = 'utf-8')
movie['movie_title'] = movie['movie_title'].apply(lambda x: x.replace('\xa0', '' ))
def cosine(df) :
    X = cosine_similarity(df, df)
    X = pd.DataFrame(X)
    X = pd.concat([X, movie['movie_title']], axis=1)
    return(X)
df = cosine(data)

from json import JSONEncoder 
class MyEncoder(JSONEncoder): 
    def default(self, o): 
        print(o.__dict__)
        return o.__dict__

def recommendation_2(title):
    
    # Get the index of the movie that matches the title
    indices = pd.Series(df.index, index=df['movie_title'])
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(df[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    s = movie['movie_title'].iloc[movie_indices]
    s = pd.DataFrame(s)
    # Return the top 10 most similar movies
    return s

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world !"

@app.route('/predict', methods=['POST'])
def prediction():
    msguser = request.get_json()
    print("user request", msguser)

    if request.method=='POST':
        #try :
        film=msguser['film']
        #print('film :' ,film, type(film))
        
        if(not film):
            print("Please send the title of the film")

        result= recommendation_2(film)
        #print("result : ", result)

        #result = {"result": result}
        #result_json = json.dumps(result, cls = MyEncoder)

        #print("result_json: ", result.to_json(orient = 'index'))
        return  result.to_json(orient = 'index')
 
        #except Exception as e:
           # print(e)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)