from flask import Flask, request, jsonify, render_template
#import jsonify
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame

app = Flask(__name__)

model = pickle.load(open('Model11.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        int_features = (request.form['first_text'])
    
        def split(int_features): 
            return [char for char in int_features] 
    
        a = (split(int_features))
        print(a)

        df = pd.DataFrame(a)
        print (df)

        numerical_df = pd.get_dummies(df,drop_first=True)
        print(numerical_df)


        df1 = numerical_df.transpose()
        print(df1)


        df2 = df1.values.tolist()
        print(df2)

# import chain 
        from itertools import chain 

# converting 3d list into 1d 
# using chain.from_iterables 
        final_result = [list(chain.from_iterable(df2))]
        print(final_result)
  

        prediction = model.predict(final_result) 
        output=prediction[0]
        

        if output == 0:
            return render_template('index.html',prediction_text="Group is promoter")
        else:
            return render_template('index.html',prediction_text="Group is non-promoter")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
    
    
    
    
  