import numpy as np
#from waitress import serve
from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('temp.pkl', 'rb'))

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

ipl = pd.read_csv('IPL_Matches_2008-2020.csv')
ipl['team1']=ipl['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
ipl['team2']=ipl['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
ipl['winner']=ipl['winner'].str.replace('Delhi Daredevils','Delhi Capitals')
ipl['toss_winner']=ipl['toss_winner'].str.replace('Delhi Daredevils','Delhi Capitals')
index_names = ipl[(ipl.team1 == 'Deccan Chargers') | (ipl.team2 == 'Deccan Chargers') | (ipl.winner == 'Deccan Chargers') | (ipl.toss_winner == 'Deccan Chargers')].index
ipl.drop(index_names, inplace = True)
ipl.drop(["date","id", "player_of_match", 'umpire1', "venue", "umpire2","result","result_margin","eliminator","method"], axis=1, inplace=True)
X = ipl.drop(["winner"], axis=1)
y = ipl["winner"]
X = pd.get_dummies(X, ["city", "team1","team2", "toss_winner", "toss_decision"], drop_first = True)

le = LabelEncoder()
y = le.fit_transform(y)

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [(x) for x in request.form.values()] #Convert string inputs to float.
    user_input = pd.DataFrame(data = [int_features], columns = ['city', 'neutral_venue', 'team1', 'team2', 'toss_winner','toss_decision'])
    user_input = pd.get_dummies(user_input)
    user_input = user_input.reindex(columns = X.columns, fill_value=0)
    pred = model.predict(user_input)
    prediction = le.inverse_transform(pred)
    # features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    
    #output = round(prediction[0], 2)
    # prediction = model.predict(int_features)  # features Must be in the form [[a, b]]
    return render_template('index.html', prediction_text='-------------------------------------------------------------------------------------------------------------> Prediction Of The Model is {} <-------------------------------------------------------------------------------------------------'.format(prediction))
    #print(prediction)


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    
    #serve(app, host="0.0.0.0", port=8080)
    app.run()
