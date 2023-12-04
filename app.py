import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

data = pd.read_csv('train dataset.csv')

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

input_cols = ['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
output_cols = ['Personality (Class label)']

scaler = StandardScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])

X = data[input_cols]
Y = data[output_cols]

model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
model.fit(X, Y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def result():
    if request.method == 'POST':
        gender = request.form['gender']
        gender_no = 1 if gender == "Female" else 2
        # if age<17:
        #   alert("Age cant be less than 17")
        age = float(request.form['age'])
        openness = float(request.form['openness'])
        neuroticism = float(request.form['neuroticism'])
        conscientiousness = float(request.form['conscientiousness'])
        agreeableness = float(request.form['agreeableness'])
        extraversion = float(request.form['extraversion'])

        result = np.array([gender_no, age, openness, neuroticism, conscientiousness, agreeableness, extraversion], ndmin=2)

        # Use the same scaler fitted on the training data to transform user input
        result_scaled = [np.float64(x) for x in result]
        print(result_scaled)
        personality = str(model.predict(result_scaled)[0])

        return render_template("submit.html", answer=personality)

if __name__ == '__main__':
    app.run()
