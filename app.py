from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import csv
app = Flask(__name__)
data = pd.read_csv('Student_Performance.csv')
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours']]
y = data['Performance Index']
num_splits = 10
kf = KFold(n_splits=num_splits)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

def get_performance_suggestion(performance_index):
    if performance_index >= 90:
        return "Peforma anda sangat baik! Pertahankan peforma anda saat ini"
    elif performance_index >= 70:
        return "Peforma anda baik. Anda berada di jalur yang tepat."
    elif performance_index >= 50:
        return "Peforma anda rata-rata. Pertimbangkan untuk belajar lebih banyak untuk memperbaiki."
    else:
        return "Peforma Anda perlu ditingkatkan. Coba belajar lebih banyak dan mencari bantuan jika diperlukan."

@app.route('/')
def main():
    global data
    shape = data.shape
    return render_template('index.html', baris=shape[0])
@app.route('/data')
def main3():
    data = []

    with open('Student_Performance.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Ambil hanya kolom yang Anda inginkan
            filtered_data = {
                "Hours Studied": row["Hours Studied"],
                "Previous Scores": row["Previous Scores"],
                "Extracurricular Activities": row["Extracurricular Activities"],
                "Sleep Hours": row["Sleep Hours"],
                "Performance Index": row["Performance Index"]
            }
            data.append(filtered_data)

    return render_template('data.html', data=data)
@app.route('/home')
def main2():
    global data
    shape = data.shape
    return render_template('index.html', baris=shape[0])
@app.route('/input')
def input():
    global data
    shape = data.shape
    return render_template('predict.html', baris=shape[0])
@app.route('/predict', methods=['POST'])
def predict():
    global data 
    shape = data.shape
    if request.method == 'POST':
        hours_studied = float(request.form['hours_studied'])
        previous_scores = float(request.form['previous_scores'])
        extracurricular_activities = int(request.form['extracurricular_activities'])
        sleep_hours = float(request.form['sleep_hours'])
        data_to_predict = {
            'Hours Studied': [hours_studied],
            'Previous Scores': [previous_scores],
            'Extracurricular Activities': [extracurricular_activities],
            'Sleep Hours': [sleep_hours]
        }
        data_to_predict_df = pd.DataFrame(data_to_predict)
        predicted_performance_index = model.predict(data_to_predict_df)
        rounded_performance_index = round(predicted_performance_index[0])
        suggestion = get_performance_suggestion(rounded_performance_index)
        data_to_add = {
            'Hours Studied': [hours_studied],
            'Previous Scores': [previous_scores],
            'Extracurricular Activities': [extracurricular_activities],
            'Sleep Hours': [sleep_hours],
            'Sample Question Papers Practiced': [1],
            'Performance Index': rounded_performance_index
        }
        data_to_add_df = pd.DataFrame(data_to_add)
        data = pd.concat([data, data_to_add_df], ignore_index=True)
        data['Extracurricular Activities'] = data['Extracurricular Activities'].map({1: 'Yes', 0: 'No'})
        data.to_csv('Student_Performance.csv', index=False)
        return render_template('result.html', prediction=predicted_performance_index[0],prediction2=rounded_performance_index, baris=shape[0], suggest=suggestion)
if __name__ == '__main__':
    app.run(debug=True)
