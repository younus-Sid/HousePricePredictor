from flask import Flask, request, render_template
import pandas as pd
import pickle, locale


app = Flask(__name__)
locale.setlocale(locale.LC_MONETARY, 'en_IN')

# Loading all the locations in various cities
data_banga = pd.read_csv("Database/PureHouseData_Bangalore.csv")
data_delhi = pd.read_csv("Database/PureHouseData_Delhi.csv")
data_pune = pd.read_csv("Database/PureHouseData_Pune.csv")

# Loading all the trained models
pipe_banga = pickle.load(open("Database/TrainedModel_Bangalore.pkl", "rb"))
pipe_delhi = pickle.load(open("Database/TrainedModel_Delhi.pkl", "rb"))
pipe_pune = pickle.load(open("Database/TrainedModel_Pune.pkl", "rb"))

@app.route('/')
def index():
    all_locations = {
        "Bangalore": sorted(data_banga.location.unique()),
        "Delhi": sorted(data_delhi.location.unique()),
        "Pune": sorted(data_pune.location.unique()),
    }
    return render_template("index.html", cities=all_locations)


@app.route('/predict', methods=['POST'])
def predict():
    city = request.form.get("city")
    location = request.form.get("location")
    bhk = request.form.get("bhk")
    bath = request.form.get("bath")
    area = request.form.get("area")

    input = pd.DataFrame([[location, area, bath, bhk]], columns=['location', 'area', 'bath', 'bhk'])
    if city == "Bangalore":
        output = pipe_banga.predict(input)[0]
    elif city == "Delhi":
        output = pipe_delhi.predict(input)[0]
    elif city == "Pune":
        output = pipe_pune.predict(input)[0]

    answer = round((output * 1e5), 2)
    return locale.currency(answer, grouping=True)


if __name__ == "__main__":
    app.run(debug=True)