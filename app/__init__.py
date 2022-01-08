from flask import Flask

# Init flask
from app.linearRegressionService import calculateLinearRegression, trainLinearRegression

app = Flask(__name__)

@app.route("/")
def hello_world():
    prediction = calculateLinearRegression()
    return f"Predicted probabilities, {prediction}!"

if __name__ == '__main__':
    trainLinearRegression()
    app.run()

