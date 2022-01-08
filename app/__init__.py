from flask import Flask
from app.linearRegressionService import calculateLinearRegression, trainLinearRegression

# Init flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    prediction = calculateLinearRegression()
    return f"Predicted probabilities, {prediction}!"

if __name__ == '__main__':
    trainLinearRegression()
    app.run()