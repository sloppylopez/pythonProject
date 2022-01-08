from flask import Flask, render_template, jsonify
from linearRegressionService import calculateLinearRegression, trainLinearRegression
import os
basedir = os.path.abspath(os.path.dirname(__file__))
# Init flask
app = Flask(__name__, template_folder='./templates', static_folder='./static')

@app.route("/heartbeat")
def heartbeat():
    return jsonify({"status": "healthy"})

@app.route("/hello_world")
def hello_world():
    calculateLinearRegression()
    return render_template('linear_regression.html', name='new_plot',
                           url='/static/images/new_plot.png')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return app.send_static_file("index.html")

if __name__ == '__main__':
    trainLinearRegression()
    app.run()