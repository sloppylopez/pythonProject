from flask import Flask, render_template, jsonify, request
import os
from services.deep_learning_service import trainDNN, getPrediction
from services.linear_regression_service import calculateLinearRegression, trainLinearRegression
from werkzeug.utils import secure_filename

basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# Init flask
app = Flask(__name__, template_folder='./templates', static_folder='./static')


# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/dnn_num_reader', methods=['GET', 'POST'])
def dnn_num_reader():
    if request.method == 'GET':
        return render_template('dnn_view.html')
    if request.method == 'POST':
        """modify/update the information for <user_id>"""
        # you can use <user_id>, which is a str but could
        # changed to be int or whatever you want, along
        # with your lxml knowledge to make the required
        # changes
        canvas_img = request.form.__getitem__('canvasImg') # a multidict containing POST data
        print(canvas_img)
        getPrediction(canvas_img)
        print('Skynet has become self-aware')
    else:
        # POST Error 405 Method Not Allowed
        """Error 405 Method Not Allowed"""


@app.route("/heartbeat")
def heartbeat():
    return jsonify({"status": "healthy"})


@app.route("/linear_regression")
def linear_regression():
    calculateLinearRegression()
    return render_template('linear_regression.html', name='new_plot',
                           url='/static/images/new_plot.png')


if __name__ == '__main__':
    trainLinearRegression()
    trainDNN()
    # app.run(debug=True)
    app.run()
