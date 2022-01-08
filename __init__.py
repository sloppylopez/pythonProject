from flask import Flask, Response, render_template
from linearRegressionService import calculateLinearRegression, trainLinearRegression
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
basedir = os.path.abspath(os.path.dirname(__file__))
# Init flask
app = Flask(__name__, template_folder='./templates', static_folder='./static')

@app.route("/")
def hello_world():
    calculateLinearRegression()
    return render_template('linear_regression.html', name='new_plot',
                           url='/static/images/new_plot.png')

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

if __name__ == '__main__':
    trainLinearRegression()
    app.run()