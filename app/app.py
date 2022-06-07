from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as ps
import plotly.graph_objects as go
import uuid


app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    request_type_str = request.method
        

    if request_type_str == 'GET':
        path = 'static/base_pic.svg'
        return render_template('index.html', href = path )
        
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = 'app/static/' + random_string + '.svg'
        model = load('app/model.joblib')
        np_arr = floats_string_to_np_arr(text)
        make_picture('app/AgesAndHeights.pkl', model, np_arr, path )

        
        return render_template('index.html', href = path[4:])


def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
  data = pd.read_pickle(training_data_filename)
  ages = data.iloc[:,0]      
  heights = data.iloc[:,-1] 
  data = data[ages > 0]
  ages = data.iloc[:,0]      
  heights = data.iloc[:,-1] 
  x_new = np.array(list(range(19))).reshape(19,1)
  pred = model.predict(x_new)

  fig = ps.scatter(x=ages, y = heights, title="Age Vs Height", labels={'x' : "Age (Years)", 
                                                               "y" : "Height (in Inches)"})
  fig.add_trace(go.Scatter(x = x_new.reshape(19), y = pred, mode = 'lines', name = 'Model'))

  new_preds = model.predict(new_inp_np_arr)

  fig.add_trace(go.Scatter( x = new_inp_np_arr.reshape(len(new_inp_np_arr)), y = new_preds, name = 'New Outputs',  mode = 'markers', marker = dict(color = 'purple', size = 20)))

  fig.write_image(output_file, width = 800, engine = 'kaleido')
  fig.show()


def floats_string_to_np_arr(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)