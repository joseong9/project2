import flask
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import re
from urllib.parse import urlsplit, parse_qsl



queryTypeRe = re.compile("^[jJ][sS][oO][nN]$")
queryTypeRe = re.compile("^[xX][mM][lL]$")
indexRunPriority = {'type': 1000, 'modeltype': 4}


app = Flask(__name__)
    
@app.route('/node/flask/o/predict', methods=['POST','GET'])
def predict():
   query_string = urlsplit(request.url).query
   parsed = dict(parse_qsl(query_string))
   params = request.get_json()
   
   runQueue = list()
   for _ in parsed.keys():
      runQueue.append((indexRunPriority[_], _))
   runQueue = sorted(runQueue, key=lambda x: x[0])
   
   for _ in runQueue:
      if _[1] == "type":
         if parsed[_[1]] == "json":
            response = dict()
            response.update("Blood Sugar Disease", result)
            return jsonify(response)
         elif parsed[_[1]] == "xml":
            response = dict()
            response.update("Blood Sugar Disease", result)
      elif _[1] == "modeltype":
         if parsed[_[1]] == "cls":
            model = tf.keras.models.load_model("./best-model_2.h5")
         elif parsed[_[1]] == "reg":
            model = tf.keras.models.load_model("./best-model_regress_9.h5")
            result = model.predic()



if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)
