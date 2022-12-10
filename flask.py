import flask
import json
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

def on_json_loading_failed_return_dict(e):
   return {}
    
@app.route('/node/flask/o/predict', methods=['POST','GET'])
def predict():
   query_string = urlsplit(request.url).query
   parsed = dict(parse_qsl(query_string))
   json_data = request.get_json()
   request.on_json_loading_failed = on_json_loading_failed_return_dict
   
   runQueue = list()
   for _ in parsed.keys():
      runQueue.append((indexRunPriority[_], _))
   runQueue = sorted(runQueue, key=lambda x : x[0])
   
   HR = json_data.get('HR')
   HRV = json_data.get('HRV')
   SDNN = json_data.get('SDNN')
   RMSSD = json_data.get('RMSSD')
   PNN50 = json_data.get('PNN50')
   VLF = json_data.get('VLF')
   LF = json_data.get('LF')
   HF = json_data.get('HF')
   gender = json_data.get('gender')
   age = json_data.get('age')
   x = [[HR, HRV, SDNN, RMSSD, PNN50, VLF, LF, HF, gender, age]]
   
   
   for _ in runQueue:
      if _[1] == "type":
         if parsed[_[1]] == "json":
            pythonTypeResult1 = float(result)
            pythonTypeResult2 = float(result)            
            response = dict()
            if pythonTypeResult1 == 0:
               response.update({"당뇨입니까?" : "당뇨입니다. 관리가 필요합니다!"})
            elif pythonTypeResult1 == 1:
               response.update({"당뇨입니까?" : "당뇨가 아닙니다."})
            return jsonify(response)
         elif parsed[_[1]] == "xml":
            response = dict()
            if pythonTypeResult1 == 0.0:
               response.update({"당뇨입니까?" : "네"})
            elif pythonTypeResult1 == 1.0:
               response.update({"당뇨입니까?" : "아니요"})
            return jsonify(response)
      elif _[1] == "modeltype":
         if parsed[_[1]] == "cls":
            model = tf.keras.models.load_model("./best-model_sigmoid.h5")
            result = model.predict(x)
         elif parsed[_[1]] == "lin":
            model = tf.keras.models.load_model("./best-model_linear.h5")
            result2 = model.predict(x)


   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
