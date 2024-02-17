import tensorflow as tf
import numpy as np
from flask import Flask,request, jsonify
#from flask_ngrok import run_with_ngrok
from flask_cors import CORS, cross_origin
import pandas as pd
model = tf.keras.models.load_model('C:/Users/Nethika/Desktop/New folder (2)/model.h5')


app =Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})
# run_with_ngrok(app)


@app.route('/')
def index():
    return jsonify("Welcome to the home page")

@app.route('/classify_grade', methods=['GET'])
def upload_image():
  KW = request.args.get('KW')
  LS = request.args.get('LS')
  CO = request.args.get('CO')
  PS = request.args.get('PS')
  DM = request.args.get('DM')
  CR = request.args.get('CR')
  TW = request.args.get('TW')
  print(KW)
  print(type(KW))
  
  
  data = {
      'KW': float(KW),
      'LS': float(LS),
      'CO': float(CO),
      'PS': float(PS),
      'DM': float(DM),
      'CR': float(CR),
      'TW': float(TW)
  }

  # Provide a list of row labels (e.g., ['Row1'])
  row_labels = ['Row1']

  # Create DataFrame with specified index
  dfX = pd.DataFrame(data, index=row_labels)
  input_data = dfX.to_numpy()
  # Make predictions on the new data
  predictions = model.predict(input_data)

  # Print the predictions
  for i, prediction in enumerate(predictions):
      print(f"Prediction for data point {i+1}: {prediction}")


  predicted_classes = np.argmax(predictions, axis=None)
  if predicted_classes == 0:
      print('better')
      prediction_final='better'
  elif predicted_classes == 1:
      print('easy')
      prediction_final='easy'
  elif predicted_classes == 2:
      print('hard')
      prediction_final='hard'
  return jsonify(prediction_final)

app.run()