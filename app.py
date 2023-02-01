from flask import Flask, request
import tensorflow as tf #importar tensorflow (libreria IA hecha por google)
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/hola", methods=['POST'])
def prediction():
    if request.method == 'POST':
        numeroApredecir = float(request.json['numero'])
        print(numeroApredecir)
        km = np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
        millas=np.array([0.621371,1.24274,1.86411,2.48548,3.10686,3.72823,4.3496,4.97097,5.59234,6.21371],dtype=float)

        capaoculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
        capaoculta2 = tf.keras.layers.Dense(units=3)
        capasalida = tf.keras.layers.Dense(units=1)
        modelo = tf.keras.Sequential([capaoculta1, capaoculta2, capasalida])
        
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss='mean_squared_error'
        )
        
        historial = modelo.fit(km, millas, epochs=1000, verbose=False)

        resultado = modelo.predict([numeroApredecir])
        print(resultado)
        
        return str(resultado)
if __name__ == "__main__":
    app.run(debug=True)