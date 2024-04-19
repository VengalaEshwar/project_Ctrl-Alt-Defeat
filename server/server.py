from flask import Flask, request, jsonify
import model
# import your_ml_module

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(data)
    print("Handled the data" , data)
    print(data)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5001) 
