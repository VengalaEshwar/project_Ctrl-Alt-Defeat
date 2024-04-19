from flask import Flask, request, jsonify
# import your_ml_module

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data["some"] =[1,2,3,4]
    prediction =  data #your_ml_module.predict(data)  # Call your machine learning model function
    print("Handled the data")
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5001) 
