import tensorflow as tf 
import requests
import numpy as np 

# load test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_test = x_test / 255.0

sample = x_test[0].tolist()

url = "http://localhost:8501/v1/models/mnist:predict"

data = {
    "instances": [sample]
}

print("Request JSON:")
print(data)

response = requests.post(url, json=data)

print("n/Status Code:", response.status_code)
print("Response: ", response.text)

if response.status_code ==200:
    result = response.json()
    predicted = np.argmax(result["predictions"][0])

    print("Actual:", y_test[0])
    print("Predicted:", predicted)

else:
    print("Prediction failed.")