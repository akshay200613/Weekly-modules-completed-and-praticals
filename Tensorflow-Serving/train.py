import tensorflow as tf 

# Load MNIST dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Normailze
x_train = x_train / 255.0

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

# compile 
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train 
model.fit(x_train, y_train, epochs=2)

# Save model
model.export("models/mnist/1")

print("Model saved sucessfully")
