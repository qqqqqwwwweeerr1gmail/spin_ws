import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import get_custom_objects


# Define the custom activation function (as shown above)
def binary_not_activation_tf(x):
    tf.debugging.assert_equal(
        tf.reduce_min(tf.cast(tf.logical_or(x == 0, x == 1), tf.int32)),
        1,
        message="All input values must be 0 or 1"
    )
    return 1 - x


# Register the custom activation with Keras
get_custom_objects().update({'binary_not': Activation(binary_not_activation_tf)})

# Create a simple neural network model for Acrobot
model = tf.keras.Sequential([
    Dense(1, input_shape=(1,), activation='binary_not'),  # 6 inputs for Acrobot observations
])

observation = [1]
action_probs = model.predict(observation, verbose=0)

print(action_probs)

model.layers
print(model.layers)




















