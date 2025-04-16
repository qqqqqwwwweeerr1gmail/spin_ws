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
    Dense(16, input_shape=(6,), activation='relu'),  # 6 inputs for Acrobot observations
    Dense(8, activation='binary_not'),  # Use the custom activation
    Dense(3, activation='softmax')  # 3 outputs for Acrobot actions (0, 1, 2)
])

# Create the Acrobot environment
env = gym.make('Acrobot-v1')
observation = env.reset()

# Main loop
for _ in range(100):
    # Ensure the observation contains only 0s and 1s (for this example, binarize it)
    observation = np.where(observation > 0, 1, 0).astype(np.float32)  # Binarize the observation
    observation = observation.reshape(1, -1)  # Reshape for the model

    # Predict action probabilities
    action_probs = model.predict(observation, verbose=0)
    action = np.argmax(action_probs[0])  # Choose the action with the highest probability

    # Take a step in the environment
    observation, reward, done, info = env.step(action)
    print(f"Step number: {_}")
    print('action:', action)
    print('observation:', observation)
    print('reward:', reward)
    print('done:', done)
    print('info:', info)
    print('---------------------------\n')
    if done:
        observation = env.reset()

env.close()























