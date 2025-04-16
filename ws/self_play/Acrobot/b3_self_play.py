import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard

# Create the Acrobot environment
env = gym.make('Acrobot-v1')  # Use Acrobot-v1 for earlier Gym versions

# Initialize the environment
env.reset()

# Set the initial observation to the specific state you provided
observation = np.array([0.83997925, 0.54261852, 0.53466358, -0.845065, -0.21872832, -1.45425407])
env.state = np.array([
    np.arctan2(observation[1], observation[0]),  # θ₁
    np.arctan2(observation[3], observation[2]),  # θ₂
    observation[4],  # θ₁_dot
    observation[5]   # θ₂_dot
])

# Initialize step counter
step_num = 0

# Variable to store the current action
current_action = 0  # Default action

# Function to handle key press
def on_press(key):
    global current_action
    try:
        if key == keyboard.Key.left:
            current_action = 1
        elif key == keyboard.Key.down:
            current_action = 0
        elif key == keyboard.Key.right:
            current_action = 2
    except AttributeError:
        pass

# Function to handle key release (optional, not used here since you don't want to release)
def on_release(key):
    global current_action
    # Reset to default action when key is released (optional)
    # current_action = 0
    pass

# Start listening for key presses
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Enable interactive mode for real-time plotting
plt.ion()

# Create a figure for plotting
fig = plt.figure(figsize=(9, 6), dpi=80)
ax = fig.add_subplot(111)
ax.set_axis_off()

# Initial render and plot setup
frame = env.render(mode='rgb_array')  # Render the environment
img = ax.imshow(frame)  # Display the initial frame
text_obj = ax.text(-150, 160, "", fontsize=10, color='white')  # Placeholder for text

# Main loop
while True:
    # Use the current action based on key presses
    action = current_action

    # Take a step in the environment
    observation, reward, done, info = env.step(action)

    # Print the step information
    print(f"Step number: {step_num}")
    print('action:', action)
    print('observation:', observation)
    print('reward:', reward)
    print('done:', done)
    print('info:', info)
    print('---------------------------\n')

    # Render the environment
    frame = env.render(mode='rgb_array')

    # Update the plot
    img.set_data(frame)  # Update the image data

    # Update the text
    text = f"Step number: {step_num}\n" \
           f"action: {action}\n" \
           f"observation: {np.round(observation, 4)}\n" \
           f"reward: {reward}\n" \
           f"done: {done}\n" \
           f"info: {info}\n" \
           "---------------------------\n"
    text_obj.set_text(text)  # Update the text on the plot

    # Refresh the plot
    plt.draw()
    plt.pause(0.01)  # Small pause to allow the plot to update

    # Increment the step counter
    step_num += 1

    # Check if the episode is done
    if done:
        print("Episode finished. Resetting environment...")
        env.reset()
        step_num = 0

    # Wait for 0.1 seconds before the next update
    time.sleep(0.1)

# Close the environment (this line won’t be reached unless you break the loop)
env.close()
listener.stop()