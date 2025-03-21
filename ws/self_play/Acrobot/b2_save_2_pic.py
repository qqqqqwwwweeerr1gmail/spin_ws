import gym
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure


# Create the Acrobot environment
env = gym.make('Acrobot-v1')  # Use Acrobot-v1 for earlier Gym versions

# Initialize the environment
env.reset()

# cos(θ₁): Cosine of the angle of the first link relative to the vertical (downward position).
# sin(θ₁): Sine of the angle of the first link relative to the vertical.
# cos(θ₂): Cosine of the angle of the second link relative to the first link.
# sin(θ₂): Sine of the angle of the second link relative to the first link.
# θ₁_dot: Angular velocity of the first link.
# θ₂_dot: Angular velocity of the second link.

step_num = 0

# Main loop
for _ in range(100):
    frame = env.render(mode='rgb_array')  # Render the environment
    action = env.action_space.sample()  # Sample random actions
    observation, reward, done, info = env.step(action)  # Take a step based on the action

    print('action:', action)
    print('observation:', observation)
    print('reward:', reward)
    print('done:', done)
    print('info:', info)
    print('---------------------------\n')

    # print(frame)
    step_num = _
    figure(figsize=(8, 6), dpi=80)
    image = Image.fromarray(frame)

    text = f"Step number: {step_num}\n" \
           f"action: {action}\n" \
           f"observation: {observation}\n" \
           f"reward: {reward}\n" \
           f"done: {done}\n" \
           f"info: {info}\n" \
           "---------------------------\n"

    plt.text(-150, 160, text, fontsize=10)

    image.save(f'./img/acrobot_{step_num}.png')
    print("Image saved as 'acrobot_rendered.png'")

    # Optionally, display the image using matplotlib (for verification)
    plt.imshow(frame)
    plt.axis('off')
    plt.show()

    time.sleep(0.1)
    time.sleep(1)

env.close()  # Close the environment