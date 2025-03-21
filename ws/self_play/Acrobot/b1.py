import gym
import time

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



# Main loop
for _ in range(100):
    env.render()  # Render the environment
    action = env.action_space.sample()  # Sample random actions
    observation, reward, done, info = env.step(action)  # Take a step based on the action

    print('action:', action)
    print('observation:', observation)
    print('reward:', reward)
    print('done:', done)
    print('info:', info)
    print('---------------------------\n')
    time.sleep(0.1)
    time.sleep(1)

env.render()
env.close()  # Close the environment