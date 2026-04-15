import sys
import time
import pygame
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401; needed to register envs


def make_env():
    env = gym.make("highway-v0", render_mode="rgb_array")
    # Tweak config if desired
    env.configure({
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 60,                # episode length (seconds of simulated time)
        "policy_frequency": 15,        # control frequency (Hz)
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction"},  # we’ll map keys to meta actions
    })
    return env


def get_action_index_by_name(env, name):
    # Robustly map action names -> indices for DiscreteMetaAction
    # highway-env exposes names on the action_type
    at = env.unwrapped.action_type
    if hasattr(at, "actions_indexes") and name in at.actions_indexes:
        return at.actions_indexes[name]
    # Fallback: try to get names from action_type.actions if available
    if hasattr(at, "actions") and isinstance(at.actions, (list, tuple)):
        for idx, a in enumerate(at.actions):
            if isinstance(a, str) and a.upper() == name.upper():
                return idx
    raise KeyError(f"Action name '{name}' not found in env action space.")


def main():
    env = make_env()
    obs, info = env.reset()

    # Init pygame
    pygame.init()
    pygame.display.set_caption("Manual play: highway-v0")
    clock = pygame.time.Clock()
    target_fps = 30

    # Build window once we know frame size
    frame = env.render()
    if frame is None:
        # some renderers require a first call after reset/step
        frame = env.render()
    h, w, _ = frame.shape
    scale = 1.0  # adjust if you want larger window
    win_w, win_h = int(w * scale), int(h * scale)
    screen = pygame.display.set_mode((win_w, win_h))

    # Discover action indices by name
    try:
        A_IDLE = get_action_index_by_name(env, "IDLE")
        A_LEFT = get_action_index_by_name(env, "LANE_LEFT")
        A_RIGHT = get_action_index_by_name(env, "LANE_RIGHT")
        A_FASTER = get_action_index_by_name(env, "FASTER")
        A_SLOWER = get_action_index_by_name(env, "SLOWER")
    except KeyError as e:
        print("Could not map action names. Ensure action type is DiscreteMetaAction.")
        print(e)
        env.close()
        pygame.quit()
        sys.exit(1)

    # Default action
    current_action = A_IDLE
    paused = False
    episode_reward = 0.0
    ep = 0

    def draw_frame(arr, info_text=""):
        surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
        if (win_w, win_h) != (arr.shape[1], arr.shape[0]):
            surf = pygame.transform.smoothscale(surf, (win_w, win_h))
        screen.blit(surf, (0, 0))

        if info_text:
            # Simple overlay text
            font = pygame.font.SysFont("Arial", 16)
            text_surf = font.render(info_text, True, (255, 255, 255))
            screen.blit(text_surf, (10, 10))

        pygame.display.flip()

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    episode_reward = 0.0
                    current_action = A_IDLE
                elif event.key == pygame.K_SPACE:
                    current_action = A_IDLE

        # Continuous key state for directional controls
        keys = pygame.key.get_pressed()
        if not paused:
            if keys[pygame.K_LEFT]:
                current_action = A_LEFT
            elif keys[pygame.K_RIGHT]:
                current_action = A_RIGHT
            elif keys[pygame.K_UP]:
                current_action = A_FASTER
            elif keys[pygame.K_DOWN]:
                current_action = A_SLOWER
            else:
                # If no directional key is pressed, keep last chosen action,
                # or uncomment the next line to revert to IDLE:
                # current_action = A_IDLE
                pass

            obs, reward, terminated, truncated, info = env.step(current_action)
            episode_reward += reward
            done = terminated or truncated
            frame = env.render()

            if done:
                ep += 1
                print(f"Episode {ep} reward: {episode_reward:.2f}")
                obs, info = env.reset()
                episode_reward = 0.0
                current_action = A_IDLE
        else:
            # Still render latest frame when paused
            frame = env.render()

        info_text = f"Ep: {ep}  Reward: {episode_reward:.2f}  Action: {current_action}  Paused: {paused}"
        if frame is not None:
            draw_frame(frame, info_text)

        clock.tick(target_fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
























