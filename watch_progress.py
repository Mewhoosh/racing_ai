"""
Watch training progress - compare models from different epochs.
Usage: Run in PyCharm or: python watch_progress.py
"""

from stable_baselines3 import PPO
from ai.racing_env import RacingEnv
import os
import pygame
import glob


# === SETTINGS ===
MODEL_DIR = "models/v6"
TRACK_PATH = "tracks/track.png"
SKIP_EVERY = 2  # Show every N-th model


def get_checkpoint_models(model_dir):
    """Find checkpoint models sorted by steps."""
    pattern = os.path.join(model_dir, "racing_ppo_*_steps.zip")
    files = glob.glob(pattern)

    def get_steps(f):
        try:
            return int(os.path.basename(f).split("_")[-2])
        except:
            return 0

    files.sort(key=get_steps)
    return files


def run_episode(model, env, model_name, current_idx, total_models, max_steps=500):
    """Run one episode and return results."""
    obs, info = env.reset()
    env.render()  # Initialize pygame window

    total_reward = 0
    steps = 0

    pygame.font.init()
    font = pygame.font.Font(None, 36)

    while steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None, True, False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return total_reward, info.get('checkpoint', 0), steps, False, True

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Render without raycasts
        if env._screen and env._renderer:
            env._renderer.clear(env._track.background_color)
            env._renderer.draw_track(env._track, show_checkpoints=True)
            env._renderer.draw_vehicle(env._car)

            text = f"[{current_idx}/{total_models}] {model_name} | Enter=skip"
            surf = font.render(text, True, (255, 255, 255))
            pygame.draw.rect(env._screen, (0, 0, 0), (5, 5, surf.get_width() + 10, 30))
            env._screen.blit(surf, (10, 10))
            pygame.display.flip()
            env._clock.tick(60)

        if terminated or truncated:
            break

    return total_reward, info.get('checkpoint', 0), steps, False, False


def main():
    models = get_checkpoint_models(MODEL_DIR)

    if not models:
        print(f"No models in {MODEL_DIR}")
        return

    models = models[::SKIP_EVERY]

    final = os.path.join(MODEL_DIR, "racing_ppo_final.zip")
    if os.path.exists(final) and final not in models:
        models.append(final)

    print(f"Found {len(models)} models in {MODEL_DIR}")
    print(f"Track: {TRACK_PATH}")

    env = RacingEnv(track_file=TRACK_PATH, render_mode="human")

    print("\n=== TRAINING PROGRESS ===")
    print("Enter = skip to next model\n")

    results = []
    for i, model_path in enumerate(models):
        name = os.path.basename(model_path).replace(".zip", "")
        print(f"[{i+1}/{len(models)}] {name}")

        model = PPO.load(model_path)
        reward, cp, steps, quit_flag, skipped = run_episode(
            model, env, name, i+1, len(models)
        )

        if quit_flag:
            break

        results.append((name, reward, cp, steps))
        status = "(skipped)" if skipped else ""
        print(f"  Reward: {reward:.1f}, CP: {cp}, Steps: {steps} {status}")

        pygame.time.wait(300)

    env.close()

    print("\n=== SUMMARY ===")
    print(f"{'Model':<30} {'Reward':>10} {'CP':>5} {'Steps':>7}")
    print("-" * 55)
    for name, reward, cp, steps in results:
        print(f"{name:<30} {reward:>10.1f} {cp:>5} {steps:>7}")


if __name__ == "__main__":
    main()

