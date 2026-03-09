"""
Watch trained agent driving.
Usage: Run in PyCharm or: python watch.py
"""

from stable_baselines3 import PPO
from ai.racing_env import RacingEnv
from core.lap_timer import LapTimer
import pygame


# === SETTINGS ===
MODEL_PATH = "models/v6/racing_ppo_final.zip"
TRACK_PATH = "tracks/track.png"


def main():
    print(f"Model: {MODEL_PATH}")
    print(f"Track: {TRACK_PATH}")

    model = PPO.load(MODEL_PATH)
    env = RacingEnv(track_file=TRACK_PATH, render_mode="human", max_steps=999999)

    obs, info = env.reset()

    # Initialize pygame window manually (without drawing raycasts)
    pygame.init()
    env._screen = pygame.display.set_mode((env._track._width, env._track._height))
    pygame.display.set_caption("Racing AI")
    env._clock = pygame.time.Clock()
    from core.renderer import Renderer
    env._renderer = Renderer(env._screen)

    font = pygame.font.Font(None, 36)

    total_laps = 0
    show_rays = False
    running = True
    clock = pygame.time.Clock()

    lap_timer = LapTimer()
    lap_timer.start_race()

    print("\n=== AGENT RUNNING ===")
    print("R = toggle raycasts")
    print("ESC = quit\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    show_rays = not show_rays

        if not running:
            break

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        lap_timer.update()

        current_laps = info.get('laps', 0)
        if current_laps > total_laps:
            lap_info = lap_timer.complete_lap()
            total_laps = current_laps
            lap_time_str = lap_timer.format_time(lap_info['time'])
            best_str = lap_timer.format_time(lap_timer.best_lap_time)
            best_mark = " (BEST!)" if lap_info['is_best'] else ""
            print(f"  Lap {lap_info['lap_number']}: {lap_time_str}{best_mark}  |  Best: {best_str}")

        # Render
        if env._screen and env._renderer:
            env._renderer.clear(env._track.background_color)
            env._renderer.draw_track(env._track, show_checkpoints=True)
            env._renderer.draw_vehicle(env._car)

            if show_rays:
                endpoints = getattr(env, '_last_raycast_endpoints', None)
                if endpoints is None:
                    _, endpoints = env._car.get_raycasts(env._track, env._max_raycast_distance)
                env._renderer.draw_raycasts(env._car, endpoints)

            # Stats overlay - line 1
            cp = info.get('checkpoint', 0)
            total_cp = info.get('total_checkpoints', 3)
            ray_status = "ON" if show_rays else "OFF"
            line1 = f"Laps: {total_laps} | CP: {cp}/{total_cp} | Rays: {ray_status} (R) | ESC=quit"

            # Stats overlay - line 2 (lap times)
            current_time_str = lap_timer.format_time(lap_timer.current_lap_time)
            best_time_str = lap_timer.format_time(lap_timer.best_lap_time)
            last_time_str = lap_timer.format_time(lap_timer.last_lap_time)
            line2 = f"Time: {current_time_str} | Best: {best_time_str} | Last: {last_time_str}"

            line1_surface = font.render(line1, True, (255, 255, 255))
            line2_surface = font.render(line2, True, (200, 255, 200))

            max_w = max(line1_surface.get_width(), line2_surface.get_width())
            bg = pygame.Rect(5, 5, max_w + 10, 55)
            pygame.draw.rect(env._screen, (0, 0, 0), bg)
            env._screen.blit(line1_surface, (10, 8))
            env._screen.blit(line2_surface, (10, 33))

            pygame.display.flip()

        clock.tick(60)

    # Print summary
    if lap_timer.lap_history:
        print(f"\n=== LAP TIMES ===")
        for i, t in enumerate(lap_timer.lap_history, 1):
            print(f"  Lap {i}: {lap_timer.format_time(t)}")
        print(f"  Best:  {lap_timer.format_time(lap_timer.best_lap_time)}")
        print(f"=================")

    env.close()
    print(f"\nDone. Laps: {total_laps}")


if __name__ == "__main__":
    main()

