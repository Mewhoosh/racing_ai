"""
Manual play mode - control vehicle with WASD keys.
Usage: python main.py
"""
from core.game_engine import GameEngine


def main():
    track_file = "tracks/track.png"
    game = GameEngine(track_file=track_file)
    game.run()


if __name__ == "__main__":
    main()
