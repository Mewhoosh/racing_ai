"""
PPO Training script for racing game.
Usage: Run in PyCharm or: python train.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from ai.racing_env import RacingEnv
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === SETTINGS ===
SAVE_PATH = "models/v6"
TRACK_FILE = "tracks/track.png"
TOTAL_TIMESTEPS = 250000
N_ENVS = 8


def rolling_mean(data, window=50):
    """Compute rolling mean with given window size."""
    if len(data) < window:
        window = max(1, len(data))
    return np.convolve(data, np.ones(window) / window, mode='valid')


def rolling_std(data, window=50):
    """Compute rolling standard deviation with given window size."""
    if len(data) < window:
        window = max(1, len(data))
    result = []
    for i in range(len(data) - window + 1):
        result.append(np.std(data[i:i + window]))
    return np.array(result)


class TrainingLogger(BaseCallback):
    """Logs episode metrics and generates training plots."""

    def __init__(self, log_freq=8192, save_path=SAVE_PATH):
        super().__init__(verbose=1)
        self.log_freq = log_freq
        self.save_path = save_path

        # Raw per-episode data
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_checkpoints = []
        self.episode_laps = []
        self.episode_collisions = []

        # Snapshot data for console logs
        self.log_timesteps = []
        self.log_mean_rewards = []
        self.log_max_rewards = []
        self.log_mean_checkpoints = []
        self.log_mean_lengths = []
        self.log_success_rate = []
        self.log_mean_laps = []

    def _on_step(self):
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])

        for info, done in zip(infos, dones):
            if done:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                if 'checkpoint' in info:
                    self.episode_checkpoints.append(info['checkpoint'])
                if 'laps' in info:
                    self.episode_laps.append(info['laps'])
                if 'collisions' in info:
                    self.episode_collisions.append(info['collisions'])

        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            n = min(50, len(self.episode_rewards))
            recent_rew = self.episode_rewards[-n:]
            recent_len = self.episode_lengths[-n:] if self.episode_lengths else [0]
            recent_cp = self.episode_checkpoints[-n:] if self.episode_checkpoints else [0]
            recent_laps = self.episode_laps[-n:] if self.episode_laps else [0]

            mean_rew = np.mean(recent_rew)
            max_rew = np.max(recent_rew)
            std_rew = np.std(recent_rew)
            mean_len = np.mean(recent_len)
            mean_cp = np.mean(recent_cp)
            max_cp = int(np.max(recent_cp))
            mean_laps = np.mean(recent_laps)
            success_rate = np.mean([1 if cp >= 1 else 0 for cp in recent_cp]) * 100

            self.log_timesteps.append(self.num_timesteps)
            self.log_mean_rewards.append(mean_rew)
            self.log_max_rewards.append(max_rew)
            self.log_mean_checkpoints.append(mean_cp)
            self.log_mean_lengths.append(mean_len)
            self.log_success_rate.append(success_rate)
            self.log_mean_laps.append(mean_laps)

            print(f"\n=== [{self.num_timesteps:,}] ===")
            print(f"  Reward:      {mean_rew:.1f} +/- {std_rew:.1f} (max: {max_rew:.1f})")
            print(f"  Checkpoints: {mean_cp:.2f} (max: {max_cp})")
            print(f"  Laps:        {mean_laps:.2f}")
            print(f"  Ep. length:  {mean_len:.0f}")
            print(f"  Success:     {success_rate:.0f}%")

        return True

    def _on_training_end(self):
        """Save training plots and metrics JSON."""
        self._save_metrics_json()
        self._save_plots()

    def _save_metrics_json(self):
        """Save raw metrics to JSON for later analysis."""
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_checkpoints": self.episode_checkpoints,
            "episode_laps": self.episode_laps,
            "log_timesteps": self.log_timesteps,
            "log_mean_rewards": self.log_mean_rewards,
            "log_max_rewards": self.log_max_rewards,
            "log_mean_checkpoints": self.log_mean_checkpoints,
            "log_mean_lengths": self.log_mean_lengths,
            "log_success_rate": self.log_success_rate,
            "log_mean_laps": self.log_mean_laps,
        }
        json_path = os.path.join(self.save_path, "metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"Metrics saved: {json_path}")

    def _save_plots(self):
        """Generate publication-style training plots."""
        if len(self.episode_rewards) < 10:
            return

        window = 50
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

        # --- 1. Episode Return (mean +/- std, rolling) ---
        ax = axes[0, 0]
        if len(self.episode_rewards) >= window:
            mean = rolling_mean(self.episode_rewards, window)
            std = rolling_std(self.episode_rewards, window)
            x = np.arange(window - 1, len(self.episode_rewards))
            ax.plot(x, mean, color='#2196F3', linewidth=1.5, label=f'Mean (window={window})')
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='#2196F3')
        else:
            ax.plot(self.episode_rewards, color='#2196F3', linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Episode Return')
        ax.legend(loc='upper left', fontsize=9)

        # --- 2. Checkpoints per Episode (rolling) ---
        ax = axes[0, 1]
        if len(self.episode_checkpoints) >= window:
            mean = rolling_mean(self.episode_checkpoints, window)
            std = rolling_std(self.episode_checkpoints, window)
            x = np.arange(window - 1, len(self.episode_checkpoints))
            ax.plot(x, mean, color='#4CAF50', linewidth=1.5, label=f'Mean (window={window})')
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='#4CAF50')
        else:
            ax.plot(self.episode_checkpoints, color='#4CAF50', linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Checkpoints')
        ax.set_title('Checkpoints per Episode')
        ax.legend(loc='upper left', fontsize=9)

        # --- 3. Episode Length (rolling) ---
        ax = axes[1, 0]
        if len(self.episode_lengths) >= window:
            mean = rolling_mean(self.episode_lengths, window)
            std = rolling_std(self.episode_lengths, window)
            x = np.arange(window - 1, len(self.episode_lengths))
            ax.plot(x, mean, color='#FF9800', linewidth=1.5, label=f'Mean (window={window})')
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='#FF9800')
        else:
            ax.plot(self.episode_lengths, color='#FF9800', linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.legend(loc='upper left', fontsize=9)

        # --- 4. Success Rate (% episodes with >= 1 checkpoint) ---
        ax = axes[1, 1]
        if len(self.log_timesteps) >= 2:
            ax.plot(self.log_timesteps, self.log_success_rate,
                    color='#9C27B0', linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Success Rate')
            ax.set_ylim(-5, 105)
        else:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center',
                    transform=ax.transAxes)

        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "training_plot.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {plot_path}")


def make_env(track_file):
    def _init():
        env = RacingEnv(track_file=track_file, render_mode=None)
        return Monitor(env)
    return _init


def train():
    os.makedirs(SAVE_PATH, exist_ok=True)

    env = SubprocVecEnv([make_env(TRACK_FILE) for _ in range(N_ENVS)])
    env = VecMonitor(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // N_ENVS,
        save_path=SAVE_PATH,
        name_prefix="racing_ppo"
    )

    logger = TrainingLogger(log_freq=8192, save_path=SAVE_PATH)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1
    )

    print(f"Starting PPO training: {TOTAL_TIMESTEPS:,} steps")
    print(f"Parallel envs: {N_ENVS} (SubprocVecEnv)")
    print(f"Track: {TRACK_FILE}")
    print(f"Save path: {SAVE_PATH}/")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, logger]
    )

    final_path = os.path.join(SAVE_PATH, "racing_ppo_final")
    model.save(final_path)
    print(f"\nTraining done! Model: {final_path}.zip")

    env.close()


if __name__ == "__main__":
    train()

