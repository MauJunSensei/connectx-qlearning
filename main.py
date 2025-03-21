import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from random import choice
import psutil
import time
import src.config as cfg
from src.qtable import QTable
from src.connectx import ConnectX
from kaggle_environments import evaluate
import tkinter as tk
from tkinter import ttk
import shutil
import concurrent.futures

def plot_training_data(
    runtime_data,
    memory_data,
    total_rewards,
    epsilon_data,
    q_diff_data,
    q_table_size_data,
    eval_random_data,
    eval_negamax_data,
):
    def remove_outliers(data, m=2):
        data = np.array(data)
        mean = np.mean(data)
        std_dev = np.std(data)
        filtered_data = data[abs(data - mean) < m * std_dev]
        return filtered_data

    def plot_data(ax, data, xlabel, ylabel, title, window_size=cfg.WINDOW_SIZE):
        print(f"Plotting {title}...")
        if window_size:
            data = [
                np.mean(data[max(0, i - window_size) : i + 1]) for i in range(len(data))
            ]
        filtered_data = remove_outliers(data)
        ax.plot(range(len(filtered_data)), filtered_data, color="orange")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    plot_data(
        axs[0, 0],
        runtime_data,
        "Episodes",
        "Elapsed Time per Episode (seconds)",
        "Runtime per Episode",
    )
    plot_data(
        axs[0, 1],
        memory_data,
        "Episodes",
        "Memory Usage (MB)",
        "Memory Usage over Episodes",
    )
    plot_data(
        axs[0, 2],
        np.cumsum(total_rewards),
        "Episodes",
        "Cumulative Reward",
        "Cumulative Reward over Episodes",
    )
    plot_data(
        axs[0, 3],
        total_rewards,
        "Episodes",
        "Moving Average Reward",
        "Moving Average Reward per Episode",
    )
    plot_data(
        axs[1, 0],
        [
            np.var(total_rewards[max(0, i - 100) : i + 1])
            for i in range(len(total_rewards))
        ],
        "Episodes",
        "Reward Variance",
        "Reward Variance over Episodes",
    )
    plot_data(
        axs[1, 1],
        epsilon_data,
        "Episodes",
        "Exploration Rate (Epsilon)",
        "Exploration Rate over Episodes",
    )
    plot_data(
        axs[1, 2],
        q_diff_data,
        "Episodes",
        "Q-value Difference",
        "Q-value Difference over Episodes",
    )
    plot_data(
        axs[1, 3],
        q_table_size_data,
        "Episodes",
        "Q-table Size",
        "Q-table Size over Episodes",
    )
    plot_data(
        axs[2, 0],
        eval_random_data,
        "Switch Intervals",
        "Evaluation Score",
        "Evaluation vs Random Agent",
    )
    plot_data(
        axs[2, 1],
        eval_negamax_data,
        "Switch Intervals",
        "Evaluation Score",
        "Evaluation vs Negamax Agent",
    )
    print("Q-table size:", q_table.get_size())
    plt.tight_layout()
    plt.show()


def choose_action(state, epsilon, env, q_table):
    if np.random.random() < epsilon:
        return choice([c for c in range(env.action_space.n) if state.board[c] == 0])
    else:
        q_values = q_table.get(state)
        valid_actions = [
            q_values[c] if state.board[c] == 0 else -np.inf
            for c in range(env.action_space.n)
        ]
        return int(np.argmax(valid_actions))


def run_episode(q_table, epsilon, q_diff_list, episode_num):
    env = ConnectX()
    if (episode_num // cfg.SWITCH_INTERVAL) % 2 == 0:
        env.trainer = env.env.train([None, "random"])
    else:
        env.trainer = env.env.train([None, "negamax"])
    state = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = choose_action(state, epsilon, env, q_table)
        next_state, reward, done, info = env.step(action)

        # Apply new rules
        if done:
            reward = 1.0 if reward == 1 else -1.0 if reward == 0 else 0.0
        else:
            reward = -0.01  # Discourage long games

        total_reward += reward

        q_values = q_table.get(state)
        next_q_values = q_table.get(next_state)

        # Update Q-value
        target = reward if done else reward + cfg.GAMMA * np.max(next_q_values)
        new_value = (1 - cfg.ALPHA) * q_values[action] + cfg.ALPHA * target
        q_diff = abs(new_value - q_values[action])
        q_diff_list.append(q_diff)
        q_table.update(state, action, new_value)

        state = next_state

    return total_reward, q_table.get_size()


def worker(
    q_table,
    progress_queue,
    reward_queue,
    q_diff_queue,
    q_table_size_queue,
    start_episode,
    epsilon_shared,
):
    q_diff_list = []

    for episode in range(
        start_episode, start_episode + cfg.NUM_EPISODES // multiprocessing.cpu_count()
    ):
        with epsilon_shared.get_lock():  # Read the shared epsilon safely
            epsilon = epsilon_shared.value  # Read but do NOT update

        total_reward, q_table_size = run_episode(q_table, epsilon, q_diff_list, episode)
        reward_queue.put(total_reward)
        progress_queue.put(1)
        q_diff_queue.put(np.mean(q_diff_list) if q_diff_list else 0)
        q_table_size_queue.put(q_table_size)


def initialize_queues():
    return (multiprocessing.Queue() for _ in range(4))


def initialize_gui():
    root = tk.Tk()
    root.title("Training Progress")
    progress_bar = ttk.Progressbar(
        root,
        orient="horizontal",
        length=400,
        mode="determinate",
        maximum=cfg.NUM_EPISODES,
    )
    progress_bar.pack(pady=20)
    desc_label = tk.Label(root, text="Training Progress")
    desc_label.pack(pady=10)
    percentage_label = tk.Label(root, text="0%")
    percentage_label.pack(pady=10)
    trainer_label = tk.Label(root, text="Current Trainer: Random")
    trainer_label.pack(pady=10)
    return root, progress_bar, percentage_label, trainer_label


def update_gui(progress_bar, percentage_label, trainer_label, i, epsilon_shared):
    progress_bar["value"] = i + 1
    percentage = (i + 1) / cfg.NUM_EPISODES * 100
    percentage_label.config(text=f"{percentage:.2f}%")
    current_trainer = "Random" if (i // cfg.SWITCH_INTERVAL) % 2 == 0 else "Negamax"
    trainer_label.config(text=f"Current Trainer: {current_trainer}")
    with epsilon_shared.get_lock():
        new_epsilon = cfg.MIN_EPSILON + (cfg.EPSILON - cfg.MIN_EPSILON) / np.sqrt(i + 1)
        epsilon_shared.value = new_epsilon
        epsilon_data.append(new_epsilon)

def evaluate_agent(q_table, i):
    """
    Evaluates the agent against random and negamax agents.
    """
    def my_agent(obs, conf):
        state_key = hex(int("".join(map(str, obs.board + [obs.mark])), 3))[2:]
        return dict_q_table.get(
            state_key,
            choice([c for c in range(conf.columns) if obs.board[c] == 0])
        )

    eval_random_data, eval_negamax_data = [], []

    if (i + 1) % cfg.SWITCH_INTERVAL == 0:
        dict_q_table = q_table.get_table()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_random = executor.submit(
                evaluate, "connectx", [my_agent, "random"], num_episodes=cfg.EVAL_EP_COUNT
            )
            future_negamax = executor.submit(
                evaluate, "connectx", [my_agent, "negamax"], num_episodes=cfg.EVAL_EP_COUNT
            )

            random_results = future_random.result()
            negamax_results = future_negamax.result()

        eval_random_data.append(np.mean([r[0] for r in random_results]))
        eval_negamax_data.append(np.mean([r[0] for r in negamax_results]))
        print(
            f"Episode {i + 1} - Random: {eval_random_data[-1]}, "
            f"Negamax: {eval_negamax_data[-1]}"
        )

    return eval_random_data, eval_negamax_data


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    env = ConnectX()
    q_table = QTable(env.action_space)
    progress_queue, reward_queue, q_diff_queue, q_table_size_queue = initialize_queues()
    threads = []
    start_time = time.time()
    process = psutil.Process()
    runtime_data, memory_data, epsilon_data, q_diff_data, q_table_size_data = [], [], [], [], []
    total_rewards = []
    last_time = start_time
    epsilon_shared = multiprocessing.Value("d", cfg.EPSILON)
    eval_random_data, eval_negamax_data = [], []

    for i in range(multiprocessing.cpu_count()):
        thread = multiprocessing.Process(
            target=worker,
            args=(
                q_table,
                progress_queue,
                reward_queue,
                q_diff_queue,
                q_table_size_queue,
                i * (cfg.NUM_EPISODES // multiprocessing.cpu_count()),
                epsilon_shared,
            ),
        )
        thread.start()
        threads.append(thread)

    root, progress_bar, percentage_label, trainer_label = initialize_gui()

    for i in range(cfg.NUM_EPISODES):
        progress_queue.get()
        reward = reward_queue.get()
        q_diff = q_diff_queue.get()
        q_table_size = q_table_size_queue.get()

        total_rewards.append(reward)
        q_diff_data.append(q_diff)
        q_table_size_data.append(q_table_size)

        update_gui(progress_bar, percentage_label, trainer_label, i, epsilon_shared)

        current_time = time.time()
        elapsed_time = current_time - last_time
        last_time = current_time

        memory_info = process.memory_info()
        runtime_data.append(elapsed_time)
        memory_data.append(memory_info.rss / (1024 * 1024))  # MB

        root.update_idletasks()
        root.update()

        eval_random, eval_negamax = evaluate_agent(q_table, i)
        eval_random_data.extend(eval_random)
        eval_negamax_data.extend(eval_negamax)

    root.destroy()
    plot_training_data(
        runtime_data,
        memory_data,
        total_rewards,
        epsilon_data,
        q_diff_data,
        q_table_size_data,
        eval_random_data,
        eval_negamax_data,
    )

    # Path to the qtable.lmdb folder
    qtable_path = "qtable.lmdb"

    # Path to the output zip file
    output_zip = "qtable_backup.zip"

    # Create a zip file of the qtable.lmdb folder
    shutil.make_archive(output_zip.replace('.zip', ''), 'zip', qtable_path)