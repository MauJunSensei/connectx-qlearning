import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from tqdm import tqdm
import psutil
import time
import src.config as cfg
from src.qtable import QTable
from src.connectx import ConnectX
from kaggle_environments import evaluate

def choose_action(state, epsilon, env, q_table):
    if np.random.random() < epsilon:
        return choice([c for c in range(env.action_space.n) if state.board[c] == 0])
    else:
        q_values = q_table.get(state)
        valid_actions = [q_values[c] if state.board[c] == 0 else -np.inf for c in range(env.action_space.n)]
        return int(np.argmax(valid_actions))

def run_episode(q_table, epsilon, q_diff_list):
    env = ConnectX()
    state = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = choose_action(state, epsilon, env, q_table)
        next_state, reward, done, info = env.step(action)

        # Apply new rules
        if done:
            if reward == 1:  # Won
                reward = 1.0
            elif reward == 0:  # Lost
                reward = -1.0
            else:  # Draw
                reward = 0.0
        else:
            reward = -0.01  # Try to prevent the agent from taking a long time to win

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
        
    return total_reward, epsilon

def worker(q_table, progress_queue, reward_queue, epsilon_queue, q_diff_queue):
    epsilon = cfg.EPSILON
    q_diff_list = []
    for _ in range(cfg.NUM_EPISODES // multiprocessing.cpu_count()):
        total_reward, epsilon = run_episode(q_table, epsilon, q_diff_list)
        reward_queue.put(total_reward)
        progress_queue.put(1)
        epsilon_queue.put(epsilon)
        q_diff_queue.put(np.mean(q_diff_list) if q_diff_list else 0)
        epsilon = max(cfg.MIN_EPSILON, epsilon * cfg.EPSILON_DECAY_RATE)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    env = ConnectX()
    q_table = QTable(env.action_space)
    progress_queue = multiprocessing.Queue()
    reward_queue = multiprocessing.Queue()
    epsilon_queue = multiprocessing.Queue()
    q_diff_queue = multiprocessing.Queue()

    threads = []
    
    start_time = time.time()
    process = psutil.Process()

    runtime_data = []
    memory_data = []
    epsilon_data = []
    q_diff_data = []
    cpu_usage_data = []

    last_time = start_time

    for _ in range(multiprocessing.cpu_count()):
        thread = multiprocessing.Process(target=worker, args=(q_table, progress_queue, reward_queue, epsilon_queue, q_diff_queue))
        thread.start()
        threads.append(thread)

    total_rewards = []
    
    with tqdm(total=cfg.NUM_EPISODES, desc="Training Progress") as pbar:
        for i in range(cfg.NUM_EPISODES):
            progress_queue.get()
            reward = reward_queue.get()
            epsilon = epsilon_queue.get()
            q_diff = q_diff_queue.get()
            total_rewards.append(reward)
            epsilon_data.append(epsilon)
            q_diff_data.append(q_diff)
            pbar.update(1)

            # Collect runtime, memory usage, and CPU usage data
            current_time = time.time()
            elapsed_time = current_time - last_time
            last_time = current_time
            memory_info = process.memory_info()
            runtime_data.append(elapsed_time)
            memory_data.append(memory_info.rss / (1024 * 1024))  # Convert to MB
            cpu_usage_data.append(psutil.cpu_percent(interval=None))  # Collect CPU usage

    for thread in threads:
        thread.join()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print("DB Size:", q_table.get_size())

    # Plot data
    def remove_outliers(data, m=2):
        mean, std = np.mean(data), np.std(data)
        return [x for x in data if mean - m * std < x < mean + m * std]

    def plot_data(ax, data, xlabel, ylabel, title, window_size=None):
        if window_size:
            data = [np.mean(data[max(0, i-window_size):i+1]) for i in range(len(data))]
        filtered_data = remove_outliers(data)
        ax.plot(filtered_data)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
    fig, axs = plt.subplots(4, 2, figsize=(15, 25))
    
    plot_data(axs[0, 0], runtime_data, 'Episodes', 'Elapsed Time per Episode (seconds)', 'Runtime per Episode', window_size=10)
    plot_data(axs[0, 1], memory_data, 'Episodes', 'Memory Usage (MB)', 'Memory Usage over Episodes', window_size=10)
    plot_data(axs[1, 0], np.cumsum(total_rewards), 'Episodes', 'Cumulative Reward', 'Cumulative Reward over Episodes', window_size=10)
    plot_data(axs[1, 1], total_rewards, 'Episodes', 'Moving Average Reward', 'Moving Average Reward per Episode', window_size=10)
    plot_data(axs[2, 0], [np.var(total_rewards[max(0, i-100):i+1]) for i in range(len(total_rewards))], 'Episodes', 'Reward Variance', 'Reward Variance over Episodes', window_size=10)
    plot_data(axs[2, 1], epsilon_data, 'Episodes', 'Exploration Rate (Epsilon)', 'Exploration Rate over Episodes', window_size=10)
    plot_data(axs[3, 0], q_diff_data, 'Episodes', 'Q-value Difference', 'Q-value Difference over Episodes', window_size=10)
    plot_data(axs[3, 1], cpu_usage_data, 'Episodes', 'CPU Usage (%)', 'CPU Usage over Episodes', window_size=10)

    plt.tight_layout()
    plt.show()

    tmp_dict_q_table = q_table.table.copy()
    dict_q_table = dict()

    for k in tmp_dict_q_table:
        if np.count_nonzero(tmp_dict_q_table[k]) > 0:
            dict_q_table[k] = int(np.argmax(tmp_dict_q_table[k]))

    my_agent = (
        """def my_agent(observation, configuration):
        from random import choice

        q_table = """
        + str(dict_q_table).replace(" ", "")
        + """

        board = observation.board[:]
        board.append(observation.mark)
        state_key = list(map(str, board))
        state_key = hex(int(''.join(state_key), 3))[2:]

        if state_key not in q_table.keys():
            return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

        action = q_table[state_key]

        if observation.board[action] != 0:
            return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

        return action
        """
    )

    with open("submission.py", "w") as f:
        f.write(my_agent)


    def mean_reward(rewards):
        return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)


    from submission import my_agent

    
    print('Q-Table size: ', len(q_table.table))
    # Run multiple episodes to estimate agent's performance.
    print(
        "My Agent vs Random Agent:",
        mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=100)),
    )
    print(
        "My Agent vs Negamax Agent:",
        mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=100)),
    )