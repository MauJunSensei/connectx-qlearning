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
        
    return total_reward, q_table.get_size()

def worker(q_table, progress_queue, reward_queue, q_diff_queue, q_table_size_queue, start_episode, epsilon_shared):
    q_diff_list = []
    
    for episode in range(start_episode, start_episode + cfg.NUM_EPISODES // multiprocessing.cpu_count()):
        with epsilon_shared.get_lock():  # Read the shared epsilon safely
            epsilon = epsilon_shared.value  # Read but do NOT update
        
        total_reward, q_table_size = run_episode(q_table, epsilon, q_diff_list)
        reward_queue.put(total_reward)
        progress_queue.put(1)
        q_diff_queue.put(np.mean(q_diff_list) if q_diff_list else 0)
        q_table_size_queue.put(q_table_size)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    env = ConnectX()
    q_table = QTable(env.action_space)
    progress_queue = multiprocessing.Queue()
    reward_queue = multiprocessing.Queue()
    q_diff_queue = multiprocessing.Queue()
    q_table_size_queue = multiprocessing.Queue()

    threads = []
    
    start_time = time.time()
    process = psutil.Process()

    runtime_data = []
    memory_data = []
    epsilon_data = []
    q_diff_data = []
    q_table_size_data = []

    last_time = start_time

    epsilon_shared = multiprocessing.Value('d', cfg.EPSILON)

    for i in range(multiprocessing.cpu_count()):
        thread = multiprocessing.Process(
            target=worker, 
            args=(q_table, progress_queue, reward_queue, q_diff_queue, q_table_size_queue, 
                  i * (cfg.NUM_EPISODES // multiprocessing.cpu_count()), epsilon_shared)
        )
        thread.start()
        threads.append(thread)

    total_rewards = []
    
    with tqdm(total=cfg.NUM_EPISODES, desc="Training Progress") as pbar:
        for i in range(cfg.NUM_EPISODES):
            progress_queue.get()
            reward = reward_queue.get()
            q_diff = q_diff_queue.get()
            q_table_size = q_table_size_queue.get()
            total_rewards.append(reward)
            q_diff_data.append(q_diff)
            q_table_size_data.append(q_table_size)
            pbar.update(1)

            # Update epsilon safely in the main process
            with epsilon_shared.get_lock():
                epsilon_shared.value = cfg.MIN_EPSILON + (cfg.EPSILON - cfg.MIN_EPSILON) / np.sqrt(i + 1)
                epsilon_data.append(epsilon_shared.value)

            # Collect runtime, memory usage
            current_time = time.time()
            elapsed_time = current_time - last_time
            last_time = current_time
            memory_info = process.memory_info()
            runtime_data.append(elapsed_time)
            memory_data.append(memory_info.rss / (1024 * 1024))  # Convert to MB

    for thread in threads:
        thread.join()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    # Plot data
    def remove_outliers(data, m=2):
        mean, std = np.mean(data), np.std(data)
        return [x for x in data if mean - m * std < x < mean + m * std]

    def plot_data(ax, data, xlabel, ylabel, title, window_size=cfg.WINDOW_SIZE):
        print(f"Plotting {title}...")
        if window_size:
            data = [np.mean(data[max(0, i-window_size):i+1]) for i in range(len(data))]
        filtered_data = remove_outliers(data)
        ax.plot(range(len(filtered_data)), filtered_data, color='orange')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    plot_data(axs[0, 0], runtime_data, 'Episodes', 'Elapsed Time per Episode (seconds)', 'Runtime per Episode')
    plot_data(axs[0, 1], memory_data, 'Episodes', 'Memory Usage (MB)', 'Memory Usage over Episodes')
    plot_data(axs[0, 2], np.cumsum(total_rewards), 'Episodes', 'Cumulative Reward', 'Cumulative Reward over Episodes')
    plot_data(axs[0, 3], total_rewards, 'Episodes', 'Moving Average Reward', 'Moving Average Reward per Episode')
    plot_data(axs[1, 0], [np.var(total_rewards[max(0, i-100):i+1]) for i in range(len(total_rewards))], 'Episodes', 'Reward Variance', 'Reward Variance over Episodes')
    plot_data(axs[1, 1], epsilon_data, 'Episodes', 'Exploration Rate (Epsilon)', 'Exploration Rate over Episodes')
    plot_data(axs[1, 2], q_diff_data, 'Episodes', 'Q-value Difference', 'Q-value Difference over Episodes')
    plot_data(axs[1, 3], q_table_size_data, 'Episodes', 'Q-table Size', 'Q-table Size over Episodes')

    plt.tight_layout()
    plt.show()

    tmp_dict_q_table = q_table.get_table()
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
    
    print('Q-Table size: ', q_table.get_size())
    # Run multiple episodes
    print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=500)))
    print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=500)))