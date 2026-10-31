import gymnasium as gym
from tqdm import tqdm

from agent import GridWorldAgent
from utils import encode_obs, save_model

learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

from env import GridWorldEnv

env = GridWorldEnv(render_mode=None)

agent = GridWorldAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

obs, info = env.reset()

episode_over = False
total_reward = 0

try:
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(encode_obs(obs))

            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(
                encode_obs(obs), action, reward, terminated, encode_obs(next_obs)
            )

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

except KeyboardInterrupt:
    print("退出中，正在保存模型...")

finally:
    try:
        model_dir = save_model(
            q_values=agent.q_values,
            project_name="grid_world",
        )
        print(f"模型已保存在{model_dir}")

    finally:
        env.close()
