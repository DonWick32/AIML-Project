from gym_tictactoe.env import TicTacToeEnv

env = TicTacToeEnv()

print("Available Actions:", env.available_actions())

actions = [0, 4, 1, 5, 2]
for action in actions:
    observation, reward, done, trunc, _ = env.step(action)
    env.render()
    if done:
        print("Game Over!")
        break

print("Available Actions:", env.available_actions())