import agentpy as ap
import numpy as np
import random
import matplotlib.pyplot as plt
import IPython.display

class MazeAgent(ap.Agent):
    def setup(self):
        self.actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.env = self.model.env
        self.reward = 0
        self.Q = self.p.Q
        self.epsilon = self.p.epsilon
        self.alpha = self.p.alpha
        self.gamma = self.p.gamma

    def execute(self):
        action = self.choose_action(self.get_position())
        state = self.get_position()
        self.env.move_by(self, self.actions[action])
        new_state = self.get_position()
        reward = self.env.get_reward(new_state)
        self.update_Q(state, action, reward, new_state)
        self.reward += reward

    def get_position(self):
        return self.env.positions[self]

    def train(self, train_steps):
        for _ in range(train_steps):
            self.execute()
            if self.get_position() == self.p.goal:
                self.env.move_to(self, self.p.init)
                self.reward = 0
                self.env.setup()

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.actions.keys()))
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def update_Q(self, state, action, reward, new_state):
        max_Q_new_state = max(self.Q[new_state].values())
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * max_Q_new_state - self.Q[state][action]
        )

class Maze(ap.Grid):
    def setup(self):
        self.rewards = np.copy(self.p.maze)
        self.rewards[self.rewards == -1] = self.p.wall_value
        self.rewards[self.p.goal] = self.p.goal_value

    def get_reward(self, state):
        reward = self.rewards[state]
        self.rewards[state] = 0 if reward > 0 else reward
        return reward

class MazeModel(ap.Model):
    def setup(self):
        self.env = Maze(self, shape=self.p.maze.shape)
        self.agent = MazeAgent(self)
        self.env.add_agents([self.agent], positions=[self.p.init])

        print("Training agent...")
        self.agent.train(self.p.train_steps)

        print("Starting simulation...")
        self.env.move_to(self.agent, self.p.init)
        self.agent.reward = 0

    def step(self):
        self.agent.execute()

    def update(self):
        if self.agent.get_position() == self.p.goal:
            print("Agent reached the goal!")
            self.stop()

    def end(self):
        print("Simulation completed.")
        print("Final Q-Table:", self.agent.Q)

def animation_plot(model, ax):
    grid = np.zeros(model.p.maze.shape)
    grid[model.p.maze < 0] = -1
    grid[model.p.goal] = 2
    color_dict = {0: "#ffffff", -1: "#000000", 1: "#0000ff", 2: "#00ff00"}
    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)
    agent = list(model.env.agents)[0]
    state = model.env.positions[agent]
    grid[state] = 1
    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)
    ax.set_title(f"Agent Q-Learning\nReward: {agent.reward}\nState: {state}")

maze = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0,  0, -1,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1, -1,  0, -1],
    [-1,  0, -1,  0,  0,  0,  0, -1,  0, -1],
    [-1,  0,  0,  0, -1, -1,  0, -1,  0, -1],
    [-1, -1, -1,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0,  0, -1, -1, -1, -1,  0, -1, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0,  0, -1, -1,  0, -1, -1,  0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
])

actions = ['up', 'down', 'left', 'right']
Q = {(x, y): {a: 0 for a in actions} for x in range(maze.shape[0]) for y in range(maze.shape[1])}

parameters = {
    "maze": maze,
    "init": (1, 1),
    "goal": (8, 8),
    "goal_value": 100,
    "wall_value": -100,
    "epsilon": 0.1,
    "alpha": 0.5,
    "gamma": 0.9,
    "train_steps": 1000,
    "Q": Q,
}

fig, ax = plt.subplots(figsize=(7, 7))
model = MazeModel(parameters)
animation = ap.animate(model, fig, ax, animation_plot)
IPython.display.HTML(animation.to_jshtml())
