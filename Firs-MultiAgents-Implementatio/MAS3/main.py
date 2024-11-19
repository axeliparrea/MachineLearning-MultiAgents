import agentpy as ap
import numpy as np
import random, json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns, IPython
from matplotlib import pyplot as plt, cm

class MazeAgent(ap.Agent):
    '''
    Initializing agent elements:
    - 4 possible actions
    - Q values as a zero matrix (unless a matrix definition is provided)
    - Policies. Values of epsilon, alpha, and gamma
    '''
    def setup(self):
        # Actions are linked to a movement in the grid.
        self.actions = {'up': (-1,0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.env = self.model.env
        m, n = self.env.shape
        self.reward = 0
        # Else assign values
        self.Q = self.p.Q
        # Learning policies
        self.epsilon = self.p.epsilon
        self.alpha = self.p.alpha
        self.gamma = self.p.gamma
        self.name = 'Speedy learner'

    '''
    Actual action execution. This process will be employed after agent has trained
    '''
    def execute(self):
        action = self.choose_action(self.get_position())
        state = self.get_position()
        self.env.move_by(self, self.actions[action])
        new_state = self.get_position()
        reward = self.env.get_reward(new_state)
        self.update_Q(state, action, reward, new_state)    # Update Q-values
        self.reward += reward

    '''
    Get position of agent in environment
    '''
    def get_position(self):
        return self.env.positions[self]


    '''
    Training. Agent will be able to perform a number of training steps.
    A complete cycle finishes until agent reaches the goal
    '''
    def train(self, train_steps):
        for i in range(train_steps):
            self.execute()
            if self.get_position() == self.p.goal:      # Execute training until agent reaches the goal
                self.env.move_to(self, self.p.init)     # Initialize environment and agent
                self.rewards = 0
                self.env.setup()


    '''
    Applying epsilon greedy policy
    '''
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.actions.keys()))
        else:
            return max(self.Q[state], key=self.Q[state].get)

    '''
    Updating Q-values according to definition
    '''
    def update_Q(self, state, action, reward, new_state):
        max_Q_new_state = max(self.Q[new_state].values())
        self.Q[state][action] = np.round(self.Q[state][action] + self.alpha * (
            reward + self.gamma * max_Q_new_state - self.Q[state][action]), 2)


'''
'''
class Maze(ap.Grid):
    def setup(self):
        # Initialize the maze environment
        self.rewards = np.copy(self.p.maze)
        self.rewards[self.rewards == -1] = self.p.wall_value
        self.rewards[self.p.goal] = self.p.goal_value

    '''
    Reward function. The returned value is used to update Q-values
    '''
    def get_reward(self, state):
        reward = self.rewards[state] - 1
        self.rewards[state] = 0 if self.rewards[state] > 0 else self.rewards[state]
        return reward

'''
'''
class MazeModel(ap.Model):
    def setup(self):
        self.env = Maze(self, shape=maze.shape)
        self.agent = MazeAgent(self)
        self.env.add_agents([self.agent], positions=[self.p.init])

        # Executing training steps for the agent. Modify parameters as needed after execution
        print('training....')
        self.agent.train(self.p.train_steps)

        print('executing....')
        # Setting the agent in its initial position, and zero reward
        self.env.move_to(self.agent, self.p.init)
        self.agent.reward = 0

    def step(self):
        self.agent.execute()

    def update(self):
        if self.agent.get_position() == self.model.p.goal:
            print('ending')
            self.stop()

    # Report found route and Q-values
    def end(self):
        self.report('Q-Table', self.agent.Q)




def animation_plot(model, ax):
    n, m = model.p.maze.shape
    grid = np.zeros((n, m))
    grid[model.p.maze < 0] = -1
    grid[model.p.goal] = 2
    # Colors: black = walls, white = floor, green = goal, blue = agent
    color_dict = {0:'#ffffff', -1:'#000000', 1:'#0000ff', 2:'#00ff00'}
    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)
    agent = list(model.env.agents)[0]
    state = model.env.positions[agent]
    grid[state] = 1
    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)
    ax.set_title("Agent Q-Learning\nReward: {}\nQ[{}]: {}".format(agent.reward, state, agent.Q[state]))


actions = ['up', 'down', 'left', 'right']

maze = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1],
                 [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1],
                 [-1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, -1],
                 [-1, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, -1, 0, -1],
                 [-1, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0, -1, 0, 0, -1],
                 [-1, -1, -1, 0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, -1],
                 [-1, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, -1],
                 [-1, -1, 0, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, -1],
                 [-1, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1],
                 [-1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                 [-1, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1],
                 [-1, -1, 0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1],
                 [-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, -1],
                 [-1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1],
                 [-1, 0, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1],
                 [-1, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1],
                 [-1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1],
                 [-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1],
                 [-1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
                 [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1]])
np.save('test_maze.npy', maze)

m, n = maze.shape
# Initializing Q-values
actions = ['down', 'left', 'right', 'up']
Q = {}
for x in range(m):
    for y in range(n):
        Q[(x, y)] = {action: 0 for action in actions}

parameters = {
    'maze': maze,
    'init': (n - 2, n - 3),
    'goal': (8, 10),
    'goal_value': 100,
    'wall_value': -100,
    'epsilon': 1,
    'alpha': 1,
    'gamma': 1,
    'train_steps': 10000,
    'steps': 100,
    'Q': Q
}

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
mazeModel = MazeModel(parameters)
#mazeModel.setup()
animation = ap.animate(mazeModel, fig, ax, animation_plot)
IPython.display.HTML(animation.to_jshtml())