import json, numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Reading information from files
f = open('params3.json')
parameters = json.load(f)
state, money, step_cost, position = [parameters[key] for key in ['state', 'money', 'step_cost', 'position']]
h, w = np.array(state).shape
plt.rcParams['figure.figsize'] = [w//2, h//2]

print('money: {}'.format(money))
print('step cost: {}'.format(step_cost))
print('position: {}'.format(position))
ax = sns.heatmap(state, linewidth=0.5, linecolor='black', annot=True, cmap='RdBu', cbar=False, vmin=-7, vmax=7)
plt.xticks([])
plt.yticks([])
plt.show()