
# Model design
import agentpy as ap
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import IPython


class MyAgent(ap.Agent):
    def setup(self):
        # 1. Initialize agent's attribute: health with random value between 60-100
        self.health = self.model.random.randint(60, 100)
        self.health_record = [] #historial
        self.position_history = []

    def setup_pos(self, env):
        self.pos = env.positions[self]
        self.position_history.append(self.pos)

    def move(self):
        new_pos = (self.pos[0] + self.model.random.randint(-1, 2),
                   self.pos[1] + self.model.random.randint(-1, 2))

        # 2. Validate and update position ensuring it stays within grid bounds
        new_x = max(0, min(new_pos[0], self.model.p.size - 1))
        new_y = max(0, min(new_pos[1], self.model.p.size - 1))
        self.pos = (new_x, new_y)
        self.position_history.append(self.pos)

    def find_nearest_neighbor(self):
        if not self.model.agents:
            return None

        # Calculate distances to all other agents
        distances = []
        for agent in self.model.agents:
            if agent != self:
                dist = np.sqrt((self.pos[0] - agent.pos[0])**2 +
                             (self.pos[1] - agent.pos[1])**2)
                distances.append((dist, agent))

        if distances:
            return min(distances, key=lambda x: x[0])[1]
        return None

    def update(self):
        # 3. Update health values - decrease by 5 each step
        self.health -= 5
        self.health_record.append(self.health)
        # Updating position values using the move method
        self.move()

        # 4. Record health data
        

class MyModel(ap.Model):
    def setup(self):
        # 5. Initialize a grid with the given values
        self.p.size = 20  # Grid size parameter
        self.room = ap.Grid(self, [self.p.size, self.p.size], track_empty=True)

        # 6. Add 10 agents of type MyAgent to the environment
        self.agents = ap.AgentList(self, 10, MyAgent)

        # Place agents randomly in the grid
        positions = [(self.random.randint(0, self.p.size-1),
                     self.random.randint(0, self.p.size-1))
                    for _ in range(10)]
        self.room.add_agents(self.agents, positions)

        # Setting position in the pos field of the agents
        self.room.agents.setup_pos(self.room)

        # Dictionary to store all health records before removal
        self.all_health_records = {f'Agent {agent.id}': agent.health_record for agent in self.agents}

    def step(self):
        # 7. Update all agents values on each step
        for agent in self.agents:
            agent.update()

        # Find agents with health <= 0 and handle their removal
        dying_agents = [agent for agent in self.agents if agent.health <= 0]
        for agent in dying_agents:
            # Extra. Find nearest neighbor and send goodbye message
            nearest = agent.find_nearest_neighbor()
            if nearest:
                print(f"Agent {agent.id} says goodbye to nearest agent {nearest.id}")

            # Update all health records before removing the agent
            self.all_health_records[f'Agent {agent.id}'] = agent.health_record
            
            # Remove dying agent
            self.room.remove_agents(agent)
            self.agents.remove(agent)

        # 8. Check if room has agents left. If no agents, stop the simulation
        if len(self.agents) == 0:
            self.stop()

        if all(agent.health <= 0 for agent in self.agents):
            self.stop()

    def end(self):
      # 9. Report Agent's positions
      self.report('Agents position', [agent.pos for agent in self.agents])

      # Collect and store health records in a model field
      self.all_health_records = {f'Agent {agent.id}': agent.health_record for agent in self.agents}


def run_and_visualize():
    parameters = {'size': 20}
    model = MyModel(parameters)
    model.run()

    # Plot health records after the simulation
    plt.figure(figsize=(10, 6))
    for agent_id, health_data in model.all_health_records.items():
        plt.plot(health_data, label=f'Agent {agent_id}')
    
    plt.title('Agent Health Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Health')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_and_visualize()

