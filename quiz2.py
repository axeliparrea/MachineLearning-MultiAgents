# Model design
import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt

class MyAgent(ap.Agent):
    def setup(self):
        # 1. Initialize agent's attribute: health with random value between 60-100
        self.health = self.model.random.randint(60, 101)
        self.health_record = []  # Health history
        self.position_history = []  # Position history

    def setup_pos(self, env):
        # 2. Initialize agent's position in the environment
        self.pos = env.positions[self]
        self.position_history.append(self.pos)

    def move(self):
        # 3. Simulate agent's movement by updating its position with random steps
        new_pos = (self.pos[0] + self.model.random.randint(-1, 2),
                   self.pos[1] + self.model.random.randint(-1, 2))

        # Validate and update position ensuring it stays within grid bounds
        new_x = max(0, min(new_pos[0], self.model.p.size - 1))
        new_y = max(0, min(new_pos[1], self.model.p.size - 1))
        self.pos = (new_x, new_y)
        self.position_history.append(self.pos)

    def find_nearest_neighbor(self):
        # 4. Find the nearest neighbor based on Euclidean distance
        if not self.model.agents:
            return None

        # Calculate distances to all other agents
        distances = []
        for agent in self.model.agents:
            if agent != self:
                dist = np.sqrt((self.pos[0] - agent.pos[0])**2 +
                               (self.pos[1] - agent.pos[1])**2)
                distances.append((dist, agent))

        # Return the nearest neighbor if any
        if distances:
            return min(distances, key=lambda x: x[0])[1]
        return None

    def update(self):
        # 5. Update health values - decrease by 5 each step
        self.health -= 5

        # Move agent and record new position
        self.move()

        # Record health data
        self.health_record.append(self.health)
        self.model.all_health_records[f'Agent {self.id}'].append(self.health)

class MyModel(ap.Model):
    def setup(self):
        # 6. Initialize a grid with the given size parameter
        self.p.size = 20  # Grid size parameter
        self.room = ap.Grid(self, [self.p.size, self.p.size], track_empty=True)

        # Add 10 agents of type MyAgent to the environment
        self.agents = ap.AgentList(self, 10, MyAgent)
        self.room.add_agents(self.agents, random=True)

        # Set initial positions for each agent
        self.room.agents.setup_pos(self.room)

        # Dictionary to store all health records before removal
        self.all_health_records = {f'Agent {agent.id}': agent.health_record for agent in self.agents}

    def step(self):
        # 7. Update all agents' health and position on each step
        for agent in list(self.agents):
            agent.update()

        # Identify agents with health <= 0 and prepare for removal
        dying_agents = [agent for agent in self.agents if agent.health <= 0]
        for agent in dying_agents:
            # Extra. Find nearest neighbor and send goodbye message
            nearest = agent.find_nearest_neighbor()
            if nearest:
                print(f"Agent {agent.id} says goodbye to nearest agent {nearest.id}")

            # Remove dying agent from grid and list of agents
            self.room.remove_agents(agent)
            self.agents.remove(agent)

        # 8. Check if room has agents left. If no agents, stop the simulation
        if len(self.agents) == 0 or all(agent.health <= 0 for agent in self.agents):
            self.stop()

    def end(self):
        # 9. Report the final positions of each agent
        self.report('Agents position', [agent.pos for agent in self.agents])

def run_and_visualize():
    # Set model parameters
    parameters = {'size': 20}
    
    # Initialize and run the model
    model = MyModel(parameters)
    model.run()

    # Plot health records after the simulation
    plt.figure(figsize=(10, 6))
    for agent_id, health_data in model.all_health_records.items():
        plt.plot(health_data, label=agent_id)
    
    # Set plot titles and labels
    plt.title('Agent Health Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Health')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Run simulation and visualize results
if __name__ == "__main__":
    run_and_visualize()
