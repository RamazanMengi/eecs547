from mesa import DataCollector
import matplotlib.pyplot as plt
import random
from mesa import Model, Agent
from mesa.time import RandomActivation
from copy import deepcopy
from collections import defaultdict, deque
import numpy as np

tasks = [{'task_name': 'Steel Hull - Block 1', 'duration': 4, 'prerequisites': [], 'subcontractor': {'steel-subcontractor': 1}, 'resources': {'steel': {'amount': 1, 'constant': 50}}, 'value': 25}, {'task_name': 'Steel Hull - Block 2', 'duration': 4, 'prerequisites': ['Steel Hull - Block 1'], 'subcontractor': {'steel-subcontractor': 1}, 'resources': {'steel': {'amount': 1, 'constant': 50}}, 'value': 10}, {'task_name': 'Steel Hull - Block 3', 'duration': 4, 'prerequisites': ['Steel Hull - Block 2'], 'subcontractor': {'steel-subcontractor': 1}, 'resources': {'steel': {'amount': 1, 'constant': 75}}, 'value': 10}, {'task_name': 'Steel Hull - Block 4', 'duration': 4, 'prerequisites': ['Steel Hull - Block 3'], 'subcontractor': {'steel-subcontractor': 1}, 'resources': {'steel': {'amount': 1, 'constant': 75}}, 'value': 15}, {'task_name': 'Steel Hull - Block 5', 'duration': 4, 'prerequisites': ['Steel Hull - Block 4'], 'subcontractor': {'steel-subcontractor': 1}, 'resources': {'steel': {'amount': 1, 'constant': 50}}, 'value': 15}, {'task_name': 'Steel Hull - Tank Tests', 'duration': 3, 'prerequisites': ['Steel Hull - Block 5'], 'subcontractor': {'steel-subcontractor': 1}, 'resources': {'steel': {'amount': 1, 'constant': 25}}, 'value': 75}, {'task_name': 'Steel Hull - Block 6', 'duration': 5, 'prerequisites': ['Steel Hull - Block 5'], 'subcontractor': {'steel-subcontractor': 1}, 'resources': {'steel': {'amount': 1, 'constant': 25}}, 'value': 20}, {'task_name': 'Steel Hull - Outfitting', 'duration': 4, 'prerequisites': ['Steel Hull - Tank Tests'], 'subcontractor': {'outfitting': 1}, 'resources': {'outfitting-equ': {'amount': 1, 'constant': 0.7}}, 'value': 60}, {'task_name': 'Aluminium Superstructure - Block 1', 'duration': 4, 'prerequisites': ['Steel Hull - Block 2'], 'subcontractor': {'aluminium-subcontractor': 1}, 'resources': {'aluminium': {'amount': 1, 'constant': 30}}, 'value': 75}, {'task_name': 'Aluminium Superstructure - Block 2', 'duration': 4, 'prerequisites': ['Aluminium Superstructure - Block 1'], 'subcontractor': {'aluminium-subcontractor': 1}, 'resources': {'aluminium': {'amount': 1, 'constant': 20}}, 'value': 30}, {'task_name': 'Aluminium Superstructure - Block 3', 'duration': 5, 'prerequisites': ['Aluminium Superstructure - Block 2'], 'subcontractor': {'aluminium-subcontractor': 1}, 'resources': {'aluminium': {'amount': 1, 'constant': 20}}, 'value': 30}, {'task_name': 'Aluminium Superstructure - Block 4', 'duration': 3, 'prerequisites': ['Aluminium Superstructure - Block 3'], 'subcontractor': {'aluminium-subcontractor': 1}, 'resources': {'aluminium': {'amount': 1, 'constant': 20}}, 'value': 75}, {'task_name': 'Aluminium Superstructure - Outfitting', 'duration': 3, 'prerequisites': ['Aluminium Superstructure - Block 4'], 'subcontractor': {'outfitting': 1}, 'resources': {'outfitting-equ': {'amount': 1, 'constant': 0.3}}, 'value': 80}, {'task_name': 'Sandblasting', 'duration': 2, 'prerequisites': ['Aluminium Superstructure - Outfitting'], 'subcontractor': {'sandblasting-subcontractor': 1}, 'resources': {'sandblasting-mat': {'amount': 1, 'constant': 1}}, 'value': 80}, {'task_name': 'Furniture - Mascoat, insulation, electrical and piping works of the below deck guest cabins', 'duration': 4, 'prerequisites': ['Sandblasting'], 'subcontractor': {'sandblasting-subcontractor': 0.2, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'sandblasting-mat': {'amount': 0.2, 'constant': 0.1}, 'electrical-mat': {'amount': 0.2, 'constant': 1}, 'piping-mat': {'amount': 0.2, 'constant': 1}, 'insulation-mat': {'amount': 0.2, 'constant': 1}}, 'value': 80}, {'task_name': 'Furniture - Below Deck Crew Cabins', 'duration': 25, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping works of the below deck guest cabins'], 'subcontractor': {'furniture-subcontractor': 0.3}, 'resources': {'furniture-mat': {'amount': 0.3, 'constant': 1}}, 'value': 35}, {'task_name': 'Furniture - Mascoat, insulation, electrical and piping works of the below deck crew cabins, crew mess and galley', 'duration': 4, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping works of the below deck guest cabins'], 'subcontractor': {'sandblasting-subcontractor': 0.2, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'sandblasting-mat': {'amount': 0.2, 'constant': 0.1}, 'electrical-mat': {'amount': 0.2, 'constant': 1.2}, 'piping-mat': {'amount': 0.2, 'constant': 1.2}, 'insulation-mat': {'amount': 0.2, 'constant': 1.2}}, 'value': 50}, {'task_name': 'Furniture - Furniture works of below deck crew cabins, crew mess and galley', 'duration': 4, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping works of the below deck crew cabins, crew mess and galley'], 'subcontractor': {'furniture-subcontractor': 0.3}, 'resources': {'furniture-mat': {'amount': 0.3, 'constant': 0.5}}, 'value': 80}, {'task_name': 'Furniture - Mascoat, insulation, electrical and piping works of the main deck', 'duration': 5, 'prerequisites': ['Sandblasting'], 'subcontractor': {'sandblasting-subcontractor': 0.2, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'sandblasting-mat': {'amount': 0.2, 'constant': 0.1}, 'electrical-mat': {'amount': 0.2, 'constant': 1}, 'piping-mat': {'amount': 0.2, 'constant': 1}, 'insulation-mat': {'amount': 0.2, 'constant': 1}}, 'value': 40}, {'task_name': 'Furniture - Furniture works of main deck VIP cabins', 'duration': 24, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping works of the main deck'], 'subcontractor': {'furniture-subcontractor': 0.25}, 'resources': {'furniture-mat': {'amount': 0.25, 'constant': 0.8}}, 'value': 50}, {'task_name': 'Furniture - Furniture works of main deck Salon, Pantry and corridor', 'duration': 24, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping works of the main deck'], 'subcontractor': {'furniture-subcontractor': 0.25}, 'resources': {'furniture-mat': {'amount': 0.25, 'constant': 2}}, 'value': 50}, {'task_name': 'Furniture - Mascoat, insulation, electrical and piping works of the Upper deck', 'duration': 5, 'prerequisites': ['Mascoat, insulation, electrical and piping works of the main deck'], 'subcontractor': {'sandblasting-subcontractor': 0.2, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'sandblasting-mat': {'amount': 0.2, 'constant': 0.1}, 'electrical-mat': {'amount': 0.2, 'constant': 1}, 'piping-mat': {'amount': 0.2, 'constant': 1}, 'insulation-mat': {'amount': 0.2, 'constant': 1}}, 'value': 60}, {'task_name': 'Furniture - Furniture works of master cabin', 'duration': 23, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping works of the Upper deck'], 'subcontractor': {'furniture-subcontractor': 0.25}, 'resources': {'furniture-mat': {'amount': 0.25, 'constant': 1}}, 'value': 75}, {'task_name': 'Furniture - Furniture works of sky lounge, pantry and corridor', 'duration': 23, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping works of the Upper deck'], 'subcontractor': {'furniture-subcontractor': 0.25}, 'resources': {'furniture-mat': {'amount': 0.25, 'constant': 1}}, 'value': 65}, {'task_name': 'Furniture - Mascoat, insulation, electrical and piping  works of the sun deck', 'duration': 5, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping works of the Upper deck'], 'subcontractor': {'sandblasting-subcontractor': 0.2, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'sandblasting-mat': {'amount': 0.2, 'constant': 0.1}, 'electrical-mat': {'amount': 0.2, 'constant': 1}, 'piping-mat': {'amount': 0.2, 'constant': 1}, 'insulation-mat': {'amount': 0.2, 'constant': 1}}, 'value': 75}, {'task_name': 'Furniture - Furniture works of wheelhouse', 'duration': 23, 'prerequisites': ['Furniture - Mascoat, insulation, electrical and piping  works of the sun deck'], 'subcontractor': {'furniture-subcontractor': 0.25}, 'resources': {'furniture-mat': {'amount': 0.25, 'constant': 1}}, 'value': 260}, {'task_name': 'Technical zones equipments installation outfitting', 'duration': 40, 'prerequisites': ['Aluminium Superstructure - Outfitting'], 'subcontractor': {'outfitting': 1}, 'resources': {'outfitting-equ': {'amount': 1, 'constant': 31}}, 'value': 80}, {'task_name': 'Stainless Steel Works', 'duration': 51, 'prerequisites': ['Steel Hull - Block 4'], 'subcontractor': {'stainless-subcontractor': 0.5}, 'resources': {'stainless-mat': {'amount': 0.5, 'constant': 4}}, 'value': 95}, {'task_name': 'Electrical Works', 'duration': 51, 'prerequisites': ['Steel Hull - Block 4'], 'subcontractor': {'electrical-subcontractor': 0.2}, 'resources': {'electrical-mat': {'amount': 0.2, 'constant': 2}}, 'value': 70}, {'task_name': 'Piping Works', 'duration': 51, 'prerequisites': ['Steel Hull - Block 4'], 'subcontractor': {'piping-subcontractor': 0.2}, 'resources': {'piping-mat': {'amount': 0.2, 'constant': 2}}, 'value': 60}, {'task_name': 'Filler and primer works - Interprime 820-1', 'duration': 40, 'prerequisites': ['Sandblasting'], 'subcontractor': {'painting-subcontractor': 0.2}, 'resources': {'painting-mat': {'amount': 0.2, 'constant': 2}}, 'value': 100}, {'task_name': 'Filler and primer works - Interprime 830', 'duration': 1, 'prerequisites': ['Sandblasting'], 'subcontractor': {'painting-subcontractor': 0.2}, 'resources': {'painting-mat': {'amount': 0.2, 'constant': 1}}, 'value': 50}, {'task_name': 'Filler and primer works - Interprime 833', 'duration': 25, 'prerequisites': ['Filler and primer works - Interprime 820'], 'subcontractor': {'painting-subcontractor': 0.2}, 'resources': {'painting-mat': {'amount': 0.2, 'constant': 1.5}}, 'value': 50}, {'task_name': 'Filler and primer works - Interprime 820-2', 'duration': 8, 'prerequisites': ['Filler and primer works - Interprime 833'], 'subcontractor': {'painting-subcontractor': 0.2}, 'resources': {'painting-mat': {'amount': 0.2, 'constant': 1.5}}, 'value': 55}, {'task_name': 'Filler and primer works - Perfection undercoat', 'duration': 38, 'prerequisites': ['Sandblasting'], 'subcontractor': {'painting-subcontractor': 0.2}, 'resources': {'painting-mat': {'amount': 0.2, 'constant': 1.5}}, 'value': 60}, {'task_name': 'Top coat paint application (Perfection Pro)', 'duration': 8, 'prerequisites': ['Filler and primer works - Perfection undercoat'], 'subcontractor': {'painting-subcontractor': 0.2}, 'resources': {'painting-mat': {'amount': 0.2, 'constant': 1.5}}, 'value': 80}, {'task_name': 'Dock Trials', 'duration': 2, 'prerequisites': ['Top coat paint application (Perfection Pro)'], 'subcontractor': {'outfitting': 1, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'outfitting-equ': {'amount': 1, 'constant': 1}, 'electrical-mat': {'amount': 0.2, 'constant': 1}, 'piping-mat': {'amount': 0.2, 'constant': 1}, 'insulation-mat': {'amount': 0.2, 'constant': 1}}, 'value': 90}, {'task_name': 'Launch and Inclining Experiment', 'duration': 1, 'prerequisites': ['Dock Trials'], 'subcontractor': {'outfitting': 1, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'outfitting-equ': {'amount': 1, 'constant': 2}, 'electrical-mat': {'amount': 0.2, 'constant': 1}, 'piping-mat': {'amount': 0.2, 'constant': 1}, 'insulation-mat': {'amount': 0.2, 'constant': 1}}, 'value': 100}, {'task_name': 'Sea Trials', 'duration': 4, 'prerequisites': ['Launch and Inclining Experiment'], 'subcontractor': {'outfitting': 1, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'outfitting-equ': {'amount': 1, 'constant': 0.1}, 'electrical-mat': {'amount': 0.2, 'constant': 1}, 'piping-mat': {'amount': 0.2, 'constant': 1}, 'insulation-mat': {'amount': 0.2, 'constant': 1}}, 'value': 100}, {'task_name': 'Delivery', 'duration': 3, 'prerequisites': ['Sea Trials'], 'subcontractor': {'outfitting': 1, 'electrical-subcontractor': 0.2, 'piping-subcontractor': 0.2, 'insulation-subcontractor': 0.2}, 'resources': {'outfitting-equ': {'amount': 1, 'constant': 0.1}, 'electrical-mat': {'amount': 0.2, 'constant': 0.1}, 'piping-mat': {'amount': 0.2, 'constant': 0.1}, 'insulation-mat': {'amount': 0.2, 'constant': 0.1}}, 'value': 100}]

subcontractor = {
    'steel-subcontractor': 3,  # Total number of subcontractors available
    'aluminium-subcontractor': 2,
    'outfitting': 3,
    'sandblasting-subcontractor': 1,
    'furniture-subcontractor': 2,
    'electrical-subcontractor': 2,
    'piping-subcontractor': 2,
    'insulation-subcontractor': 2,
    'stainless-subcontractor': 1,
    'painting-subcontractor': 2,
}

resources = {
    'steel': 0,  # Total number of subcontractors available
    'aluminium': 0,
    'outfitting-equ': 0,
    'sandblasting-mat': 0,
    'furniture-mat': 0,
    'electrical-mat': 0,
    'piping-mat': 0,
    'insulation-mat': 0,
    'stainless-mat': 0,
    'painting-mat': 0,
}

def update_resource_requirements(tasks, gt_value):
    for task in tasks:
        for resource, details in task['resources'].items():
            details['amount'] = details['constant'] * gt_value

# Base ProjectManager Agent
class ProjectManager(Agent):
    def __init__(self, unique_id, model, start_task):
        super().__init__(unique_id, model)
        self.gt = random.randint(300, 500)
        self.current_task_index = start_task
        self.tasks = deepcopy(model.tasks)
        update_resource_requirements(self.tasks, self.gt)
        self.task_duration_elapsed = 0

    def current_task(self):
        # Check if the current task index is within the range of available tasks
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        else:
            return None  # Or handle appropriately if no tasks are available

    def step(self):
        if self.current_task_index is not None and self.current_task_index < len(self.tasks):
            task = self.current_task()
            self.task_duration_elapsed += 1
            if self.task_duration_elapsed >= task['duration']:
                if self.model.resources_available(task['resources']):
                    self.complete_task(task)
                self.current_task_index += 1
                self.task_duration_elapsed = 0
            else:
                self.purchase_resources(task)

    def purchase_resources(self, task):
        for resource, req in task['resources'].items():
            amount_needed = req['amount']
            if self.model.resources[resource] < amount_needed:
                self.model.purchase_resource(self, resource, amount_needed)

    def complete_task(self, task):
        for resource, req in task['resources'].items():
            amount_needed = req['amount']
            if self.model.resources[resource] >= amount_needed:
                self.model.resources[resource] -= amount_needed

    def bid_for_resources(self, task):
        for resource, req in task['resources'].items():
            amount_needed = req['amount']
            bid_value = task['value'] * random.uniform(0.8, 1.2)
            if self.model.total_capacity - self.model.used_capacity >= amount_needed:
                self.model.bids.append((self, resource, bid_value, amount_needed))

    def receive_resources(self, resource, amount):
        # Update the agent's resource inventory
        self.model.resources[resource] += amount
        print(f"Agent {self.unique_id} received {amount} units of {resource}")

# Q-Learning Project Manager extending base ProjectManager
class QLearningProjectManager(ProjectManager):
    def __init__(self, unique_id, model, start_task, q_table=None, learning_rate=0.01, discount_factor=0.95, epsilon=0.1):
        super().__init__(unique_id, model, start_task)
        self.q_table = defaultdict(lambda: defaultdict(float)) if q_table is None else q_table
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_history = []
        self.action_history = []  # Record actions taken
        self.reward_history = []  # Record rewards received
        self.total_reward = 0  # Sum of rewards received over an episode

    def step(self):
        if current_task := self.current_task():
            current_state = self.get_state_description()
            action = self.select_action()
            self.perform_action(action)
            reward = self.model.evaluate_performance()
            self.total_reward += reward  # Accumulate reward
            new_state = self.get_state_description()
            self.update_q_values(action, reward, new_state)
            self.action_history.append((current_state, action))  # Store action
            self.reward_history.append((current_state, reward))  # Store reward
            super().step()
        else:
            self.active = False


    def select_action(self):
        current_state = self.get_state_description()
        if not self.q_table[current_state]:
            self.q_table[current_state] = {'bid': 0.0, 'purchase': 0.0}

        action = np.random.choice(list(self.q_table[current_state].keys())) if np.random.rand() < self.epsilon else max(self.q_table[current_state], key=self.q_table[current_state].get)
        return action

    def update_q_values(self, action, reward, new_state):
        old_value = self.q_table[self.get_state_description()][action]
        future_max = max(self.q_table[new_state].values(), default=0)
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * future_max)
        self.q_table[self.get_state_description()][action] = new_value

        # Log the update for visualization
        self.state_history.append((self.get_state_description(), action, self.q_table[self.get_state_description()]))

    def get_state_description(self):
        return f"Task-{self.current_task_index}"

    def perform_action(self, action):
        if action == 'bid':
            self.bid_for_resources(self.current_task())
        elif action == 'purchase':
            self.purchase_resources(self.current_task())

def plot_q_values(agent):
    fig, ax = plt.subplots(figsize=(12, 8))
    states = sorted(set(state for state, _, _ in agent.state_history))
    actions = sorted(set(action for _, action, _ in agent.state_history))

    for action in actions:
        values = [agent.q_table[state][action] for state in states]
        ax.plot(states, values, label=f"Action: {action}")

    ax.set_xlabel("State")
    ax.set_ylabel("Q-value")
    ax.set_title("Q-values Over Time by Action")
    ax.legend()
    plt.show()

def plot_performance(agent):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot rewards
    rewards = [reward for _, reward in agent.reward_history]
    ax1.plot(rewards, label='Rewards Over Time')
    ax1.set_title('Rewards Over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.legend()

    # Plot actions
    actions = [action for _, action in agent.action_history]
    ax2.plot(actions, label='Actions Over Time', marker='o', linestyle='None')
    ax2.set_title('Actions Taken Over Time')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Action')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# ShipBuilding Model including resource management and bidding
class ShipBuildingModel(Model):
    def __init__(self):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.total_capacity = 150000
        self.initial_resources = {key: 0 for key in resources.keys()}
        self.used_capacity = 0
        self.resources = {key: 0 for key in resources.keys()}
        self.tasks = tasks
        self.bids = []
        self.data_collector = DataCollector(
            model_reporters={"Resources": lambda m: deepcopy(m.resources)}
        )

        for i in range(5):
            pm = QLearningProjectManager(i, self, start_task=i * (len(tasks) // 5))
            self.schedule.add(pm)

    def resources_available(self, required_resources):
        total_needed = sum(req['amount'] for _, req in required_resources.items())
        return self.total_capacity - self.used_capacity >= total_needed

    def purchase_resource(self, agent, resource, amount):
        if self.total_capacity - self.used_capacity >= amount:
            self.resources[resource] += amount
            self.used_capacity += amount

    def bid_and_allocate_resources(self):
        total_bids = defaultdict(float)
        for bid in self.bids:
            _, resource, bid_value, _ = bid
            total_bids[resource] += bid_value

        for bid in self.bids:
            agent, resource, bid_value, amount_needed = bid
            if total_bids[resource] > 0:
                proportion = bid_value / total_bids[resource]
                amount_allocated = min(amount_needed, proportion * self.resources[resource])
                if self.resources[resource] >= amount_allocated:
                    self.resources[resource] -= amount_allocated
                    agent.receive_resources(resource, amount_allocated)

        self.bids.clear()

    def reset_resources(self):
        self.resources = deepcopy(self.initial_resources)
        self.used_capacity = 0

    # # Example reward function based on resource efficiency
    # def evaluate_performance(self):
    #     # Simple reward function: reward agents for having resources
    #     resource_efficiency = sum(self.resources.values()) - self.used_capacity
    #     return resource_efficiency

    def reset_model(self):
        # Reset resources and other necessary components
        self.schedule = RandomActivation(self)
        self.resources = {key: 0 for key in resources.keys()}
        self.used_capacity = 0
        self.bids = []
        self.data_collector = DataCollector(
            model_reporters={"Resources": lambda m: deepcopy(m.resources),
                             "Total Reward": lambda m: sum(agent.total_reward for agent in m.schedule.agents)}
        )

        # Reinitialize agents with retained Q-tables
        for i in range(5):
            q_table = None
            if hasattr(self, 'schedule') and len(self.schedule.agents) > i:
                q_table = self.schedule.agents[i].q_table
            pm = QLearningProjectManager(i, self, start_task=i * (len(tasks) // 5), q_table=q_table)
            pm.total_reward = 0  # Initialize total reward for the new episode
            self.schedule.add(pm)

    def evaluate_performance(self):
        # Example: Reward could be inversely proportional to the resources used
        used_resources = sum(self.resources.values())
        return max(0, 1000 - used_resources)  # Simple reward function

    def step(self):
        self.reset_resources()
        self.schedule.step()
        self.bid_and_allocate_resources()
        self.data_collector.collect(self)
        print(f"Step completed. Current resource levels: {self.resources}, Used capacity: {self.used_capacity}")


# Plotting function to visualize resource levels over time
def plot_resources(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    for resource in resources.keys():
        data['Resources'].apply(lambda x: x[resource]).plot(ax=ax, label=resource)
    plt.title('Resource Levels Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Resource Quantity')
    plt.legend()
    plt.show()

def main():
    model = ShipBuildingModel()
    num_episodes = 100  # Define the number of episodes
    episode_rewards = []

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        model.reset_model()  # Reset the model to initial state, keeping Q-values
        for i in range(600):  # Number of steps per episode
            model.step()

        data = model.data_collector.get_model_vars_dataframe()
        total_reward = data['Total Reward'].iloc[-1]
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} completed. Total Reward: {total_reward}")

        if episode == num_episodes - 1:  # Optionally, plot only the last episode's data
            plot_resources(data)
            pm = model.schedule.agents[0]  # Example: plot for the first agent
            plot_q_values(pm)
            plot_performance(pm)

    # Plot episode-by-episode improvements
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o')
    plt.title('Improvement Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == "__main__":
    main()
