
 

Integrating Agent Based Models (ABM) with Q-Learning in Yacht Construction: Enhancing Operational Allocations






Course Code: SI 652 / EECS 547
Course Name: Incentives and Strategic Behavior in Computational Systems
Lecturer Name: Professor Grant Schoenebeck
Student Name: Ramazan Mengi (rmengi)
Table of Contents
1. INTRODUCTION	4
1.1.	PROJECT GOALS AND PROBLEM STATEMENT	4
1.2.	IMPORTANCE OF THE PROBLEM	4
1.3.	AGENT-BASED MODELING AND Q-LEARNING APPROACH	4
1.4.	LIBRARIES AND TOOLS USED	4
1.5.	ADDRESSING THE INDUSTRY GAP	5
2.	RELATED WORK	5
2.1.	AGENT-BASED MODELS AND MACHINE LEARNING	5
2.2.	DEEP REINFORCEMENT LEARNING IN MULTI-AGENT SYSTEMS	5
2.3.	APPLICATIONS IN COMPLEX PROBLEM SOLVING	6
2.4.	RELEVANCE TO PROJECT MANAGEMENT AND CONSTRUCTION	6
3.	MODEL AND METHODS	6
3.1.	OVERVIEW	6
3.2.	MODEL DESCRIPTION	6
3.3.	KEY COMPONENTS:	6
3.4.	AGENT DESIGN	8
3.5.	METHODS	8
4.	ANALYSIS / RESULTS	9
4.1.	RESOURCE LEVELS OVER TIME (GRAPH: R100.PNG)	9
4.2.	REWARDS OVER TIME (GRAPH: RE100.PNG)	10
4.3.	Q-VALUES OVER TIME BY ACTION (GRAPH: Q100.PNG)	11
IMPROVEMENT OVER EPISODES	11
4.4.	COMPARATIVE RESULTS	12
5.	SUMMARY OF RESULTS	13
5.1.	KEY OBSERVABLES:	13
5.2.	INTERPRETATION OF OVERALL OUTCOMES	13
5.3.	FUTURE WORK	13
5.4.	CONCLUSION	14
6.	REFERENCES	14




1. Introduction

1.1.	Project Goals and Problem Statement
The construction of luxury yachts is a complex endeavor that combines advanced engineering, design, and operational management to meet high standards of luxury and performance. The project aims to optimize yacht construction processes by addressing the critical challenge of inefficient resource allocation among project managers. Specifically, the goal is to develop an agent-based model leveraging Cooperative Artificial Intelligence (AI) and Q-learning techniques. This model will enable project managers to predict other project managers' bids and strategically allocate resources and bids more effectively, thus enhancing overall efficiency in yacht building.
1.2.	Importance of the Problem
The yacht building industry is characterized by its high stakes and precision requirements, where the smallest inefficiencies can lead to significant cost overruns and delays. Current resource allocation methods are often reactive and made under time constraints, leading to suboptimal decision-making and resource wastage. By introducing a predictive and cooperative element through AI, project managers can anticipate needs and adjust their strategies in real time, thereby transforming the traditional approach to project management in this sector.
1.3.	Agent-Based Modeling and Q-Learning Approach
In response to these challenges, the proposed project utilizes an agent-based modeling framework implemented through the Mesa library. This framework simulates interactions between multiple project managers (agents) who bid for resources essential for completing various construction tasks. Each agent operates based on a set of tasks derived from real yacht construction processes, such as assembling different hull blocks or installing technical equipment.
The Q-learning aspect, integrated into the project managers' decision-making processes, allows agents to learn from past actions and optimize their bids for resources over time. This reinforcement learning technique is crucial for adapting to the dynamic bidding environment and improving resource allocation efficiency.
1.4.	Libraries and Tools Used
The project employs Python for its implementation with several libraries:
Mesa: For creating the agent-based models.
Matplotlib: To visualize results and the learning process of agents.
NumPy: For handling numerical operations especially within the learning algorithms.
1.5.	Addressing the Industry Gap
The yacht building industry's current project management and resource allocation practices often lack the capability to adapt quickly to changing circumstances without sacrificing efficiency. This project directly addresses this gap by providing a tool that supports dynamic decision-making and improves the strategic planning capabilities of project managers. With the predictive power of Q-learning, project managers can forecast the bidding behavior of competitors, allowing for smarter, data-driven decisions that optimize resource use and reduce wastage.

2.	Related Work
Agent-based models (ABMs) and their integration with Q-learning have seen significant application across various domains, including production, construction, and project management. This section explores key publications and research developments related to your project on integrating Cooperative AI in yacht construction.
2.1.	Agent-Based Models and Machine Learning
ABMs are particularly well-suited for simulating complex systems where individual entity behaviors and interactions lead to emergent properties. These models capture the discrete nature of system components and their stochastic interactions, which is essential for mimicking real-world dynamics in biological, social, and engineered systems. The integration of machine learning with ABMs enhances their predictive capabilities, enabling more effective decision-making based on dynamic system responses to varied conditions [1].
2.2.	Deep Reinforcement Learning in Multi-Agent Systems
The application of deep reinforcement learning (DRL), particularly through multi-agent deep Q-networks (DQNs), extends the capabilities of traditional Q-learning by handling environments with delayed rewards more effectively. Multi-agent DQNs have been shown to significantly improve learning performance and decision-making speed in complex environments. These models allow for a nuanced approach to reinforcement learning where agents learn and adapt based on a combination of individual and shared experiences [2].
2.3.	Applications in Complex Problem Solving
The use of multi-agent deep reinforcement learning (MADRL) is especially notable in scenarios involving cooperative or competitive interactions among agents. This approach has been successful in network resource allocation, traffic signal control, and other areas requiring high-level coordination among multiple decision-makers. MADRL facilitates the sharing of knowledge, such as Q-values and reward signals, among agents, enhancing the collective decision-making process [3].
2.4.	Relevance to Project Management and Construction
In the context of project management and construction, integrating ABMs with reinforcement learning offers a powerful tool for optimizing resource allocation and operational efficiency. This approach allows project managers to simulate and predict outcomes of different strategies under varying conditions, which is particularly useful in complex, dynamic environments like yacht construction. The ability to predict competitor behavior and adapt bidding strategies in real-time can lead to significant improvements in resource utilization and project timelines.

3.	Model and Methods

3.1.	Overview
The simulation model, implemented using the Mesa library, manages a set of project managers competing for limited resources in a dynamic yacht construction environment. The goal is to enhance resource allocation and project efficiency using agent-based modeling and Q-learning.
3.2.	Model Description
The ShipBuildingModel is the core of the simulation. It manages tasks, resources, and agents (project managers). Each project manager is initialized with different tasks, simulating a realistic environment where each manager might be at different stages of the yacht construction process.
3.3.	Key Components:
•	Tasks: There are 40 tasks in total. Detailed tasks such as "Steel Hull - Block 1" each with attributes like duration, prerequisites, subcontractors, resources, and a value. The tasks are received from a Shipyard in Tuzla, Istanbul. All the tasks are real-life tasks [4]. Tasks are derived from a Gantt Chart.
 
•	Duration: Each task has its own duration which is equivalent to number of steps.
•	Prerequisites: Each task has prerequisites; therefore, a task cannot be completed without completing the tasks before.
•	Resources: Shared resources like steel, aluminum, and outfitting equipment, crucial for task completion. Resources initially start with 0 value.
•	resources = {
    'steel': 0,  # Resources
    'aluminum': 0,
    'outfitting-equ': 0,
    'sandblasting-mat': 0,
    'furniture-mat': 0,
    'electrical-mat': 0,
    'piping-mat': 0,
    'insulation-mat': 0,
    'stainless-mat': 0,
    'painting-mat': 0,
}
•	Project Managers: Agents with strategies to bid for, purchase, and utilize resources.
3.4.	Agent Design
Each ProjectManager agent handles tasks sequentially, requiring resources which they obtain through bidding. The agents are designed to simulate real-world decision-making in project management, with actions influenced by available resources and task urgency.
The QLearningProjectManager extends the ProjectManager by incorporating a learning mechanism where the agent improves its bidding strategy based on past experiences, optimizing resource procurement and task execution over time.
3.5.	Methods
Experiment Setup
Number of Agents: Five project managers.
Stochastic Environment: Each manager starts at different stages, creating a non-deterministic environment that enhances the simulation's realism and complexity.
Resource Allocation Mechanisms
VCG Mechanism: Initially tested but found inadequate for improving overall shipyard performance due to its non-strategic nature in resource distribution.
Proportional Allocation: Implemented after VCG, where resources are allocated based on the proportionality of bids, allowing more strategic and flexible resource distribution.
Q-Learning Integration
Q-learning was introduced to adapt and optimize bidding strategies autonomously. Agents learn from the outcomes of their actions (bids) and adjust their strategies to maximize efficiency. The Q-learning setup involves:
•	States: Defined by the current task and available resources.
•	Actions: Bidding or purchasing decisions.
•	Rewards: Based on the completion of tasks and efficient use of resources.
Learning Parameters:
•	Learning Rate (αα): 0.01
•	Discount Factor (γγ): 0.95
•	Exploration Rate (ϵϵ): 0.1
Experimentation
The model was evaluated over 100 episodes, each simulating the complete construction process from start to finish. The number of steps required to complete all tasks served as the primary efficiency metric.
Performance Metrics:
•	Step Reduction: Initially over 600 steps required, reduced to under 550 steps after Q-learning stabilization.
•	Resource Utilization: Monitored to assess the efficiency of resource allocation strategies.
Evaluation Metrics
•	Number of Steps: Measures the operational efficiency; fewer steps indicate better resource management and task scheduling.
•	Total Reward: Cumulative reward across episodes, indicating the overall success of the learning agents in optimizing their actions.

4.	Analysis / Results
4.1.	Resource Levels Over Time (Graph: r100.png)
The first graph indicates the resource allocation across the simulation's timeline. The significant spikes in 'steel' and 'outfitting-equ' suggest a concentrated need for these materials at specific intervals. The fluctuation is indicative of the project phases requiring different resources, with 'steel' being in high demand early on, likely due to its use in hull construction. The efficient usage of 'steel' aligns with the tasks' prerequisites, suggesting the simulation's effectiveness in allocating resources where and when they're needed.
The relative scarcity of resources like 'aluminum' and 'painting-mat' at various points reflects the project's progression and the impact of project managers' bidding strategies on resource distribution.
 
4.2.	Rewards Over Time (Graph: re100.png)
The second graph displays a pattern of rewards over time, with peaks corresponding to the completion of high-value tasks and troughs possibly indicating periods of resource scarcity or task inactivity. The overall trend of rewards is an important metric, as it correlates with the efficiency of resource allocation and task management. Consistent rewards throughout the simulation could indicate a balance between task completion and resource usage.
 
4.3.	Q-values Over Time by Action (Graph: q100.png)
The Q-values graph shows the learning progression of agents for different actions. Initially, we see a preference for bidding, which gradually evolves as the agents learn the optimal balance between bidding and purchasing resources. The stable Q-values later in the simulation suggest that the agents have learned effective strategies for resource allocation.
 
Improvement Over Episodes
The progression of the total reward accumulated by the agents across 100 episodes. Each point on the graph represents the total reward obtained at the end of an episode, reflecting the combined effectiveness of all project managers in the simulation.
Analysis of Improvement Over Episodes
The trend in the graph demonstrates fluctuations in total reward from one episode to the next, indicating variations in performance. The overall pattern does not suggest a clear upward trend, which might have been expected with learning agents. However, there are several high points suggesting episodes of particularly efficient resource management and task completion.
•	Variability: The oscillation in total rewards across episodes might imply that while agents are learning and improving, the stochastic nature of the environment or the complexity of the task scheduling could introduce variability in performance.
•	Learning Stabilization: Despite the variability, the agents' performance does not show a downward trend, which might have suggested poor learning. Instead, the agents seem to stabilize around certain reward benchmarks.
•	Optimal Policies: Occasional peaks in rewards could represent instances where agents found particularly successful strategies or combinations of actions. Identifying what led to these peaks could provide insights into the most effective strategies.
 
4.4.	Comparative Results
Before Q-Learning Integration:
The system required over 600 steps to complete all tasks.
Project managers' actions were less informed by previous outcomes, leading to less efficient resource allocation.
After Q-Learning Integration (Episode 100):
Steps to completion dropped to under 550, showing an improvement in the efficiency of resource utilization.
Total Reward: 694,240.10, indicates that the project managers have collectively improved their strategies, aligning actions with optimal outcomes.
The bidding strategies of the project managers led to a more dynamic and responsive allocation of resources.

5.	Summary of Results
The simulation project involving agent-based modeling and Q-learning within a yacht construction context yielded notable outcomes. Over the course of 100 episodes, project managers demonstrated a learning curve, indicated by the fluctuation in total rewards and the reduction in steps required to complete tasks.
5.1.	Key Observables:
The resource level graphs showed strategic resource usage, with pivotal spikes in 'steel' and 'outfitting-equipment' reflecting task-critical moments.
Rewards over time fluctuated, with high peaks indicating successful task completions and strategic resource allocations.
Q-values over time exhibited the agents' learning progress, with a shift from initial explorative actions to more balanced strategies between bidding and purchasing.
Improvement over episodes displayed variability in total rewards, with episodes of high rewards highlighting optimal strategies, despite the absence of a consistent upward trajectory.
The final episode achieved a significant total reward, signaling the potential for agents to optimize their performance in a complex environment.
5.2.	Interpretation of Overall Outcomes
The results imply that agent-based models, enhanced by Q-learning, are capable of navigating complex, multi-agent environments effectively. Despite the inherent variability and stochastic nature of the simulation, the agents adapted and improved, showcasing the resilience and potential of AI-driven project management tools.
5.3.	Future Work
Given an additional 6-12 months to continue working on this project, several features could be introduced to enhance the model's complexity and realism:
•	Subcontractor Quotation Bidding: Integrating a system where project managers can solicit and evaluate bids from subcontractors, considering both cost and quality, to find the best fit for their tasks.
•	Client Decision List and Deadlines: Including client interactions where project managers must negotiate and adjust their strategies based on client demands and strict deadlines.
•	Milestone Achievements and Payments: Implementing a financial module that ties the completion of milestones to payments from clients, affecting the project's cash flow and agent strategies.
•	Worker Implementation: Adding labor as a resource that needs to be managed, with factors such as worker availability, skill levels, and labor costs influencing project outcomes.
•	More Dynamic and Chaotic Environment: Creating a simulation that includes unpredictable elements such as market fluctuations, supply chain disruptions, and other real-world challenges.
•	Advanced Q-learning: Further developing the Q-learning algorithm to better adapt to the enriched environment, potentially exploring Deep Q-Networks (DQN) or other sophisticated reinforcement learning techniques.
5.4.	Conclusion
The project demonstrates that agent-based simulations coupled with Q-learning provide a solid foundation for exploring resource management and decision-making in complex scenarios. The initial results are promising, indicating that with further development and the addition of more nuanced features, the model could serve as a powerful tool for optimizing project management in the yacht construction industry and beyond.

6.	References
1.	Madani, K., & Cammarata, G. (2022). Agent-based modeling in complex biological systems: Advancements and applications. Frontiers in Systems Biology, 3, 959665. https://doi.org/10.3389/fsysb.2022.959665
2.	Vlahogianni, E. I., & Karlaftis, M. G. (2022). Enhancing transportation systems via deep learning: A deep reinforcement learning approach. Applied Sciences, 12(7), 3520. https://doi.org/10.3390/app12073520
3.	Jia, N., Wang, L., & Zhang, J. (2021). Multi-agent systems for resource allocation and scheduling in a distributed environment. Applied Sciences, 11(22), 10870. https://doi.org/10.3390/app112210870
4.	Tasks are received from Mengi Yay Shipyard in Tuzla, Istanbul Turkey. They are real-life tasks used in agreement with the client for NB113 (Virtus XP 52m)
The code can found at 
