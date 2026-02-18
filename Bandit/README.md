# Multi-Armed Bandit Experiment

## Overview
This project implements a multi-armed bandit experiment using the epsilon-greedy strategy with decay and Thompson Sampling. The application allows users to interact with the bandit problem through a graphical user interface (GUI) built with Tkinter. Users can manually pull arms or let an agent run simulations to explore the bandit environment.

## Project Structure
```
bandit
├── src
│   ├── bandits_app.py        # Entry point for the application
│   ├── bandit_logic.py       # Core logic for the bandit problem
│   └── bandit_gui.py         # GUI implementation using Tkinter
└── README.md                 # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd bandit
   ```

2. Install the required packages:
   ```
   pip install matplotlib
   ```

3. Run the application:
   ```
   python src/bandits_app.py
   ```

## Usage
- The application provides controls for configuring the agent's parameters, including the number of loops, memory size, epsilon value, and decay factor.
- Users can manually pull each bandit arm or allow the agent to run simulations based on the selected strategy (epsilon-greedy or Thompson Sampling).
- The GUI displays a live plot of cumulative rewards and a summary of the bandit performance.

## Multi-Armed Bandit Problem
The multi-armed bandit problem is a classic problem in probability theory and decision-making. It involves a scenario where a gambler must choose between multiple slot machines (bandits), each with an unknown probability of payout. The goal is to maximize the total reward over time by balancing exploration (trying different arms) and exploitation (choosing the best-known arm).

## Implemented Strategies
- **Epsilon-Greedy**: This strategy selects a random arm with probability epsilon and the best-known arm with probability (1 - epsilon). Epsilon decays over time to favor exploitation as the agent learns.
- **Thompson Sampling**: A Bayesian approach that maintains a probability distribution for each arm and selects arms based on their expected rewards.

## Acknowledgments
This project is inspired by the exploration-exploitation dilemma in reinforcement learning and aims to provide a practical implementation of the multi-armed bandit problem.