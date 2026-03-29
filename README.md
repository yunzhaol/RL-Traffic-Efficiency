# CS9170: Optimizing Urban Traffic Efficiency with Reinforcement Learning

## Problem Definition:
Traditional fixed-time traffic signals are inefficient. The goal is to use RL to dynamically control traffic light timing at intersections based on real-time queue lengths and waiting times, minimizing overall vehicle delay and congestion.

- **State**: Traffic conditions (vehicle counts, waiting vehicles, direction, signal phase)
- **Action**: Traffic light phase (e.g., North-South vs East-West green)
- **Reward**: Negative congestion (based on vehicle count and waiting time)

---

## Methodology

### Simulation Environment
- **CityFlow** 
- **SUMO (Simulation of Urban MObility)**

### RL Model
- **Deep Q-Network (DQN)**
- **Proximal Policy Optimization(PPO)**

### Training
- Epsilon-greedy exploration
- Gradient clipping for stability
- Reward scaling for better convergence

---

