# Reinforcement Learning in Airline Revenue Management

MSc dissertation (First Class), University of Warwick – MMORSE.

A Q-learning agent for single-leg airline seat inventory control with overbooking, including a newly proposed **opportunity-cost heuristic** to guide the agent's accept/reject decisions.

## Problem

Airlines sell a fixed number of seats to stochastic, price-differentiated demand arriving over a booking horizon. Sell too cheaply too early and high-fare demand is displaced; hold out too long and the flight departs with empty seats. Overbooking adds a second trade-off: accepting bookings beyond capacity hedges against no-shows, at the risk of denied-boarding costs.

The problem is formulated as a Markov Decision Process:

- **State**: remaining time in the booking horizon and current bookings by fare class
- **Actions**: accept or reject an incoming booking request
- **Reward**: fare revenue, less overbooking penalties realised at departure
- **Transitions**: driven by empirically derived, non-homogeneous arrival rates per fare class

## Approach

A tabular Q-learning agent is trained against a simulated booking process, benchmarked against standard revenue-management policies. The proposed opportunity-cost heuristic shapes the reward signal to reflect the expected displacement cost of committing a seat, accelerating convergence toward bid-price-like behaviour.

## Notebook structure

The full implementation is in [`rl_in_arm.ipynb`](rl_in_arm.ipynb):

1. **Import / clean arrival rates** – preparing empirical demand arrival data
2. **Model classes** – environment (booking simulator with overbooking and no-shows) and agent
3. **Model implementation** – training the Q-learning agent
4. **Bayesian optimisation** – tuning learning rate, discount factor, and exploration schedule
5. **Parameter sensitivity** – robustness of the learned policy to demand and cost assumptions
6. **Effects of opportunity cost** – impact of the proposed heuristic on revenue and convergence

Note: some scenario comparisons require adjusting parameters in the relevant cells and re-running.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
colorama==0.4.4
bayesian-optimization
```

## Context

This work sits within the operations research literature on dynamic pricing and capacity control (Littlewood's rule, EMSR heuristics, bid-price controls), exploring whether model-free RL can recover and extend these policies without an explicit demand model.
