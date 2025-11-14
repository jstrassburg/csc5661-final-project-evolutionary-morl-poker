# csc5661-final-project-evolutionary-morl-poker

This project was conceived as a final project for the CSC 5661 - Reinforcement Learning course as part of their [Master of Science in Machine Learning](https://www.msoe.edu/academics/graduate-degrees/m-s-in-machine-learning/) program.

## Overview

Poker is a partially observable game. We can observe our own cards, any shared or communal cards, the pot size, any bets, who is left in the hand, who bet last, and who raised. It is said that poker is a game of beating the player and not their cards. Different playing styles can be observed as well (and experienced poker players will react to different playing styles/archetypes differently).

In this project, I will build a baseline evolutionary reinforcement learning agent to play poker. I will then build a multi-objective evolutionary reinforcement learning agent and compare the performance. The multi-objective agent will seek out a different policy for each poker archetype.

## Poker Archetypes

- Tight-aggressive (TAG) - A normally tight player who gets aggressive with quality hands.
- Loose-aggressive (LAG) - A player who aggressively plays lots of hands regardless of quality.
- Loose-passive (calling station) - A player who plays lots of pots but doesn't aggressively raise quality hands.
- Tight-passive (nit or rock) - A player who plays few hands and only commits chips with quality hands.

Sometimes we also consider:

- The Maniac - A player who aggressively raises and re-raises regardless of hand strength.
- The Wizard - A player who knows about these archetypes and will change their playing style to avoid detection.

## Learning Objectives

- Understand evolutionary reinforcement learning
- Understand multi-objective reinforcement learning