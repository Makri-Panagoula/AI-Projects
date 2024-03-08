# Overview

This repository contains code for search algorithms and related agents used in pathfinding problems. Below is a summary of the key components:

## search.py

The `search.py` file implements various search algorithms. Here's an overview of the main data structures used:

- **Explored Nodes List**: Tracks nodes that have already been explored.
- **Path-Finding List**: Helps in finding the path to the end goal by continuously concatenating new actions onto the path.
- **Frontier**: Determines which nodes to expand next. This varies for each algorithm; refer to the implementation for details.

Additionally, most functions include another list containing the nodes in the frontier for easier access.

## searchAgents.py

In `searchAgents.py`, we analyze the consistency and admissibility of a heuristic. Here's what we've discovered:

- **Consistency**: The heuristic function ensures that the difference between heuristic values of consecutive states is always ≤ 1, maintaining consistency.
  
- **Admissibility**: Our heuristic guarantees that the estimated cost to reach the goal state is never overestimated. By considering the Manhattan distance from the current state to every unvisited corner or dot, we ensure reliability and accuracy.

Feel free to explore the code and documentation for more details on implementation and usage.
