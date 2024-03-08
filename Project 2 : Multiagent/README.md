## Minimax Algorithm with Helper Functions

In the implementation of the minimax algorithm and its variations, I've opted to create two helper functions—one for Pacman and one for the ghosts. These functions call one another, contributing to a cleaner and more straightforward code structure.

### Evaluation Functions

The evaluation functions employed in the algorithm exhibit significant similarity. The primary distinction arises in the treatment of successor states: question 1 focuses on examining the successor state, while question 5 involves evaluating the current state.
