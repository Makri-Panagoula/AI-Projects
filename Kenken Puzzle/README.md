# Kenken Puzzle
Our goal in this assignment is given a file with the Kenken's problem to produce a solution in the most time-effective way.
## Command Line Arguments

When executing the program, provide the following arguments in the specified order: input file, file with solutions, and the algorithm (+ heuristic) you want to use. Choose from `FC`, `FC+MRV`, `MAC`, `MAC+LCV`.

The `Kenken` class includes an additional function called `check`, which performs the following checks:
- Whether the program output matches the solutions file (prints "EVERYTHING IS PERFECT" if matching).
- Whether the program didn't find a solution (returns `None`).
- Where the program output and solutions differ.

Both the input file and solutions file must adhere to the following prototype (as input_files examples):

        Kenken Board Size

        Target#CliqueParticipants#Operation

        Target#CliqueParticipants#Operation

        Target#CliqueParticipants#Operation

        ...

        Target#CliqueParticipants#Operation

        Target: The target number of each clique.

        CliqueParticipants: Cells participating in each clique.

        Operation: The operation performed in each clique.

The Operation can also be an equality sign for a single cell that must be implemented, even if it is not mentioned in the statement, as confirmed by the course collaborators.

The `CliqueParticipants` has the format A-B-G- ... -O, with each cell of the board being a number.

## Modelization

Variables' modeling and parsing follow the same logic as described above (from 0 to size * size -1).

Neighboring variables of each variable include every other variable (but just once) with which it is involved in a constraint. This includes variables in the same row, column, and grid. The `Grid` class keeps track of the list of variables in each grid, its operation, and its target. The `Kenken` class maintains a dictionary (`var_info`) that corresponds the variable to its `Grid`.

## Constraint Function

The constraint function checks the following:
- If the two neighbors are the same variable and the grid operation is "=", the function checks whether the variable's value is equal to the target.
- If the two neighbors are in the same row/column, it checks whether the variables are equal.
- If they don't belong to the same row/column, the function examines whether they belong to the same grid and checks whether their combination, along with the already assigned variables, is acceptable according to the target. Division and subtraction are only done with two variables.
