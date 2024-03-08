--Command line arguments :
    While executing the program add the following arguments in corresponding order : input file, file with solutions , algorithm (+ heuristic ) you want to use choosing from FC,FC+MRV,MAC,MAC+LCV. There is an extra function in Kenken class ,check, that checks whether program output is the same as solutions(prints "EVERYTHING IS PERFECT" in that case),whether it didn't find a solution(None),or where the program output and solutions differ. Input file and solutions file must follow the same prototype as here: 
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

        The CliqueParticipants has the format A-B-G- ... -O, with each cell of the board being a number.    
        
    The parsing is being made based on this model , variables' modeling follow the exact same logic as well (from 0 to size * size -1). 

--Modelization:
    Neighbors of each variable include every other variable (but just once) with which it is involved in a constraint , variables in the same row,column and grid. Obviously it doesn't include the variable itself unless the grid's operation is "=".There is an extra class called Grid ,where we keep the list of variables in this grid,its operation and its target.In the Kenken class we keep a dictionary(var_info) that corresponds the variable to its Grid. 

--Constraint Function :
    We check whether the two neighbors are the same variable, then the grid operation is "=" , so we merely have to check whether variable's value is equal with target.Otherwise,we check if they're in the same row / column we check whether the variables are equal and if not we examine whether they belong to the same grid and check whether their combination as well as with the already assigned variables is acceptable accordind to target.
    Division and subtraction are being done only with two variables.