# logicPlan.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

from typing import Dict, List, Tuple, Callable, Generator, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}


#______________________________________________________________________________
# QUESTION 1

def sentence1() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    A  = Expr('A')          #make each variable an expression first so that it can be used on functions
    B = Expr('B')
    C = Expr('C')
    expressions = []                            #we place every literal into the list and then conjoin them all together since every single one has to be true
    expressions.append( A | B )
    expressions.append( ~ A % ( ~ B | C ) )
    expressions.append( disjoin( ~ A, ~ B, C) )
    return conjoin(expressions)


def sentence2() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    A  = Expr('A')              #same as previously , we simply apply the functions using the priority of operands as it is known
    B = Expr('B')
    C = Expr('C')
    D = Expr('D')
    expressions = []                                    
    expressions.append( C % ( B | D ) )
    expressions.append(  A >> ( ~ B &  ~ D ) )
    expressions.append( ( ~ (B & ~ C) >> A) )
    expressions.append( ( ~ D >> C) )
    return conjoin(expressions)    



def sentence3() -> Expr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    (Project update: for this question only, [0] and _t are both acceptable.)
    """
    #we just use the corresponding functions and make each literal and then conjoin since they all have to be true.

    alive_1 = PropSymbolExpr("PacmanAlive[1]")
    alive_0 = PropSymbolExpr("PacmanAlive[0]")
    born = PropSymbolExpr("PacmanBorn[0]")
    killed = PropSymbolExpr("PacmanKilled[0]")
    cond_1 = alive_1 % ((alive_0 & ~ killed) | (~ alive_0 & born))
    cond_2 = ~ ( alive_0 & born)
    return conjoin(cond_1,cond_2,born)     



def findModel(sentence: Expr) -> Dict[Expr, bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    cnf_sentence = to_cnf(sentence)
    return pycoSAT(cnf_sentence)

def findModelCheck() -> Dict[Any, bool]:
    """Returns the result of findModel(Expr('a')) if lower cased expressions were allowed.
    You should not use findModel or Expr in this method.
    This can be solved with a one-line return statement.
    """
    class dummyClass:
        """dummy('A') has representation A, unlike a string 'A' that has repr 'A'.
        Of note: Expr('Name') has representation Name, not 'Name'.
        """
        def __init__(self, variable_name: str = 'A'):
            self.variable_name = variable_name
        
        def __repr__(self):
            return self.variable_name
    return  { dummyClass('a') : True }
    

def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """
    if(findModel(premise & ~ conclusion) == False) :    #the inversion of the entailment must be unsattisfiable for the entailment to be true
        return True
    return False

def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """
    return not pl_true(inverse_statement,assignments)

#______________________________________________________________________________
# QUESTION 2

def atLeastOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals  ist is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    return disjoin(literals)            #literals are already in cnf form and all we have to do is apply or operator to all of them , since we only need one being true



def atMostOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """

    pairs = itertools.combinations(literals,2)          #take every combination of literals

    to_conjoin = []
    for pair in pairs :                                 #make sure that there don't exist 2 that are both true in a pair by checking every possible pair
        to_conjoin.append(disjoin(~pair[0],~pair[1]))

    return conjoin(to_conjoin)                          #all the "pairs" must be legal



def exactlyOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    return conjoin(atLeastOne(literals),atMostOne(literals))                #we have exactly one true literal , when we simultaneously have more than one and less than one being true 

#______________________________________________________________________________
# QUESTION 3

def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]=None) -> Expr:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    now, last = time, time - 1
    possible_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t
    # the if statements give a small performance boost and are required for q4 and q5 correctness
    if walls_grid[x][y+1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))                        
    if walls_grid[x+1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last)) 
    if walls_grid[x-1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))                           
    if not possible_causes:
        return None
    
    return  (PropSymbolExpr(pacman_str, x, y, time=now) % disjoin(possible_causes) )   #its current coords is a result of one of those movements(there must be exactly one movement)



def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    now, last = time, time - 1
    moved_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    if walls_grid[x][y+1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(pacman_str, x, y, time=last) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    failed_move_causes: List[Expr] = [] # using merged variables, improves speed significantly
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
        walls_grid: 2D array of either -1/0/1 or T/F. Used only for successorAxioms.
            Do NOT use this when making possible locations for pacman to be in.
        sensorModel(t, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
        successorAxioms(t, walls_grid, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
        - Results of calling sensorModel(...), unless None.
        - Results of calling successorAxioms(...), describing how Pacman can end in various
            locations on this time step. Consider edge cases. Don't call if None.
    """
    pacphysics_sentences = []
    pacman_pos = []
    non_pacman_pos = []

    for x,y in all_coords :
        non_pacman_pos.append(PropSymbolExpr(wall_str,x,y) >> ~PropSymbolExpr(pacman_str, x, y, time=t))
    
    for x,y in non_outer_wall_coords :
        pacman_pos.append(PropSymbolExpr(pacman_str, x, y, time=t))

    #For every wall we ensure Pacman won't be there
    pacphysics_sentences.append(conjoin(non_pacman_pos))

    #Pacman is at exactly one location that isn't a wall => the first two conditions must simultaneously be satisfied
    pacphysics_sentences.append(exactlyOne(pacman_pos))

    directions = []
    for action in DIRECTIONS :
        directions.append(PropSymbolExpr(action, time=t))
    
    #Pacman takes exactly one action at timestep t
    pacphysics_sentences.append(exactlyOne(directions))

    #Results of calling sensorModel(...), unless None.
    if sensorModel != None :        
        pacphysics_sentences.append(sensorModel(t,non_outer_wall_coords))

    #We make sure t > 0 so that it won't become negative in successorAxioms
    if successorAxioms != None and t > 0:        
        pacphysics_sentences.append(successorAxioms(t,walls_grid,non_outer_wall_coords))
        
    return conjoin(pacphysics_sentences)



def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - action1 = to ensure match with autograder solution
        - problem = an instance of logicAgents.LocMapProblem
    Note:
        - there's no sensorModel because we know everything about the world
        - the successorAxioms should be allLegalSuccessorAxioms where needed
    Return:
        - a model where Pacman is at (x1, y1) at time t = 1
        - a model where Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    #Check that at both given timestamps data is valid , all the restrictions are being satisfied by calling pacphysicsAxioms for both time = 0 and time = 1

    KB.append(pacphysicsAxioms(0,all_coords,non_outer_wall_coords,walls_grid,None,allLegalSuccessorAxioms))
    KB.append(pacphysicsAxioms(1,all_coords,non_outer_wall_coords,walls_grid,None,allLegalSuccessorAxioms))
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time = 0))              #Pacman’s current location (x0, y0)
    KB.append(PropSymbolExpr(action0, time = 0))                         #Pacman takes action0
    KB.append(PropSymbolExpr(action1, time = 1))                         #Pacman takes action1

    #premise has to be true no matter what so we conjoin with everything else we want to be true for every model.
    premise = conjoin(KB)                
    model1 = findModel(premise & PropSymbolExpr(pacman_str, x1, y1, time=1))
    model2 = findModel(premise & ~PropSymbolExpr(pacman_str, x1, y1, time=1))

    return (model1,model2)



#______________________________________________________________________________
# QUESTION 4

def positionLogicPlan(problem) -> List:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls_grid = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls_grid.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
            range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []
    #adding Pacman's initial coords to KB
    KB.append(PropSymbolExpr(pacman_str,x0,y0,time = 0))

    for t in range(50) :
        
        print("Time step is : ",t)
        non_pacman_pos = []
        pacman_pos = []
        directions = []

        for x,y in walls_list :
            non_pacman_pos.append(~PropSymbolExpr(pacman_str, x, y, time=t))

        #For every wall we ensure Pacman won't be there at every timestep t
        KB.append(conjoin(non_pacman_pos))

        #Ensure Pacman will only be at exactly one location at every timestep t
        for x,y in non_wall_coords :
            pacman_pos.append(PropSymbolExpr(pacman_str, x, y, time=t))
        KB.append(exactlyOne(pacman_pos))

        #Check if we are currently at goal state, xg,yg refering to goal coords
        if( t > 0 ) :
            goal = conjoin(conjoin(KB) , PropSymbolExpr(pacman_str, xg, yg, time=t))
            model = findModel(goal)
            if( model != False ) :
                return(extractActionSequence(model,DIRECTIONS))
                
        #Ensure Pacman will take exactly one action at every timestep t
        for action in DIRECTIONS :
            directions.append(PropSymbolExpr(action, time=t))
        KB.append(exactlyOne(directions))   

        #Transition Model Sentences (for time > 0 so that it won't get negative in the SuccessorAxiomSingle)
        # checking for the next iteration to make sure that if we are in goal it can come legally from previous state 
        for x,y in non_wall_coords :
            KB.append(pacmanSuccessorAxiomSingle(x,y,t + 1,walls_grid)) 

#______________________________________________________________________________
# QUESTION 5

def foodLogicPlan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []
    #adding Pacman's initial coords to KB
    KB.append(PropSymbolExpr(pacman_str,x0,y0,time = 0))
    #Initializing Food[x,y]_t variables, setting them true for the coords where there is food
    for x,y in food:
        KB.append(PropSymbolExpr(food_str, x, y, time=0))


    for t in range(50) :
        
        print("Time step is : ",t)
        non_pacman_pos = []
        pacman_pos = []
        directions = []

        for x,y in walls_list :
            non_pacman_pos.append(~PropSymbolExpr(pacman_str, x, y, time=t))

        #For every wall we ensure Pacman won't be there at every timestep t
        KB.append(conjoin(non_pacman_pos))

        #Ensure Pacman will only be at exactly one location at every timestep t
        for x,y in non_wall_coords :
            pacman_pos.append(PropSymbolExpr(pacman_str, x, y, time=t))
        KB.append(exactlyOne(pacman_pos))

        #Check if all food that existed is eaten, every Food[x,y]_t variable should be False, then we are at goal state
        if( t > 0 ) :
            goal = []
            for x,y in food:
                goal.append(~PropSymbolExpr(food_str, x, y, time=t))
            model = findModel(conjoin(conjoin(KB) , conjoin(goal)))
            if( model != False ) :
                return(extractActionSequence(model,DIRECTIONS))
                
        #Ensure Pacman will take exactly one action at every timestep t
        for action in DIRECTIONS :
            directions.append(PropSymbolExpr(action, time=t))
        KB.append(exactlyOne(directions))   

        #Transition Model Sentences as in question 4 to ensure pacman's coords are legal
        #Food Successor Axiom : if pacman's coords for time t are food coords then food is eaten => no food in x,y for time = t+1, otherwise
        #if there wasn't already food for time = t => there still won't be any for time = t + 1 obviously. Likewise,
        #if there isn't food at x,y for time = t + 1 => it was either eaten at time = t or it didn't exist for time = t neither.

        for x,y in non_wall_coords :
            KB.append(pacmanSuccessorAxiomSingle(x,y,t + 1,walls)) 
            KB.append(((PropSymbolExpr(food_str, x, y, time=t) & PropSymbolExpr(pacman_str, x, y, time=t) ) | ~PropSymbolExpr(food_str, x, y, time=t)) % ~PropSymbolExpr(food_str, x, y, time=t+1)  ) 
#______________________________________________________________________________
# QUESTION 6

def localization(problem, agent) -> Generator:
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    KB = []

    #We add to KB where walls are
    for x,y in walls_list :
        KB.append(PropSymbolExpr(wall_str,x,y))

    #We add to KB where walls aren't
    for x,y in all_coords :
        if (x,y) not in walls_list :
            KB.append(~PropSymbolExpr(wall_str,x,y))        

    for t in range(agent.num_timesteps):

        #We add pacphysics,actions and percept information to KB 
        KB.append(pacphysicsAxioms(t,all_coords,non_outer_wall_coords,walls_grid,sensorAxioms,allLegalSuccessorAxioms))             #localization => use sensorAxioms & allLegalSuccessorAxioms
        KB.append(PropSymbolExpr(agent.actions[t] , time = t))
        KB.append(fourBitPerceptRules(t,agent.getPercepts()))               #localization => use fourBitPerceptRules
        possible_locations = []

        #Find possible locations with updated KB
        for x,y in non_outer_wall_coords :

            #If there is a satisfying assignment where restrictions are met and pacman is at x,y for time t , we have a possible location
            if(findModel(conjoin(KB) & PropSymbolExpr(pacman_str, x, y, time=t)) != False) :
                possible_locations.append((x,y))

            #Pacman is provably at (x,y) at time t when KB => Pacman[x,y]_t and is provably not at x,y when KB => - Pacman[x,y]_t
            if(entails(conjoin(KB),PropSymbolExpr(pacman_str, x, y, time=t))) :
                KB.append(PropSymbolExpr(pacman_str, x, y, time=t))

            if(entails(conjoin(KB),~PropSymbolExpr(pacman_str, x, y, time=t))) :
                KB.append(~PropSymbolExpr(pacman_str, x, y, time=t))

        agent.moveToNextState(agent.actions[t])
        yield possible_locations

#______________________________________________________________________________
# QUESTION 7

def mapping(problem, agent) -> Generator:
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))        

    KB.append(PropSymbolExpr(pacman_str,pac_x_0, pac_y_0 , time=0))         #We add Pacman's initial location to KB

    #We add to KB whether or not there is a wall at initial location (that we know of for sure)
    if known_map[pac_x_0][pac_y_0] == 1 :
        KB.append(PropSymbolExpr(wall_str,pac_x_0, pac_y_0))
    elif known_map[pac_x_0][pac_y_0] == 0 :
        KB.append(~PropSymbolExpr(wall_str,pac_x_0, pac_y_0))
        

    for t in range(agent.num_timesteps):

        #We add pacphysics,actions and percept information to KB 
        KB.append(pacphysicsAxioms(t,all_coords,non_outer_wall_coords,known_map,sensorAxioms,allLegalSuccessorAxioms))      #mapping => use sensorAxioms & allLegalSuccessorAxioms
        KB.append(PropSymbolExpr(agent.actions[t] , time = t))
        KB.append(fourBitPerceptRules(t,agent.getPercepts()))           #mapping => use fourBitPerceptRules

        #Find provable walls with updated KB
        for x,y in non_outer_wall_coords :

            #There is provably a wall in x,y when KB => Wall[x,y] and there is provably not one when KB => - Wall[x,y]

            if(entails(conjoin(KB),PropSymbolExpr(wall_str, x, y))) :
                KB.append(PropSymbolExpr(wall_str, x, y))
                known_map[x][y] = 1                                     #since we know there is a wall we add coords with value 1

            elif(entails(conjoin(KB),~PropSymbolExpr(wall_str, x, y))) :
                KB.append(~PropSymbolExpr(wall_str, x, y))
                known_map[x][y] = 0                                     #since we know there isn't a wall we add coords with value 0
            
            else :
                known_map[x][y] = -1                                    #since we can't know whether or not there is a wall we add coords with value -1,ambiguous result

        agent.moveToNextState(agent.actions[t])
        yield known_map

#______________________________________________________________________________
# QUESTION 8

def slam(problem, agent) -> Generator:
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    KB.append(PropSymbolExpr(pacman_str,pac_x_0, pac_y_0 , time=0))             #We add Pacman's initial location to KB

    #We add to KB whether or not there is a wall at initial location (that we know of for sure)
    if known_map[pac_x_0][pac_y_0] == 1 :
        KB.append(PropSymbolExpr(wall_str,pac_x_0, pac_y_0))
    elif known_map[pac_x_0][pac_y_0] == 0 :
        KB.append(~PropSymbolExpr(wall_str,pac_x_0, pac_y_0))

    for t in range(agent.num_timesteps):

        #We add pacphysics,actions and percept information to KB 
        KB.append(pacphysicsAxioms(t,all_coords,non_outer_wall_coords,known_map,SLAMSensorAxioms,SLAMSuccessorAxioms))          #SLAM => use SLAMSensorAxioms & SLAMSuccessorAxioms
        KB.append(PropSymbolExpr(agent.actions[t] , time = t))
        KB.append(numAdjWallsPerceptRules(t,agent.getPercepts()))               #SLAM => use numAdjWallsPerceptRules
        agent.moveToNextState(agent.actions[t])

        for x,y in non_outer_wall_coords :              #exactly as in q7

            if(entails(conjoin(KB),PropSymbolExpr(wall_str, x, y))) :
                KB.append(PropSymbolExpr(wall_str, x, y))
                known_map[x][y] = 1

            elif(entails(conjoin(KB),~PropSymbolExpr(wall_str, x, y))) :
                KB.append(~PropSymbolExpr(wall_str, x, y))
                known_map[x][y] = 0
            
            else :
                known_map[x][y] = -1

        possible_locations = []     #exactly as in q6

        for x,y in non_outer_wall_coords :

            if(findModel(conjoin(KB) & PropSymbolExpr(pacman_str, x, y, time=t)) != False) :
                possible_locations.append((x,y))

            if(entails(conjoin(KB),PropSymbolExpr(pacman_str, x, y, time=t))) :
                KB.append(PropSymbolExpr(pacman_str, x, y, time=t))

            if(entails(conjoin(KB),~PropSymbolExpr(pacman_str, x, y, time=t))) :
                KB.append(~PropSymbolExpr(pacman_str, x, y, time=t))
                
        yield (known_map, possible_locations)


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

#______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time = t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], time=t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker numAdjWallsPerceptRules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = SLAMSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)

#______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def modelToString(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualizeCoords(coords_list, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualizeBoolArray(bool_arr, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()
