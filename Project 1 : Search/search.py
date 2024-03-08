
# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """

    """Essentially following lecture's code, explored saves the states that have already been visited 
    so as to avoid infinite loops,frontier includes the nodes that have been expanded but not yet 
    explored, we implement with a stack for its lifo identity since we are doing dfs.We have an additional
    structure to trace back the actions to goal state,cur_path """

    explored=[]
    frontier=util.Stack()
    start=problem.getStartState()
    cur_path=[]
    frontier.push((start,cur_path))

    while not frontier.isEmpty():

        (state,cur_path)=frontier.pop()

        if problem.isGoalState(state):
            return cur_path

        explored.append(state)

        for new_state,new_action,new_cost in problem.getSuccessors(state):
            
            if new_state not in explored:

                "Frontier includes nodes, with state and the updated path"
                frontier.push((new_state, cur_path + [new_action]))
            
    return cur_path 


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    """Same as before only now we use a queue, because we need the fifo order, since we have to visit 
    all the child nodes before moving to the next level. Now, to expand a node it shouldn't be in the frontier either.
    frontierBool keeps the information of the nodes in the frontier since we can't have direct access through the queue."""

    explored=[]
    frontier=util.Queue()
    frontierBool=[]
    start=problem.getStartState()
    cur_path=[]
    frontier.push((start,cur_path))    
    frontierBool.append(start)

    while not frontier.isEmpty():

        (state,cur_path)=frontier.pop()

        if problem.isGoalState(state):
            return cur_path

        frontierBool.remove(state)
        explored.append(state) 

        for new_state,new_action,new_cost in problem.getSuccessors(state):
            
            if new_state not in explored and new_state not in frontierBool:

                frontier.push((new_state, cur_path + [new_action]))
                frontierBool.append(new_state)

    return cur_path    
            
            
def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""

    "Similar as before, other than we have a priority queue as frontier with the path cost as priority."

    explored=[]
    frontier=util.PriorityQueue()
    frontierBool=[]
    start=problem.getStartState()
    cur_path=[]
    frontier.push((start,cur_path),0)    
    frontierBool.append(start)

    while not frontier.isEmpty():

        (state,cur_path)=frontier.pop()

        if problem.isGoalState(state):
            return cur_path

        frontierBool.remove(state)
        explored.append(state)

        for new_state,new_action,new_cost in problem.getSuccessors(state):
            
            new_path=cur_path + [new_action]
            path_cost=problem.getCostOfActions(new_path)
            

            if new_state not in explored :

                if new_state not in frontierBool:

                    frontier.push((new_state,new_path),path_cost)
                    frontierBool.append(new_state)

                else :

                    for node in frontier.heap:
                        

                        """We have to find the node in the priority queue and through him the previous cost to compare
                        it with the new one, we can't simply update because we insert tuples not the mere state and therefore
                        if we have a new path to the same state it won't be identified with the previous node,so we won't compare the cost to the state. 
                        Node consists of a (x,y,(statename,path))"""

                        if new_state==node[2][0] and problem.getCostOfActions(node[2][1])>path_cost:

                            frontier.heap.remove(node)
                            frontier.update((new_state, new_path),path_cost)

    return cur_path    
 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0



def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    """Similar to ucs function only that here the priority is equal to cost+heuristic"""

    explored=[]
    frontier=util.PriorityQueue()
    frontierBool=[]
    start=problem.getStartState()
    cur_path=[]
    frontier.push((start,cur_path),heuristic(start,problem))    
    frontierBool.append(start)

    while not frontier.isEmpty():

        (state,cur_path)=frontier.pop()

        if problem.isGoalState(state):
            return cur_path

        frontierBool.remove(state)
        explored.append(state)
        
        for new_state,new_action,new_cost in problem.getSuccessors(state):
            
            new_path=cur_path + [new_action]
            f=problem.getCostOfActions(new_path)+heuristic(new_state,problem)
            

            if new_state not in explored :

                if new_state not in frontierBool:

                    frontier.push((new_state,new_path),f)
                    frontierBool.append(new_state)

                else :

                    for node in frontier.heap:
                        
                        prev_f=problem.getCostOfActions(node[2][1]) + heuristic(new_state,problem)

                        "Exactly as in the ucs algorithm"

                        if new_state==node[2][0] and prev_f>f:

                            frontier.heap.remove(node)
                            frontier.update((new_state, new_path),f)


    return cur_path      


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

