# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
        GhostPos=successorGameState.getGhostPositions()
        foods=newFood.asList()
        dist=[]
        iterator=0
        ghostdist=[]  
        import sys   

        "Winning state is best state so we return a big value."   

        if successorGameState.isWin():
            return sys.maxsize

        """For the evaluation function we estimate the distance between the closest food and ghost.The first
        should positively affect the result (the closer,the higher score),therefore we subtract it, where the latter 
        negatively affects it (the closer,the worst) so we multiply it by the moves it will stay still (the more the better) 
        and we add it."""

        for food in foods:
            new=manhattanDistance(newPos,food)
            dist.append(new)

        for ghost in GhostPos:

            "If we fall into a ghost, worst case scenario, we return a really small value."

            if ghost == newPos and newScaredTimes[iterator]==0:

                return -sys.maxsize
            else:

                new=manhattanDistance(newPos,ghost) * newScaredTimes[iterator]

            iterator+=1
            ghostdist.append(new)

        """Other than distance,we want to show the impact of the number of dots and ghosts, so we check if it isn't 0
        and divide with each term respectively.(The more food => the smaller the term we subtract => bigger value,
        the less ghosts => the bigger the term we add => bigger value.)."""

        #no ghosts => good thing=> default value big but not as much as in winning state

        ming=min(ghostdist,default=sys.maxsize/4)

        #no food => bad thing=> default value small but not as much as in losing state (keep in mind it will be subtracted)

        minf=min(dist,default=sys.maxsize/4)
        num_food=len(dist)
        ghosts=len(ghostdist)

        if not num_food :
            num_food=1

        if not ghosts :
            ghosts=1            

        evaluation = ming / ghosts - (minf / num_food)

        return successorGameState.getScore() + evaluation

        


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        import sys

        def minAgent(state,agent,depth) :

            actions=state.getLegalActions(agent)

            #Having no actions left or  depth=0 => terminal state

            if not depth or not actions :

                return self.evaluationFunction(state)

            #initialize with biggest possible value
            value = sys.maxsize

            for action in actions:

                #We visit agents in a cyclical increasing order starting from 0 hence the mod.
                agentNum = state.getNumAgents()
                newAgent=(agent + 1) % agentNum

                if not newAgent:

                    """Agent=0 => Pacman playing => call maxAgent with next state and 
                    depth decreased by 1 since both Pacman and ghosts have made one move"""

                    value = min( maxAgent(state.generateSuccessor(agent,action),depth-1) , value)
                else :

                    #Call minAgent with next state for the next ghost.

                    value = min( minAgent(state.generateSuccessor(agent,action),newAgent,depth) , value)

            return value

        #Symmetrical Solution with minAgent only now we don't need to pass an agent index since we know it's 0
        def maxAgent(state,depth) :

            actions=state.getLegalActions(0)

            if not depth or not actions:

                return self.evaluationFunction(state)

            #Initialize with the smallest possible value
            value = -sys.maxsize-1

            for action in actions:

                #Call minAgent for the first ghost(1), for the next state (agentIndex=0)

                value = max( minAgent(state.generateSuccessor(0,action),1,depth) , value)

            return value

        #Initial Function
        #wanted=[ max value, corresponding action] (we choose list because it's mutable)    

        wanted=[]

        for action in gameState.getLegalActions(0) :

            newValue = minAgent(gameState.generateSuccessor(0,action),1,self.depth)

            #Initialize in case it's empty
            if not wanted:
                wanted.append(newValue)
                wanted.append(action)

            #Found a bigger one => replace

            elif wanted[0] < newValue:
                wanted[0]=newValue
                wanted[1]=action
        
        return wanted[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    """The algorithm is essentially identical with minimax only we pass two more parameters in the functions , a and b
    and we prune according to the algorithm in lectures."""

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        import sys

        def minAgent(state,agent,depth,a,b) :

            actions=state.getLegalActions(agent)

            #Having no actions left or  depth=0 => terminal state

            if not depth or not actions :

                return self.evaluationFunction(state)

            #initialize with biggest possible value
            value = sys.maxsize
            new_b=b

            for action in actions:

                #We visit agents in a cyclical increasing order starting from 0 hence the mod.
                agentNum = state.getNumAgents()
                newAgent=(agent + 1) % agentNum

                if not newAgent:

                    """Agent=0 => Pacman playing => call maxAgent with next state and 
                    depth decreased by 1 since both Pacman and ghosts have made one move"""

                    value = min( maxAgent(state.generateSuccessor(agent,action),depth-1 , a , new_b) , value)
                else :

                    #Call minAgent with next state for the next ghost.

                    value = min( minAgent(state.generateSuccessor(agent,action),newAgent,depth, a, new_b) , value)

                #min pruning

                if value < a :
                    return value
                    
                new_b=min(new_b,value)

            return value

        #Symmetrical Solution with minAgent only now we don't need to pass an agent index since we know it's 0
        def maxAgent(state,depth,a,b) :

            actions=state.getLegalActions(0)

            if not depth or not actions:

                return self.evaluationFunction(state)

            #Initialize with the smallest possible value
            value = -sys.maxsize-1
            new_a=a

            for action in actions:

                #Call minAgent for the first ghost(1), for the next state (agentIndex=0)

                value = max( minAgent(state.generateSuccessor(0,action),1,depth ,new_a , b) , value )

                #max pruning

                if value > b :
                    return value

                new_a=max(new_a,value)

            return value

        #Initial Function
        #wanted=[ max value, corresponding action] (we choose list because it's mutable)    

        wanted=[]
        a= -sys.maxsize-1
        b=sys.maxsize

        for action in gameState.getLegalActions(0) :

            newValue = minAgent(gameState.generateSuccessor(0,action),1,self.depth,a,b)

            #Initialize in case it's empty
            if not wanted:
                wanted.append(newValue)
                wanted.append(action)

            #Found a bigger one => replace

            elif wanted[0] < newValue:
                wanted[0]=newValue
                wanted[1]=action
            
            #max pruning 

            if newValue > b : 
                return wanted[1]

            a=max(a,newValue)   

        return wanted[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    """The algorithm is essentially identical with minimax only we have to replace the min function
    in the minAgent (turned into chanceAgent now) to the sum of probabilities * values."""

    def getAction(self, gameState: GameState):
        
        import sys

        def chanceAgent(state,agent,depth) :

            actions=state.getLegalActions(agent)

            #Having no actions left or  depth=0 => terminal state

            if not depth or not actions :

                return self.evaluationFunction(state)

            #value= Sum of probability * values instead of min
            value = 0

            #uniform distribution => prob= 1/(b-a) where [a,b] spectrum of values
            prob=1/len(actions)

            for action in actions:

                #We visit agents in a cyclical increasing order starting from 0 hence the mod.
                agentNum = state.getNumAgents()
                newAgent=(agent + 1) % agentNum

                if not newAgent:

                    """Agent=0 => Pacman playing => call maxAgent with next state and 
                    depth decreased by 1 since both Pacman and ghosts have made one move"""

                    value +=  prob * maxAgent(state.generateSuccessor(agent,action),depth-1) 
                else :

                    #Call minAgent with next state for the next ghost.

                    value += prob * chanceAgent(state.generateSuccessor(agent,action),newAgent,depth) 

            return value

        #Symmetrical Solution with minAgent only now we don't need to pass an agent index since we know it's 0
        def maxAgent(state,depth) :

            actions=state.getLegalActions(0)

            if not depth or not actions:

                return self.evaluationFunction(state)

            #Initialize with the smallest possible value
            value = -sys.maxsize-1

            for action in actions:

                #Call minAgent for the first ghost(1), for the next state (agentIndex=0)

                value = max( chanceAgent(state.generateSuccessor(0,action),1,depth) , value)

            return value

        #Initial Function
        #wanted=[ max value, corresponding action] (we choose list because it's mutable)    

        wanted=[]

        for action in gameState.getLegalActions(0) :

            newValue = chanceAgent(gameState.generateSuccessor(0,action),1,self.depth)

            #Initialize in case it's empty
            if not wanted:
                wanted.append(newValue)
                wanted.append(action)

            #Found a bigger one => replace

            elif wanted[0] < newValue:
                wanted[0]=newValue
                wanted[1]=action
        
        return wanted[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    GhostPos=currentGameState.getGhostPositions()
    foods=newFood.asList()
    dist=[]
    iterator=0
    ghostdist=[]  
    import sys   

    "Winning state is best state so we return a big value."   

    if currentGameState.isWin():
        return sys.maxsize

    """For the evaluation function we estimate the distance between the closest food and ghost.The first
    should positively affect the result (the closer,the higher score),therefore we subtract it, where the latter 
    negatively affects it (the closer,the worst) so we multiply it by the moves it will stay still (the more the better) 
    and we add it."""

    for food in foods:
        new=manhattanDistance(newPos,food)
        dist.append(new)

    for ghost in GhostPos:

        "If we fall into a ghost, worst case scenario, we return a really small value."

        if ghost == newPos and newScaredTimes[iterator]==0:

            return -sys.maxsize
        else:

            new=manhattanDistance(newPos,ghost) * newScaredTimes[iterator]

        iterator+=1
        ghostdist.append(new)

    """Other than distance,we want to show the impact of the number of dots and ghosts, so we check if it isn't 0
    and divide with each term respectively.(The more food => the smaller the term we subtract => bigger value,
    the less ghosts => the bigger the term we add => bigger value.)."""

    #no ghosts => good thing=> default value big but not as much as in winning state

    ming=min(ghostdist,default=sys.maxsize/4)

    #no food => bad thing=> default value small but not as much as in losing state (keep in mind it will be subtracted)

    minf=min(dist,default=sys.maxsize/4)
    num_food=len(dist)
    ghosts=len(ghostdist)

    if not num_food :
        num_food=1

    if not ghosts :
        ghosts=1            

    evaluation = ming / ghosts - (minf / num_food)

    return currentGameState.getScore() + evaluation    



# Abbreviation
better = betterEvaluationFunction
