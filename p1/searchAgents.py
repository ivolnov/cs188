# searchAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
This file contains all of the agents that can be selected to
control Pacman.  To select an agent, use the '-p' option
when running pacman.py.  Arguments can be passed to your agent
using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a searchFunction=depthFirstSearch

Commands to invoke other search strategies can be found in the
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
import sys
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import searchAgents

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a
    supplied search problem, then returns actions to follow that path.

    As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in dir(searchAgents):
                heur = getattr(searchAgents, heuristic)
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in dir(searchAgents) or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = getattr(searchAgents, prob)
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game board. Here, we
        choose a path to the goal.  In this phase, the agent should compute the path to the
        goal and store it in a local variable.  All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in registerInitialState).  Return
        Directions.STOP if there is no further action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # Number of search nodes expanded

        "*** YOUR CODE HERE ***"
        self._start = self.startingPosition + (self.corners,)

    def getStartState(self):
        "Returns the start state (in your state space, not the full Pacman state space)"
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self._start

    def isGoalState(self, state):
        "Returns whether this search state is a goal state of the problem"
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return not state[2]

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            x, y = state[0], state[1]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if self.walls[nextx][nexty]:
                continue

            corners = [corner for corner in state[2] if corner != (nextx, nexty)]
            successor = ((nextx, nexty, tuple(corners)), action, 1)
            successors.append(successor)

        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem; i.e.
    it should be admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    #return 0 # Default to trivial solution
    isWall = problem.walls[state[0]][state[1]]
    #util.pause()
    #return manhattenSumHeuristic(state,problem)
    #return remainingCornersHeuristic(state, problem)
    return isWallHeuristic(state, problem)


def manhattenSumHeuristic(state, problem):
    distances = [util.manhattanDistance(corner, (state[0], state[1])) for corner in state[2]]
    return sum(distances)


def remainingCornersHeuristic(state, problem):
    return len(state[2])


def isWallHeuristic(state, problem):
    return 1 if problem.walls[state[0]][state[1]] else 0


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a
    Grid (see game.py) of either True or False. You can call foodGrid.asList()
    to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, problem.walls gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use. For example,
    if you only want to count the walls once and store that value, try:
      problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    #for height in xrange(state[1].height-1, 0, -1):
    #    for width in xrange(0, state[1].width):
    #        sys.stdout.write(str(state[1][width][height]))
    #        sys.stdout.write(' ')
    #    sys.stdout.write('\n')

    if 'food' not in problem.heuristicInfo:
        storeCapsulePositions(state, problem)

    #if 'index' not in problem.heuristicInfo:
     #   buildReverseIndex(state, problem, 3)

    #util.pause()

    return farthestHeuristic(state, problem)


def lessCapsulesToTheRightHeuristic(state, problem):
    """
    Computes the amount of capsules to the right of pacman.

    :param state:
    :param problem:
    :return: number of capsules to the right
    """
    food = state[1]
    hero = state[0]

    right = 0

    for x in xrange(hero[1], food.width):
        for y in xrange(0, food.height):
            if food[x][y]:
                right += 1

    return right

def closestFoodHeuristic(state, problem):
    """
    Computes the shortest distance from the pacman position to a capsule by iterating over all capsules and ignoring
    those that had been eaten. If the distance had been already computed uses the value from the dictionary.

    :param state:
    :param problem:
    :return: distance to the closest capsule
    """
    shortest = 999999
    for capsule, distances in problem.heuristicInfo['food'].iteritems():
        # capsule already eaten
        if not state[1][capsule[0]][capsule[1]]:
            continue
        # have never computed distance from that point to this capsule
        if state[0] not in distances:
            distance = util.manhattanDistance(state[0], capsule)
            distances[state[0]] = distance
        shortest = min(shortest, distances[state[0]])

    return shortest

def averageDistanceHeuristic(state, problem):
    """
    Computes the sum of all distances from the pacman position to capsules by iterating over them and ignoring
    those that had been eaten. If the distance had been already computed uses the value from the dictionary.

    :param state:
    :param problem:
    :return: average distance to capsule
    """
    total = 0
    count = 0
    for capsule, distances in problem.heuristicInfo['food'].iteritems():
        # capsule already eaten
        if not state[1][capsule[0]][capsule[1]]:
            continue
        # have never computed distance from that point to this capsule
        if state[0] not in distances:
            distance = util.manhattanDistance(state[0], capsule)
            distances[state[0]] = distance
        total += distances[state[0]]
        count += 1

    return total / count


def farthestHeuristic(state, problem):
    """
    Computes the farthest capsule from the pacman position
    by iterating over them and ignoring those that had been eaten.
    If the distance had been already computed uses the value from the dictionary.

    :param state:
    :param problem:
    :return: farthest distance
    """
    closest = 99999
    farthest = 0
    total = 0
    count = 0

    for capsule, distances in problem.heuristicInfo['food'].iteritems():
        # capsule already eaten
        if not state[1][capsule[0]][capsule[1]]:
            continue
        # have never computed a distance from that point to this capsule
        if state[0] not in distances:
            distance = util.manhattanDistance(state[0], capsule)
            distances[state[0]] = distance
        closest = min(closest, distances[state[0]])
        farthest = max(farthest, distances[state[0]])
        total += distances[state[0]]
        count += 1

    return farthest


def closestClusterHeuristic(state, problem):
    index = problem.heuristicInfo['index']
    hero = state[0]
    food = state[1]
    closest = 99999

    if hero not in index:
        return food.height + food.width

    for pivot in index[hero]:
        if not food[pivot[0]][pivot[1]]:
            continue
        closest = min(closest, index[hero][pivot])

    return closest


def storeCapsulePositions(state, problem):
    """
    Called once to store capsules from grid in the more convenient dictionary.

    :param state:
    :param problem:
    :return: None
    """
    food = state[1]
    problem.heuristicInfo['food'] = {}
    # lets collect all capsules as (x, y) tuples to iterate over them later instead of the whole grid matrix.
    for x in xrange(0, food.width):
        for y in xrange(0, food.height):
            if food[x][y]:
                capsule = (x, y)
                # each capsule will map to dictionary that maps points in space to their distances to the capsule
                problem.heuristicInfo['food'][capsule] = {}


def buildReverseIndex(state, problem, influence = 5):
    """
    Maps each capsule to dictionary of pivot capsules of all its clusters in the given range
    and manhattan distances between this capsule and each pivot.

    Creates index dictionary like:
    {(x_capsule, y_capsule) => {
            (x_pivot_1, y_pivot_1) => manhattan distance,
            (x_pivot_2, y_pivot_2) => manhattan distance
        }
    }

    :param state:
    :param problem:
    :param influence: range of clustering
    :return: reverse index
    """
    hero = state[0]
    food = state[1].asList()
    index = {}
    reverse = {}

    # group capsules in the given range around every pivot
    for pivot in food:
        index[pivot] = []
        for neighbor in food:
            if abs(pivot[0] - neighbor[0]) <= influence or abs(pivot[1] - neighbor[1]) <= influence:
                index[pivot].append(neighbor)

    # for each capsule store pivot capsule of each cluster it relates to and its distance
    for pivot, neighbours in index.iteritems():
        for neighbor in neighbours:
            if neighbor not in reverse:
                reverse[neighbor] = {}
            reverse[neighbor][pivot] = util.manhattanDistance(hero, pivot)

    problem.heuristicInfo['index'] = reverse


def isFoodThereHeuristic(state, problem):
    return 0 if state[1][state[0][0]][state[0][1]] else .5


def isWall(state, problem):
    return .5 if problem.walls[state[0][0]][state[0][1]] else 0

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return search.breadthFirstSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
      A search problem for finding a path to any food.

      This search problem is just like the PositionSearchProblem, but
      has a different goal test, which you need to fill in below.  The
      state space and successor function do not need to be changed.

      The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
      inherits the methods of the PositionSearchProblem.

      You can use this search problem to help you fill in
      the findPathToClosestDot method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test
        that will complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[state[0]][state[1]]

##################
# Mini-contest 1 #
##################

class ApproximateSearchAgent(Agent):
    "Implement your contest entry here.  Change anything but the class name."

    def registerInitialState(self, state):
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"
        self.actions = []
        currentState = state
        starttime = time.time()
        expanded = 0

        while currentState.getFood().count() > 0:
            closest = closestByManhatten(currentState)
            problem = PositionSearchProblem(currentState, goal=closest, warn=False)

            actions = search.aStarSearch(problem, manhattanHeuristic)

            expanded += problem._expanded

            for action in actions:
                self.actions.append(action)
                currentState = currentState.generateSuccessor(0, action)

        totalCost = FoodSearchProblem(state).getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        print('Search nodes expanded: %d' % expanded)


    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in registerInitialState).  Return
        Directions.STOP if there is no further action to take.

        state: a GameState object (pacman.py)
        """
        "*** YOUR CODE HERE ***"

        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    return search.bfs(prob)


def closestByManhatten(gameState):
    """
    Returns a list of capsule coordinates ordered by how close they are to pacman.

    :param gameState:
    :return: list of (x,y)
    """
    food = gameState.getFood()
    pacman = gameState.getPacmanPosition()
    closest = []
    distance = 99999
    for x in xrange(0, food.width):
        for y in xrange(0, food.height):
            if food[x][y]:
                capsule = (x, y)
                currentDistance =  util.manhattanDistance(pacman, capsule)
                if currentDistance < distance:
                    closest.insert(0, capsule)
                    distance = currentDistance
    return tieResolver(closest, gameState)


def tieResolver(capsules, gameState, range = 5):
    """
    Resolves ties in a list of capsules by first ordering them by maze distance
    and then picking horizontal move over vertical.

    :param capsules:
    :param gameState:
    :param range: how many capsules from the given list should be sorted by maze distance.
    :return: the chosen one
    """
    pacman = gameState.getPacmanPosition()
    capsules = sorted(capsules[:range], key = lambda capsule: mazeDistance(pacman, capsule, gameState))
    for capsule in capsules:
        if pacman[1] == capsule[1]: # chose horizontal
            return capsule

    return capsules[0]