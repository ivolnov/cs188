# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from util import mazeDistance
from game import Directions, Actions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

  def evaluationFunction(self, currentGameState, action):
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

    "*** YOUR CODE HERE ***"
    dot = closestFood(newPos, currentGameState.getFood().asList())
    ghost = closestAngryGhost(newPos, newGhostStates)
    #capsule = closestCapsule(newPos, currentGameState.getCapsules())

    value = 0 if action == Directions.STOP else 10
    value += 0 if not ghost or ghost > 3 else -10 / ghost
    value += reciprocalOf(dot, successorGameState)

    return value

def scoreEvaluationFunction(currentGameState):
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

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    def minimax(gameState, agentId, func):
        result = -func(9999, -9999)
        index = agentId % gameState.getNumAgents()
        actions = gameState.getLegalActions(index)

        if len(actions) == 0: return gameState.getScore()

        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            cost = dispatch(successor, agentId + 1)
            result = func(result, cost)

        return result

    def dispatch(gameState, agentId):
        if agentId  == gameState.getNumAgents() * self.depth:
            return self.evaluationFunction(gameState)
        else:
            if agentId % gameState.getNumAgents() == 0:
                return minimax(gameState, agentId, max)
            else:
                return minimax(gameState, agentId, min)

    maximum = -99999
    choice = Directions.STOP

    for action in gameState.getLegalPacmanActions():
        successor = gameState.generatePacmanSuccessor(action)
        cost = dispatch(successor, 1)

        if cost > maximum:
            choice = action
            maximum = cost

    return choice

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    CEILING = 99999
    FLOOR = -99999

    def eligibleToPrune(cost, fn, alfa, beta):
        if fn.__name__ == 'max':
            return cost >= beta
        else:
            return cost <= alfa

    def dispatch(gameState, agentid, alfa, beta):
        if agentid == gameState.getNumAgents() * self.depth:
            return self.evaluationFunction(gameState)
        if agentid % gameState.getNumAgents() == 0:
            return alfaBeta(gameState, agentid, max, alfa, beta)
        else:
            return alfaBeta(gameState, agentid, min, alfa, beta)

    def alfaBeta(gameState, agentId, fn, alfa, beta):
        result = -fn(FLOOR, CEILING)
        index = agentId % gameState.getNumAgents()

        actions = gameState.getLegalActions(index)

        if len(actions) == 0:
            return self.evaluationFunction(gameState)

        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            cost = dispatch(successor, agentId + 1, alfa, beta)

            if eligibleToPrune(cost, fn, alfa, beta):
                return cost

            result = fn(result, cost)

            if fn.__name__ == 'max':
                alfa = max(alfa, cost)
            else:
                beta = min(beta, cost)

        return result

    choice = None
    maxCost = FLOOR

    for action in gameState.getLegalPacmanActions():
      successor = gameState.generatePacmanSuccessor(action)
      cost = dispatch(successor, 1, FLOOR, CEILING)

      if cost > maxCost:
        maxCost = cost
        choice = action

    return choice


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    FLOOR = -99999

    choise = None
    maxCost = FLOOR

    def dispatch(gameState, agentId):
        if agentId == gameState.getNumAgents() * self.depth:
            return self.evaluationFunction(gameState)
        if agentId % gameState.getNumAgents() == 0:
            return maxNode(gameState, agentId)
        else:
            return expectiNode(gameState, agentId)

    def maxNode(gameState, agentId):
        result = FLOOR
        index = agentId % gameState.getNumAgents()

        actions = gameState.getLegalActions(index)

        if len(actions) == 0:
            return self.evaluationFunction(gameState)

        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            cost = dispatch(successor, agentId + 1)
            result = max(result, cost)

        return result

    def expectiNode(gameState, agentId):
        index = agentId % gameState.getNumAgents()

        actions = gameState.getLegalActions(index)

        if len(actions) == 0:
            return self.evaluationFunction(gameState)

        expectation = 1.0 / len(actions) # uniformly
        result = 0

        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            result += dispatch(successor, agentId + 1) * expectation

        return result

    for action in gameState.getLegalPacmanActions():
      successor = gameState.generatePacmanSuccessor(action)
      cost = dispatch(successor, 1)

      if cost > maxCost:
        maxCost = cost
        choise = action

    return choise


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Just a linear combination, a weighted sum of relevant data from given game state.
                 Score is a crucial part as it the metric of death.
                 Closest maze distance of food pills with shortest manhattan distance to solve ties between them.
                 Number of scared ghosts is important as it stimulates pacman to eat capsules.
                 Number of food left to eat helps to motivate pacman to dive into dead ends otherwise maze distance of
                 leaving it makes it non optimal.

  """
  "*** YOUR CODE HERE ***"
  ghostStates = currentGameState.getGhostStates()
  score = currentGameState.getScore()

  scaredGhostsNum = sum([1 if ghost.scaredTimer > 0 else 0 for ghost in ghostStates])
  closestMazeDot = closestFoodMazeDistance(currentGameState)
  foodLeft = currentGameState.getNumFood()
  walls = wallsCount(currentGameState)

  value = .0 + score - closestMazeDot + 100 * scaredGhostsNum - 10 * foodLeft - 0.1 * walls

  return value


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def __init__(self):
      self._agent = AlphaBetaAgent(evalFn = 'betterEvaluationFunction', depth = '2')

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    return self._agent.getAction(gameState)


def closestAngryGhost(pacman, ghosts):
    closest = None

    for ghost in ghosts:
        if ghost.scaredTimer > 0: continue
        distance = manhattanDistance(pacman, ghost.getPosition())
        closest = closest if closest and closest < distance else distance

    return closest


def closestFood(pacman, food, direction):
    closest = 999999

    horizontal = {
        Directions.STOP: xrange(0, 0),
        Directions.NORTH: xrange(0, food.width),
        Directions.EAST: xrange(pacman[0], food.width),
        Directions.SOUTH: xrange(0, food.width),
        Directions.WEST: xrange(0, pacman[0]),
    }[direction]

    vertical = {
        Directions.STOP: xrange(0, 0),
        Directions.NORTH: xrange(pacman[1], food.height),
        Directions.EAST: xrange(0, food.height),
        Directions.SOUTH: xrange(0, pacman[1]),
        Directions.WEST: xrange(0, food.height),
    }[direction]

    for x in horizontal:
        for y in vertical:
            if not food[x][y]:
                continue
            distance = manhattanDistance(pacman, (x, y))
            closest = min(distance, closest)

    return closest


def closestCapsule(pacman, capsules):
    closest = None

    for capsule in capsules:
        distance = manhattanDistance(pacman, capsule)
        closest = closest if closest and closest < distance else distance

    return closest


def reciprocalOf(distance, gameState):
    grid = gameState.getFood()
    base = grid.width + grid.height
    return base - distance if distance else base


def foodDensity(pacman, food, radius=5):
    density = 0

    for x in xrange(pacman[0] - radius, pacman[0] + radius):
        for y in xrange(pacman[1] - radius, pacman[1] + radius):
            if food[x][y]: density += 1

    return density


def directionPenalty(pacman, ghosts, direction):
    for ghost in ghosts:
        if ghost[1] == pacman[1] and abs(pacman[0] - ghost[0]) < 3: # they are on one horizontal line
            if ghost[0] < pacman[0] and direction == Directions.WEST: # ghost is on the left
                return 1
            if ghost[0] > pacman[0] and direction == Directions.EAST: # ghost is on the right
                return 1
        if ghost[0] == pacman[0] and abs(pacman[1] - ghost[1]) < 3: # they are on one vertical line
            if ghost[1] < pacman[1] and direction == Directions.SOUTH: # ghost is below pacman
                return 1
            if ghost[1] > pacman[1] and direction == Directions.NORTH: # ghost is above pacman
                return 1
    return 0


def closestFoodMazeDistance(gameState, searchLimit=5):
    foodList = gameState.getFood().asList()
    if len(foodList) == 0:
        return 0

    pacman = gameState.getPacmanPosition()
    shortest = 9999999
    clusters = {}
    distances = []

    for dot in foodList:
        distance = manhattanDistance(pacman, dot)
        if distance in clusters:
            clusters[distance].append(dot)
        else:
            clusters[distance] = [dot]

        shortest = min(shortest, distance)

    for dot in clusters[shortest][:searchLimit]:
        distances.append(mazeDistance(dot, gameState))

    distances.sort()

    return distances[0]

def wallsCount(gameState):
    pacman = gameState.getPacmanPosition()
    walls = gameState.getWalls()
    count = 0
    for x in xrange(pacman[0] - 1, pacman[0] + 1):
        for y in xrange(pacman[1] - 1, pacman[1] + 1):
            if walls[x][y]:
                count += 1
    return count


