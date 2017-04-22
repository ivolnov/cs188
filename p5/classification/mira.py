# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
import itertools
import math

PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    "*** YOUR CODE HERE ***"

    maxScore = 0
    winner = None

    for cap in Cgrid:
      weights = {label: util.Counter() for label in self.legalLabels}

      for iteration in xrange(self.max_iterations):
        for datum, label in itertools.izip(trainingData, trainingLabels):
          products = util.Counter()
          for l in self.legalLabels:
            products[l] = weights[l] * datum
          prediction = products.argMax()
          if prediction != label:
            tau = self._evalTau(datum, weights[label], weights[prediction], cap)
            bias = self._scaleVector(tau, datum)
            weights[prediction] -= bias
            weights[label] += bias

      self.weights = weights
      score = 0

      for i, prediction in enumerate(self.classify(validationData)):
        if prediction == validationLabels[i]:
          score += 1

      if score > maxScore:
        maxScore = score
        winner = self.weights

    self.weights = winner

  def _evalTau(self, features, correctLabelWeights, incorrectLabelWeights, cap):
    """
    Calculates tau coefficient for MIRA algorithm.

    :param features: a feature vector for particular datum
    :param correctLabelWeights: current weights for a correct label
    :param incorrectLabelWeights: current weights for a label that had been predicted
    :param cap: cap constant to bound tau from above
    :return:
    """
    numerator = (incorrectLabelWeights - correctLabelWeights) * features + 1.0
    denominator = 2 * (features * features)
    tau = numerator / denominator
    return min(tau, cap)

  def _scaleVector(self, scalar, vector):
    """
    Returns a copy of a given vector with each attribute multiplied by a given factor.

    :param scalar: a factor
    :param vector: a vector to copy
    :return:
    """
    copy = vector.copy()
    for key, value in copy.items():
      copy[key] = value * scalar
    return copy

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"
    features = [(feature,  self.weights[label1][feature] - self.weights[label2][feature]) for feature in self.features]
    features.sort(key=lambda tup: tup[1], reverse=True)
    return map(lambda tup: tup[0], features[:100])

