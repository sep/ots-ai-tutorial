# scikit-learn ships with a few datasets for you to play around with,
# we're using the MNIST digit dataset
from sklearn.datasets import load_digits
# We'll be using a classic neural network for the model today, but
# other models are provided by the library
from sklearn.neural_network import MLPClassifier
# We'll use this to evaluate the model
from sklearn.model_selection import train_test_split
# Used to draw some things about the ML we're doing
import matplotlib.pyplot as plot
# sklearn accepts numpy arrays.  Most ML libraries do, so it's best to
# use them from the start Pandas is also a great option.
import numpy as np


# Utility for training the learner from a dataset
def trainLearners(learners, dataDict):
    for learner in learners:
        learner.fit(dataDict['trainingData'], dataDict['trainingLabels'])
    # no need to return, learners are changed by side-effect


# Evaluates the accuracy of a previously trained learner based on the test data
def evaluateLearner(learner, dataDict):
    right = []
    wrong = []
    predictions = learner.predict(dataDict['testData'])
    for prediction, data, label in zip(predictions, dataDict['testData'], dataDict['testLabels']):
        # We're keeping track of actual misses instead of just counts
        # so that we can display some misclassified examples later.
        # It'll help us understand what kinds of mistakes ML models
        # are likely to make
        if prediction == label:
            right.append((prediction, label, data))
        else:
            wrong.append((prediction, label, data))
    # I can't tell you how many times I've made the mistake of doing
    # integer division here
    total = float(len(right) + len(wrong))
    accuracy = len(right) / total
    return {
        'right' : right,
        'wrong' : wrong,
        'accuracy' : accuracy
    }


# Just wraps evaluateLearner around a list of potential learners
def evaluateLearners(learners, dataDict):
    return list(map(lambda learner: evaluateLearner(learner, dataDict), learners))


# Wrap scikit-learns train_test_split in a function that packs them
# into a dictionary for convenience
def splitData(data, labels, testSize):
    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, labels, test_size=testSize)
    return {
        'trainingData' : dataTrain,
        'trainingLabels' : labelTrain,
        'testData' : dataTest,
        'testLabels' : labelTest
    }


# Used for displaying the misses we get from the classification models
def binResults(pairList):
    results = [ [], [], [], [], [], [], [], [], [], [], [] ]
    for label, actual, data in pairList:
        results[label].append((actual,data))
    return results


# Used for displaying the misses we get from the classification models
def reconstructImageMatrix(vector, width):
    result = []
    current = []
    count = 0
    for el in vector:
        current.append(el)
        count += 1
        if count == width:
            result.append(current)
            current = []
            count = 0
    return np.array(result)


# Used for displaying the misses we get from the classification models
def displayImageMatrix(npMat):
    plot.gray()
    plot.matshow(npMat)
    plot.show()

    
# Used for displaying the misses we get from the classification models
def displayDataVector(vec, width):
    mat = reconstructImageMatrix(vec, width)
    displayImageMatrix(mat)


# Used for displaying the misses we get from the classification models
def displayExamples(pairs, width=8):
    byLabel = binResults(pairs)
    for label, examples in enumerate(byLabel):
        if len(examples) > 0:
            actual, data = examples[0]
            print("Classified a {} as a {}".format(actual, label))
            displayDataVector(data, width)

            
if __name__ == "__main__":
    digits = load_digits()
    # What sorts of problems would happen if we radically change the
    # amount of testing data in either direction, towards 1 or 0?
    dataDict = splitData(digits.data, digits.target, 0.1)
    # Let's look up some of the other ML models that scikitlearn provides and try them.
    learners = [MLPClassifier()]
    trainLearners(learners, dataDict)
    results = evaluateLearners(learners, dataDict)
    print([result['accuracy'] for result in results])
    displayExamples(results[0]['wrong'])
