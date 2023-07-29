from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
import numpy as np

def trainLearners(learners, dataDict):
    for learner in learners:
        learner.fit(dataDict['trainingData'], dataDict['trainingLabels'])
    # no need to return, learners are changed by side-effect


def evaluateLearner(learner, dataDict):
    right = []
    wrong = []
    predictions = learner.predict(dataDict['testData'])
    for prediction, data, label in zip(predictions, dataDict['testData'], dataDict['testLabels']):
        if prediction == label:
            right.append((prediction, label, data))
        else:
            wrong.append((prediction, label, data))
    total = float(len(right) + len(wrong))
    accuracy = len(right) / total
    return {
        'right' : right,
        'wrong' : wrong,
        'accuracy' : accuracy
    }


def evaluateLearners(learners, dataDict):
    return list(map(lambda learner: evaluateLearner(learner, dataDict), learners))


def splitData(data, labels, testSize):
    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, labels, test_size=testSize)
    return {
        'trainingData' : dataTrain,
        'trainingLabels' : labelTrain,
        'testData' : dataTest,
        'testLabels' : labelTest
    }


def binResults(pairList):
    results = [ [], [], [], [], [], [], [], [], [], [], [] ]
    for label, actual, data in pairList:
        results[label].append((actual,data))
    return results


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


def displayImageMatrix(npMat):
    plot.gray()
    plot.matshow(npMat)
    plot.show()

    
def displayDataVector(vec, width):
    mat = reconstructImageMatrix(vec, width)
    displayImageMatrix(mat)


def displayExamples(pairs, width=8):
    byLabel = binResults(pairs)
    for label, examples in enumerate(byLabel):
        if len(examples) > 0:
            actual, data = examples[0]
            print("Classified a {} as a {}".format(actual, label))
            displayDataVector(data, width)

            
if __name__ == "__main__":
    digits = load_digits()
    dataDict = splitData(digits.data, digits.target, 0.1)
    learners = [MLPClassifier()]
    trainLearners(learners, dataDict)
    results = evaluateLearners(learners, dataDict)
    print(results[0]['accuracy'])
    displayExamples(results[0]['wrong'])
