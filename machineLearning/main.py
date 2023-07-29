from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot

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
            right.append((prediction, data))
        else:
            wrong.append((prediction, data))
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
    

if __name__ == "__main__":
    digits = load_digits()
    dataDict = splitData(digits.data, digits.target, 0.1)
    learners = [MLPClassifier()]
    trainLearners(learners, dataDict)
    results = evaluateLearners(learners, dataDict)
    print(results[0]['accuracy'])
    
    # print(digits.data.shape)
    # plot.gray()
    # plot.matshow(digits.images[0])
    # plot.show()
    
