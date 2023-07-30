from nltk import tokenize
from sklearn.neural_network import MLPClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from pathlib import Path

POSITIVE = 1
NEGATIVE = -1
analyzer = SentimentIntensityAnalyzer()
baseReviewDir = os.path.join(".", "data", "aclImdb")
testDataDir = os.path.join(baseReviewDir, "test")
trainDataDir = os.path.join(baseReviewDir, "train")


def getChildPaths(rootDir):
    return map(lambda fname : os.path.join(rootDir, fname), os.listdir(rootDir))


def getInstanceList(instanceRoot):
    posDir = os.path.join(instanceRoot, "pos")
    negDir = os.path.join(instanceRoot, "neg")
    posPaths = getChildPaths(posDir)
    negPaths = getChildPaths(negDir)
    return {
        "positive" : posPaths,
        "negative" : negPaths
    }


def isPositiveSum(compounds):
    total = sum(compounds)
    if total > 0:
        return POSITIVE
    else:
        return NEGATIVE


def isPositiveCount(compounds):
    posCount = 0
    negCount = 0
    for score in compounds:
        if score > 0:
            posCount += 1
        elif score < 0:
            negCount += 1
    if posCount > negCount:
        return POSITIVE
    else:
        return NEGATIVE


def isPositiveExtrema(compounds):
    mostNegative = POSITIVE
    mostPositive = NEGATIVE
    for score in compounds:
        mostNegative = min(score, mostNegative)
        mostPositive = max(score, mostPositive)
    if mostPositive > abs(mostNegative):
        return POSITIVE
    else:
        return NEGATIVE


def getTextScores(path):
    text = Path(path).read_text()
    sentences = tokenize.sent_tokenize(text)
    sentiments = []
    for sentence in sentences:
        valence = analyzer.polarity_scores(sentence)
        sentiments.append(valence["compound"])
    return sentiments


def scoreInstances(instanceDict):
    results = {
        "positive" : [],
        "negative" : []
    }
    for path in instanceDict["positive"]:
        results["positive"].append(getTextScores(path))
    for path in instanceDict["negative"]:
        results["negative"].append(getTextScores(path))
    return results


def analyzeInstances(instanceDict, predict=isPositiveExtrema):
    results = {
        "positive" : [],
        "negative" : []
    }
    scores = scoreInstances(instanceDict)
    for score in scores["positive"]:
        results["positive"].append(predict(score))
    for score in scores["negative"]:
        results["negative"].append(predict(score))
    return results


def computeListAccuracy(resultsList, trueSentiment):
    total = float(len(resultsList))
    correct = resultsList.count(trueSentiment)
    return correct / total


def computeAccuracy(resultsDict):
    return {
        "positive" : computeListAccuracy(resultsDict["positive"], POSITIVE),
        "negative" : computeListAccuracy(resultsDict["negative"], NEGATIVE)
    }


def bareVADER():
    testPaths = getInstanceList(testDataDir)
    trainPaths = getInstanceList(trainDataDir)
    predictions = analyzeInstances(testPaths)
    accuracy = computeAccuracy(predictions)
    print(accuracy)


def pad(lst, targetLength):
    lst.extend([0] * (targetLength - len(lst)))


def sklearnUsingVADER():
    testPaths = getInstanceList(testDataDir)
    trainPaths = getInstanceList(trainDataDir)
    testScores = scoreInstances(testPaths)
    trainScores = scoreInstances(trainPaths)
    model = MLPClassifier()
    maxLenVector = max(map(len, testScores["positive"]))
    maxLenVector = max(maxLenVector, max(map(len, testScores["negative"])))
    maxLenVector = max(maxLenVector, max(map(len, trainScores["positive"])))
    maxLenVector = max(maxLenVector, max(map(len, trainScores["negative"])))
    trainLabels = [POSITIVE] * len(trainScores["positive"])
    trainLabels += [NEGATIVE] * len(trainScores["negative"])
    testLabels = [POSITIVE] * len(testScores["positive"])
    testLabels += [NEGATIVE] * len(testScores["negative"])
    trainData = trainScores["positive"] + trainScores["negative"]
    testData = testScores["positive"] + testScores["negative"]
    for lst in trainData:
        pad(lst, maxLenVector)
    for lst in testData:
        pad(lst, maxLenVector)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    results = {
        "positive" : 0,
        "negative" : 0
    }
    for prediction, truth in zip(predictions, testLabels):
        if truth == POSITIVE:
            if prediction == truth:
                results["positive"] += 1
        else:
            if prediction == truth:
                results["negative"] += 1
    results["positive"] /= float(len(testScores["positive"]))
    results["negative"] /= float(len(testScores["negative"]))
    print(results)

if __name__ == "__main__":
    #bareVADER()
    sklearnUsingVADER()
