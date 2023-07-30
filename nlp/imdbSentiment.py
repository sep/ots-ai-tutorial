from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize
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


if __name__ == "__main__":
    testPaths = getInstanceList(testDataDir)
    trainPaths = getInstanceList(trainDataDir)
    predictions = analyzeInstances(testPaths)
    accuracy = computeAccuracy(predictions)
    print(accuracy)
