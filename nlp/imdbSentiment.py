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


def analyzeReview(path, getReviewSentiment=isPositiveSum):
    text = Path(path).read_text()
    sentences = tokenize.sent_tokenize(text)
    sentiments = []
    for sentence in sentences:
        valence = analyzer.polarity_scores(sentence)
        sentiments.append(valence["compound"])
    sentiment = getReviewSentiment(sentiments)
    #print(path,'\n', text, '\n', sentiment)
    return sentiment


def analyzeInstances(instanceDict):
    results = {
        "positive" : [],
        "negative" : []
    }
    for path in instanceDict["positive"]:
        results["positive"].append(analyzeReview(path))
    for path in instanceDict["negative"]:
        results["negative"].append(analyzeReview(path))
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
