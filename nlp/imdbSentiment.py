# Splits large bodies of text into tokens
from nltk import tokenize
# Our classic sentiment analysis approach for exercise 2a
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Machine learning approach for exercise 2b
from sklearn.neural_network import MLPClassifier
# Path Manipulation and file reading tools
import os
from pathlib import Path

POSITIVE = 1
NEGATIVE = -1
DEFAULT_VADER = SentimentIntensityAnalyzer()
baseReviewDir = os.path.join(".", "data", "aclImdb")
testDataDir = os.path.join(baseReviewDir, "test")
trainDataDir = os.path.join(baseReviewDir, "train")


# Here are some tools for us to be able to read in a big corpus of text
# We're just walking directories and storing paths in a dictionary for later use
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


# This takes a path, reads in the text, and provides a valence score on every word in the text
def getTextScores(path, analyzer=DEFAULT_VADER):
    text = Path(path).read_text()
    # NLTK has tons of useful tools for natural language processing.
    # Splitting text up into sentences is the tip of the iceberg
    sentences = tokenize.sent_tokenize(text)
    sentiments = []
    for sentence in sentences:
        # We're discarding a lot of information here.
        # First, we're reducing the vader sentiment scores to just their compound polarity
        # Second, we're destroying the sentence structure of the document
        # We could have recorded the sentiments as a list of lists
        # How would that change the program?
        # Do you think it would help get better overall sentiment readings?
        # Why or why not?
        valence = analyzer.polarity_scores(sentence)
        sentiments.append(valence["compound"])
    return sentiments


# This is a wrapper around getTextScores that lets us score a whole dictionary of paths
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



def isPositiveSum(compounds):
    total = sum(compounds)
    if total > 0:
        return POSITIVE
    else:
        return NEGATIVE

# TODO: Fill this in
def isPositiveCount(compounds):
    raise NotImplementedError


# TODO: Fill this in
def isPositiveExtrema(compounds):
    raise NotImplementedError

# TODO: Come up with one or two of your own!

# This predicts the overall sentiment of the instances
# Note how we pass in a prediction function to be used later
# Getting the sentiment of the text is the start of telling
# whether or not the whole thing is positive, not the end.
def analyzeInstances(instanceDict, predict=isPositiveSum):
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


# Utility for scoring our predictions once they're finished
def computeListAccuracy(resultsList, trueSentiment):
    total = float(len(resultsList))
    correct = resultsList.count(trueSentiment)
    return correct / total


# Utility for scoring our predictions once they're finished
# We'll revisit these ideas in the segment on Machine Learning
def computeAccuracy(resultsDict):
    return {
        "positive" : computeListAccuracy(resultsDict["positive"], POSITIVE),
        "negative" : computeListAccuracy(resultsDict["negative"], NEGATIVE)
    }


# Using VADER sentiment analysis and simple sentiment scores
def bareVADER():
    testPaths = getInstanceList(testDataDir)
    trainPaths = getInstanceList(trainDataDir)
    predictions = analyzeInstances(testPaths)
    accuracy = computeAccuracy(predictions)
    print(accuracy)


# Utility for using VADER sentiment scores as the basis for classic ML approaches
def pad(lst, targetLength):
    lst.extend([0] * (targetLength - len(lst)))


# We'll walk through this together as exercise 2B
# We could do something more advanced on top of the VADER scoring.
# Here's one potential approach.  We use the input sentiment vectors
# as input to machine learning, and learn a function from those
# vectors to the appropriate label
def sklearnUsingVADER():
    # load up the data
    testPaths = getInstanceList(testDataDir)
    trainPaths = getInstanceList(trainDataDir)
    testScores = scoreInstances(testPaths)
    trainScores = scoreInstances(trainPaths)
    # set up our ML classifier
    model = MLPClassifier()
    # The technique we chose requires that all input vectors be of the
    # same size, so lets figure out what the longest vector is and pad
    # them.
    # There are, of course, other ways of normalizing the length of
    # the input vector.  We could take the approaches we did earlier
    # for the basic VADER approach (e.g. summing, averages, etc)
    # How do you think our choice of aligning the vectors impacts the ML model?
    # What information is being removed or diluted?
    maxLenVector = max(map(len, testScores["positive"]))
    maxLenVector = max(maxLenVector, max(map(len, testScores["negative"])))
    maxLenVector = max(maxLenVector, max(map(len, trainScores["positive"])))
    maxLenVector = max(maxLenVector, max(map(len, trainScores["negative"])))
    # Building out a matching label set for each training instance.
    # scikit-learn (and most ML tools) expect to receive two equal
    # length inputs, one of labels and one of sets of features.  We'll
    # come back to this idea in the ML section
    trainLabels = [POSITIVE] * len(trainScores["positive"])
    trainLabels += [NEGATIVE] * len(trainScores["negative"])
    testLabels = [POSITIVE] * len(testScores["positive"])
    testLabels += [NEGATIVE] * len(testScores["negative"])
    trainData = trainScores["positive"] + trainScores["negative"]
    testData = testScores["positive"] + testScores["negative"]
    # Do the padding we mentioned earlier
    for lst in trainData:
        pad(lst, maxLenVector)
    for lst in testData:
        pad(lst, maxLenVector)
    # Train the model
    model.fit(trainData, trainLabels)
    # Predict the held out examples
    predictions = model.predict(testData)
    results = {
        "positive" : 0,
        "negative" : 0
    }
    # Figure out how well we did
    for prediction, truth in zip(predictions, testLabels):
        if truth == POSITIVE:
            if prediction == truth:
                results["positive"] += 1
        else:
            if prediction == truth:
                results["negative"] += 1
    results["positive"] /= float(len(testScores["positive"]))
    results["negative"] /= float(len(testScores["negative"]))
    # Display that information to the user
    print(results)


if __name__ == "__main__":
    #bareVADER()         # Exercise 2A
    #sklearnUsingVADER() # Exercise 2B
    print("Hello World")
