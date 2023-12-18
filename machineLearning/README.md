# sklearnTutorial.py

## Exercise 0

Make sure you can run the tutorial code.  Run the `sklearnTutorial.py` script

## Exercise 1

Explore some of the learners baked into SciKit Learn:

* Multilayer Neural Network (the demo algorithm)
* [K Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
* [Decision Tree](https://scikit-learn.org/stable/modules/tree.html#classification)
* [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html#classification)

### Things to do

* Get them turning over
* Compare them one to another
  * Before you do a bakeoff, which do you think will work best?
  * Are any of them weaker for specific digits?
  * Do you notice a difference in training times?
* Browse the documentation describing the techniques

# Exercise 2

Swap out the underlying datasets for the learner experiment.  We
started with `digits`, but scikit learn ships with [several different
ones](https://scikit-learn.org/stable/datasets/toy_dataset.html):

* `load_iris`
* `load_diabetes`
* `load_wine`
* `load_breast_cancer`

You'll need to comment out `displayExamples` for this exercise,
because it's only set up to display digits.

### Things to do

* Explore the shape of the dataset
  * How many examples
  * How many unique labels?
  * How big are the feature sets
    * Number of features
    * Variation within a single feature
* How do you think the above will influence the performance of learners?

# pyTorchTutorial.py

## Exercise 0

* Open up the source code and browse through it
* Uncomment lines 112-114
* Run the script and take a look
* Comment out lines 112-114
* Run the script

## Exercise 1

Examine the chain definition more carefully.  See if you can change
the size of the intermediate layers in the neural network.

# pyTorchTutorialP2.py

Take a look at the code in this file.  It exactly replicates what's in
the previous exercise, without doing the hand configuration of the
neural network.

See if you can map the parts of the model definition line for line back to `pyTorchTutorial.py`