# NLP Tutorial

# `VADERdemo.py`

This is an annotated version of the demo that ships with the VADER
sentiment analysis tool.  It's broken up into steps and littered with
comments explaining what's going on as we work through the example.

## Exercise 0 - Kick the tires

Get everything installed and the virtual environment configured and
loaded up. `setup.sh` should help with that, either by being directly
runnable or by showing you which libraries are required and included.

Once you have the virtual enviroment created and the packages
installed, try running vaderDemo directy.  This will depend on how
you're managing your environments, but may look something like
`./venv/bin/python3 VADERdemo.py`

## Exercise 1 - Word Sentiment Analysis

VADER (and many tools like it) works by associating a score or valence
to individual words and phrases.  In exercise 1, we'll be looking at:

* (a) words that generally denote good things
* (b) words that generally denote bad things
* (c) the messy middle.

## Exercise 2 - Sentence Level Analysis

Exercise 1c showed us that some words have ambiguous interpretations.
When words are a little iffy on the good/bad scale, we need to lean
into larger contexts.  That means sentence level analysis.  In this exercise, we'll look at:

* (a) sentences that are obviously good or bad
* (b) sentences that are a little ambiguous
* (c) sentences that you make with the intent of confusing VADER

## Exercise 3 - Paragraph Level Analysis

# `imdbSentiment.py`

## Exercise 1 - Run Sentiment Analysis Using Just VADER

## Exercise 2 - Try Different Segmenting & Score Aggregation Tools

## Exercise 3 - Use VADER to Vectorize Text for Classic ML