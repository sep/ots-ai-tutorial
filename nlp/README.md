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

Often we don't care about single sentences when analyzing text.  We
care about tweets and blog posts and books.  VADER is capable of doing
paragraph (and beyond) level analysis.  In this exercise, we'll look
at the default implementation of paragraph level analysis in VADER and
consider how sentence order impacts the performance of the analysis.

## Exercise 4 - What if I disagree with VADER?

VADER is a lovely & useful tool, but it's also *certainly* **Research
Grade** code. It was published as part of an academic paper, and has
had some life after that because it's so useful.

If you disagree with its default configurations, you can change them,
but the path isn't clear and differs based on what you want to change.
In any event, you're going to need to muck with the code directly in
some cases.

### Individual Words & Emoji
VADER ships with a default lexicon, available at:
`./venv/lib/python3.11/site-packages/vaderSentiment/vader_lexicon.txt`

If you disagree with the valence of individual words, you're left with two options:

* Edit them directly in the shipped file
* Supply your own sentiment lexicon when constructing the sentiment analyzer

You have the same ability to modify the emoji definitions directly.  The base definitions are found at:

`./venv/lib/python3.11/site-packages/vaderSentiment/emoji_utf8_lexicon.txt`

### Phrases, Boosters, & Negations

Sometimes phrases in a language hang together (that is, we speak using
idioms sometimes). A classic example is when we tell someone to "break
a leg".  On it's surface, that's a negative sentiment sentence, but
culturaly we know we're wishing the recipient good luck.

Idiomatic expressions and their interpretations are generally handled
via special case.  In VADER, the list of idioms is hard coded and
cannot be passed in via file or reference.  To edit these, you'd need
to edit the definitons, found in
`./venv/lib/python3.11/site-packages/vaderSentiment/vaderSentiment.py`
on line 71 in `SENTIMENT_LADEN_IDIOMS` and on line 78 in
`SPECIAL_CASES`.

Boosters are words that increase or enhance the sentiment of a following word, like "really". Negations are words that invert the valence of the following word, like "not".  Again, these words need defined for sentiment analysis to work properly, and again they're defined early on inside the source code of VADER.
Boosters are on line 46 in `BOOSTER_DICT` and negations on line 33 in `NEGATE`.

### What was the exercise again?

* Navigate to the hard coded definitions
  * Do any of these suprise you?
  * Are there some you think need added? Do so!
* Navigate to the data files and see if you can intuit their format  


# `imdbSentiment.py`

## Exercise 1 - Run Sentiment Analysis Using Just VADER

## Exercise 2 - Try Different Segmenting & Score Aggregation Tools

## Exercise 3 - Use VADER to Vectorize Text for Classic ML