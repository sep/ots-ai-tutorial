# Off-the-shelf AI Tutorial

# Notes And Goals

This tutorial is meant to be a whirlwind introduction to the broad
landscape of artificial intelligence.  Although we often talk about AI
as a monolith, it turns out it's a large collection of techniques
built for solving specific classes of problems.

In the tutorial, we'll explore those techniques by looking at
different off the shelf tools for their specific areas.  The goal is
to be an inch deep and a mile wide; we want to install and use the
tools on a small set of problems to:

* get a feel for that process
* see the breadth of the AI landscape
* start to build a mapping from real world problems to AI techniques

The tools we're using range from "published library which is really
just part of someone's dissertation" to "well maintained open source
library". It's intentional, because it's representative of reality.

# Requirements

* An environment management tool for python
  * the demo uses `venv`
* `pip` for managing python libraries
* `python`, the following libraries, and their supports
  * `matplotlib`
  * `sklearn`
  * `pytorch`
  * `torchvision`
  * `vaderSentiment`
  * `nltk`
    * You'll want to run `nltk.download all` before the day of the tutorial
  * `gymnasium`
    * You'll want to pip install gymnasium[classic_control] as well
  * `pathfinding`

* The following datasets
  * NIST digit data (included with sklearn)
  * Better Resolution NIST digit data (downloadable with torchvision)
    * `datasets.MNIST('data', download=True, transform=transform, train=True)`
    * `dataset.MNIST('data', download=True, transform=transform, train=False)`
  * [ICMDB Sentiment dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
  * Pathfinding [maps](https://movingai.com/benchmarks/bgmaps/bgmaps-map.zip) and [scenarios](https://movingai.com/benchmarks/bgmaps/bgmaps-scen.zip)
  
  
# Setup

Currently, there are setup scripts available for any system that can
run `apt`, which is to say many linux distros.  These were originally
built to configure virtual machine images or remote boxes (e.g. on
digital ocean) for use in the tutorial.  They could also be useful to
you in getting your own machine configured for the tutorial.

## `./configureMachine.sh`

Downloads general development tools and installs various `apt-get`able
python libraries for use in the tutorials.

## `./nlp/setup.sh`

Sets up a virtual environment for the natural language processing
tutorial and `pip` installs several python libraries used in the NLP
tutorial, specifically:

* `vaderSentiment` - VADER sentiment analysis tool
* `nltk` - The natural language tool kit

Further, it downloads the IMDB sentiment dataset from standard and
unpacks it into `./nlp/data/`

## `./planning/setup.sh`

Sets up a virtual environment for the planning tutorial and `pip`
installs the python libraries used in the tutorial, specifically:

* `gymnasium` - formerly OpenAI.gymnasium, a reinforcement learning framework
* `gymnasium[classic_control]` - the classic control example domains
* `tqdm` - a terminal-based progress bar gymnasium relies on
* `pathfinding` - a 2D pathfinding library

It also downloads a 2D pathfinding dataset (maps from the Baldur's Gate
series) and unpacks them to `./planning/maps` and
`./planning/scenarios`


# History (Given At)

* Indy.Code() 2023
* CodeMash 2024

# Credits & Acknowledgments

## Developers

* Jordan Thayer

## Editing, Feedback

* Robert Herbig
* Will Trimble

## Beta Testers

* Lee Harold

## Datasets

* [Nathan Sturtevant for providing pathfinding benchmark sets](https://movingai.com/benchmarks/grids.html)
* [Stanford for providing the IMDB sentiment data sets](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
