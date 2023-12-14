# ots-ai-tutorial
Off-the-shelf AI Tutorial (Originally Prepared for Indy.Code() 2023)

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
  * `gymnasium`
  * `pathfinding`
  
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
