# Reproduces and extends the demo code packaged with VaderSentmiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize
import random

# Spins up your sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def exercise1a():
    # - VADER works best when analysis is done at the sentence level
    # - (but it can work on single words or entire novels).
    # - One reason you might analyze single words is because they're annotation to an image or video.
    # - Here are some positive examples
    conceptList = ["balloons", "cake", "candles", "happy birthday", "friends", "laughing", "smiling", "party"]
    conceptSentiments = 0.0
    for concept in conceptList:
        vs = analyzer.polarity_scores(concept)
        print("{:-<15} {}".format(concept, str(vs['compound'])))
        conceptSentiments += vs["compound"]
    print("AVERAGE SENTIMENT OF TAGS/LABELS: \t" + str(round(conceptSentiments / len(conceptList), 4)))
    print("\t")

def exercise1b():
    # - These are some negative examples
    conceptList = ["riot", "fire", "fight", "blood", "mob", "war", "police", "tear gas"]
    conceptSentiments = 0.0
    for concept in conceptList:
        vs = analyzer.polarity_scores(concept)
        print("{:-<15} {}".format(concept, str(vs['compound'])))
        conceptSentiments += vs["compound"]
    print("AVERAGE SENTIMENT OF TAGS/LABELS: \t" + str(round(conceptSentiments / len(conceptList), 4)))


def exercise1c():
    # - These examples are somewhat complicated
    conceptList = [
        "overcome", "hard-won", "baroque", "antique", "solvent", "negative", "positive",
        "cancer", "libra"
    ]
    conceptSentiments = 0.0
    for concept in conceptList:
        vs = analyzer.polarity_scores(concept)
        print("{:-<15} {}".format(concept, str(vs['compound'])))
        conceptSentiments += vs["compound"]
    print("AVERAGE SENTIMENT OF TAGS/LABELS: \t" + str(round(conceptSentiments / len(conceptList), 4)))


def exercise2a():
    # Exercise 2 - Sentence Level Analysis
    sentences = [
        "VADER is smart, handsome, and funny.",  # positive sentence example
        "VADER is smart, handsome, and funny!",  # punctuation emphasis adjusts intensity
        "VADER is very smart, handsome, and funny.", # booster words handled correctly (sentiment intensity adjusted)
        "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
        "VADER is VERY SMART, handsome, and FUNNY!!!", # combination of signals
        "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!", 
        "VADER is not smart, handsome, nor funny.",  # negation sentence example
        "The book was good.",  # positive sentence
        "At least it isn't a horrible book.",  # negated negative sentence with contraction
        "The book was only kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
        "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation
        "Today SUX!",  # negative slang with capitalization emphasis
        "Today only kinda sux! But I'll get by, lol", # mixed sentiment with slang and constrastive conjunction "but"
        "Make sure you :) or :D today!",  # emoticons handled
        "Catch utf-8 emoji such as 💘 and 💋 and 😁",  # emojis handled
        "Not bad at all"  # Capitalized negation
    ]

    # Analyze typical example cases, including handling of:
    # -- negations
    # -- punctuation emphasis & punctuation flooding
    # -- word-shape as emphasis (capitalization difference)
    # -- degree modifiers (intensifiers such as 'very' and dampeners such as 'kind of')
    # -- slang words as modifiers such as 'uber' or 'friggin' or 'kinda'
    # -- contrastive conjunction 'but' indicating a shift in sentiment; sentiment of later text is dominant
    # -- use of contractions as negations
    # -- sentiment laden emoticons such as :) and :D
    # -- utf-8 encoded emojis such as 💘 and 💋 and 😁
    # -- sentiment laden slang words (e.g., 'sux')
    # -- sentiment laden initialisms and acronyms (for example: 'lol')
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))

    #- About the scoring: 
    #  -- The 'compound' score is computed by summing the valence
    #     scores of each word in the lexicon, adjusted according to
    #     the rules, and then normalized to be between -1 (most
    #     extreme negative) and +1 (most extreme positive).  This is
    #     the most useful metric if you want a single unidimensional
    #     measure of sentiment for a given sentence.  Calling it a
    #     'normalized, weighted composite score' is accurate.""

    #  -- The 'pos', 'neu', and 'neg' scores are ratios for
    # proportions of text that fall in each category (so these should
    # all add up to be 1... or close to it with float operation).
    # These are the most useful metrics if you want multidimensional
    # measures of sentiment for a given sentence.

def exercise2b():
    tricky_sentences = [
        "Sentiment analysis has never been good.",
        "Sentiment analysis has never been this good!",
        "Most automated sentiment analysis tools are shit.",
        "With VADER, sentiment analysis is the shit!",
        "Other sentiment analysis tools can be quite bad.",
        "On the other hand, VADER is quite bad ass",
        "VADER is such a badass!",  # slang with punctuation emphasis
        "Without a doubt, excellent idea.",
        "Roger Dodger is one of the most compelling variations on this theme.",
        "Roger Dodger is at least compelling as a variation on the theme.",
        "Roger Dodger is one of the least compelling variations on this theme.",
        "Not such a badass after all.",  # Capitalized negation with slang
        "Without a doubt, an excellent idea."  # "without {any} doubt" as negation
    ]
    
    # - Analyze examples of tricky sentences that cause trouble to other sentiment analysis tools.
    #  -- special case idioms - e.g., 'never good' vs 'never this good', or 'bad' vs 'bad ass'.
    #  -- special uses of 'least' as negation versus comparison
    for sentence in tricky_sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs)))

def exercise2c():
    # Now you try.  The goal here is to be confusing to automated
    # analysis while still being very obvious to human consumers.
    # Some strategies you might consider:
    # * Sarcasm
    # * Literary or Historical References
    # * Scare Quotes
    your_sentences = [
    ]
    
    # - Analyze examples of tricky sentences that cause trouble to other sentiment analysis tools.
    #  -- special case idioms - e.g., 'never good' vs 'never this good', or 'bad' vs 'bad ass'.
    #  -- special uses of 'least' as negation versus comparison
    for sentence in your_sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs)))


def exercise3a():
    # - VADER works best when analysis is done at the sentence level
    # - (but it can work on single words or entire novels).
    paragraph = "It was one of the worst movies I've seen, despite good reviews. \
        Unbelievably bad acting!! Poor direction. VERY poor production. \
        The movie was bad. Very bad movie. VERY BAD movie!"
    # -- For example, given the above paragraph text from a hypothetical movie review
    # -- You could use NLTK to break the paragraph into sentence
    # -- tokens for VADER, then average the results for the paragraph
    # -- like this:

    sentence_list = tokenize.sent_tokenize(paragraph)
    paragraphSentiments = 0.0
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs["compound"])))
        paragraphSentiments += vs["compound"]
    print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(paragraphSentiments / len(sentence_list), 4)))


def exercise3b():
    # You may have noticed that the aggregation in VADER isn't order-aware.
    # Do you think that's a problem?
    paragraph = "It was one of the worst movies I've seen, despite good reviews. \
        Unbelievably bad acting!! Poor direction. VERY poor production. \
        The movie was bad. Very bad movie. VERY BAD movie!"
    sentence_list = tokenize.sent_tokenize(paragraph)
    for iteration in range(10):
        paragraphSentiments = 0.0
        random.shuffle(sentence_list)
        for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            print("{:-<69} {}".format(sentence, str(vs["compound"])))
            paragraphSentiments += vs["compound"]
        print("iteration " + str(iteration) + " AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(paragraphSentiments / len(sentence_list), 4)))
        print()


def scoreSequence(valences):
    # Exercise 3c, rewrite me
    vs = 0
    for score in valences:
        vs += score["compound"]
    return vs
        
def exercise3c():
    # Consider Implementing an aggregation technique which is sentence order specific
    paragraph = "It was one of the worst movies I've seen, despite good reviews. \
        Unbelievably bad acting!! Poor direction. VERY poor production. \
        The movie was bad. Very bad movie. VERY BAD movie!"
    sentence_list = tokenize.sent_tokenize(paragraph)
    for iteration in range(10):
        random.shuffle(sentence_list)
        scores = map(analyzer.polarity_scores, sentence_list)
        vs = scoreSequence(scores)
        print("iteration " + str(iteration) + " AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(vs / len(sentence_list), 4)))
        print()


if __name__ == "__main__":
    # Exercise 1
    exercise1a()
    exercise1b()
    exercise1c()

    # Exercise 2
    exercise2a()
    exercise2b()
    exercise2c()

    # Exercise 3
    exercise3a()
    exercise3b()
    exercise3c()    

