# ShakespeareanTextGenerator
Build a text generation model that works with 2-grams/3-grams/4-grams to imitate the style of William Shakespeare.

## Intro
In the foloowing repo we will create a _digital poet_ that resemble Shakespear poetry tecnique. The objective it to input a starting phrase and complete it based on Shakespear work.

This is possible thanks to word encoding tecnique called N-grams, that predict the probability of the next word given a sentence of _N_ words.

## Clean dataset
1. First thing we have to do is to load Shakespear poetry in our code. We can easily perform it using `nltk` library, that have preloaded some Corpus/Books/text data.

    ```python
    from nltk.corpus import shakespeare
    shakespeare.fileids()
    ```

    This lines of code shows the preloaded books. For semplicity and passion I'll chose Hamlet (_The Tragedy of Hamlet, Prince of Denmark_), the famus novel of _to be or not to be_.
2. We have to parse the XML containing the book content and extract the relevant text as a sting of 179465 elements!
3. For semplicity we also transform all the text into lowercase and remove punctuation.
4. Later we tokenize the text. This means splitting each word and creating a list of clean data, with lowercase words adn no punctuation

## N-Grams
N-Grams is a old tecnique. It require:

- **Corpus** (all text data or records of dataset): For us is Hamlet novel form Shakespear
- **Vocabulary** (all unique words in the dataset): For us are all unique words in the book
- **Tokens** (each individual word in sequence)

With this objects, we pass from text tokenized and clean (_as seen before_), to then create sub sets of frases with N words (_word embedig_) to predict with certain probability the next word in the phrase.

In this exercise we will compare the performance of diffrent grams:
- bigrams: Groups of 2 words
- trigrams: Groups of 3 words
- quadgrams: Groups of 4 words

With this subsets of the entire book, we can find the occurence of the sequence, and the possible words that appears after that specific sequence of words (`from_bigram_to_next_token_counts`). Then we can compute the probability of the next word (`from_bigram_to_next_token_probs`) and then select one of the possible words (`sample_next_token`)to generate text (`generate_text_from_bigram`).

**Example**:
Given the famus phrase _"To be, or not to be"_ we will preform the tasks seen since now:
1. Clean from punctuation and uppercase letters
    Result:  _"to be or not to be"_
2. Tokenization. Split each word and create list of words
    Result: ['to','be','or','not','to','be']
3. N-grams:
    - bigram = ['to','be'] and the next words that can occur are [('a', 0.08823529411764706), ('buried', 0.058823529411764705), ...]
    - trigram = ['to','be','or'] and the next word that can occur is [('not', 1.0)]
    - quadgram = ['to','be','or','not'] and the next word that can occur is [('not', 1.0)]

It is easy to notice that the more words we count in the gram (the bigger the N in the gram), the better the algoritm will perform to find the most accurate next word. But there is a trade of, because the biger the N, also the more accurate the input have to be otherwise the gram will not be founded in the text, therfore no word will be advised after!

## Test Case
I prepared a code to test the accuracy of generating the entire monolog of the book

```python
goal = 'to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles'
```

Given this first 30 words, I start with bigram ('to', 'be') and generate the next 30 words. Then I calculate the accuracy with the goal phrase. After doing it for also trigram and quadgram we can easily see that quadgram perform better, without leavinh space to the weighted random choise to chose a wrong word. Beacuse after _"to be"_ there can be many words, but after _"to be or not"_ there can only be _"to"_.

```
Generated text from bigram ('to', 'be'):
to be a villain kills my father s leave what says polonius lord polonius i would not this sir and therefore i forbid my tears but yet i hold my peace i
Accuracy: 0.058823529411764705

Generated text from trigram ('to', 'be', 'or'):
to be or not to crack the wind of me as if you would drive me into a towering passion horatio peace who comes here enter osric osric your lordship speaks most infallibly
Accuracy: 0.14705882352941177

Generated text from quadgram ('to', 'be', 'or', 'not'):
to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles
Accuracy: 1.0
```

## Human test

I asked my roomates to try my digital poet, telling him that he could give me some words and my poet was going to complete the prase. Of course it took a while for him to understand that he have to be specific with a number of words between 2 and 4, plus the words have to appear in that specific orther otherwise there won't be a match to detect the next word. At the end of each gram test I collected his feedbacks:

- bigram: _'this is not even close to what I was expecting! This digital poet is not starting well... As I thought... machine can't replace and not even eìimitate humans'_

- trigram: _'Mmmmh it started good, but then it start pasting random words... Gringo what is this ?!?!'_

- quadgram: _'Wow!! This are the exact words I was expecting... but still... everyone can find the test and start reading it... '_

## Code explenation
For detail explenation of how the code word see the comments in the file ([Shakespear_NLP.ipynb](https://github.com/GRINGOLOCO7/ShakespeareanTextGenerator/blob/main/Shakespear_NLP.ipynb))and run it after installing `nltk`