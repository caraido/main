# -*- coding: utf-8 -*-
"""utils.text -- text and NLP helpers (spaCy-based)."""

import re

import spacy

# Loaded lazily on first use to keep module-import time low.
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm')
    return _nlp


# Backward-compat: the original utils.utils exposed a module-level `nlp` symbol.
# It is now lazily-initialised via _get_nlp(); access via utils.text.nlp will
# trigger the load.  Consumers should prefer `_get_nlp()` going forward.
class _NlpProxy:
    def __getattr__(self, item):
        return getattr(_get_nlp(), item)

    def __call__(self, *args, **kwargs):
        return _get_nlp()(*args, **kwargs)


nlp = _NlpProxy()

def replace_underscores(text, replacement):
   return re.sub(r'_{2,}', " "+replacement, text)


def add_space_after_comma(text):
    return re.sub(',',', ',text)


def get_sentence_tense(sentence):
    """
    Determines the tense of a given sentence based on temporal keywords, verb conjugations,
    and sentence structure using spaCy's NLP processing.

    The function first checks for specific temporal keywords that indicate past, present,
    or future tenses. If no keywords are found, it evaluates the tense of the verbs in the
    sentence using their Part-of-Speech (POS) tags. Additionally, it analyzes the sentence
    structure to determine tense in cases like imperative statements or sentences without a
    clear subject.

    :param sentence: The input sentence whose tense needs to be identified.
    :type sentence: str

    :return: The determined tense of the sentence. Possible values are:
             - "past": Indicates the sentence is in past tense.
             - "present": Indicates the sentence is in present tense.
             - "future": Indicates the sentence is in future tense.
             - "NA": Indicates the tense could not be determined (e.g., no verbs or keywords identified).
    :rtype: str
    """
    # Parse the sentence using spaCy
    sentence = sentence.replace('\r', '')
    sentence = sentence.lstrip()
    doc = nlp(sentence.replace(",", ""))

    # Keywords for temporal indicators
    past_keywords = {"yesterday", "last", "ago", "just now", "earlier"}
    present_keywords = {"right now", "usually", "often", "always", "currently", "now", "sometimes", "every", 'please'}
    future_keywords = {"tomorrow", "next", "soon", "in the future", "will"}

    tense = None

    # Identify temporal keywords
    for token in doc:
        if token.text.lower() in past_keywords:
            return "past"
        elif token.text.lower() in present_keywords:
            return "present"
        elif token.text.lower() in future_keywords:
            return "future"

    # Identify tense based on verb conjugation
    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            # Check the verb tense
            if token.tag_ in {"VBD", "VBN"}:  # Past tense or past participle
                return "past" if tense is None else tense
            elif token.tag_ in {"VBP", "VBZ"}:  # Present tense
                return "present" if tense is None else tense
            elif token.tag_ == "VB":  # Base form (could be infinitive)
                if "will" in [t.text.lower() for t in doc]:
                    return "future"
                elif tense:
                    return tense

    # 1. Find all root VERBs (should usually be just one)
    root_verbs = [token for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"]
    if not root_verbs:
        return "NA"

    # We'll assume there's only one root verb in a simple sentence
    root_verb = root_verbs[0]

    # 2. Check for subjects in the sentence
    subjects = [token for token in doc if token.dep_ in ("nsubj", "nsubjpass")]

    # 3. If there's no subject at all, that's usually imperative ("Open the door.")
    if not subjects:
        return "present"

    # 4. If the subject is explicitly "you", it can still be imperative ("You open the door now!")
    #    We check .lemma_ to handle “You” vs. “you”
    if len(subjects) == 1 and subjects[0].lemma_.lower() == "you":
        return "present"

    # If no verbs or temporal indicators found
    return "NA"


def get_sentence_subject_number(sentence):
    """
    Determines the grammatical subject's number and person in a given sentence.

    This function processes a given sentence using spaCy's NLP model and identifies the grammatical
    subject (nsubj) or root of the sentence. Based on the subject's text or grammatical properties,
    it determines whether the subject is first person singular, third person singular, or plural.
    If the subject cannot be identified or categorized, the function returns 'NA'.

    :param sentence: A string representing the sentence to analyze.
    :type sentence: str
    :return: A string specifying the grammatical subject's person and number. Possible values
             include 'first person singular', 'third person singular', 'plural', or 'NA'.
    :rtype: str
    """
    # Parse the sentence using spaCy
    doc = nlp(sentence)
    for token in doc:
        # Look for the subject (nsubj = nominal subject)
        if token.dep_ == "nsubj" or token.dep_ == 'ROOT':
            #
            if token.text.lower() in {"he", "she", "it","i"} or token.tag_ == "NN" or token.tag_ == "NNP":
                return "singular"
            # Plural: 'we', 'they', or plural nouns
            elif token.text.lower() in {"we", "they"} or token.tag_ == "NNS" or token.tag_=="NNPS":
                return "plural"
    return "NA"


def get_sentence_subject_person(sentence):
    """
    Determines the grammatical person of the subject in a given sentence.

    Analyzes the given sentence using spaCy to identify the nominal subject (nsubj)
    and evaluates whether the subject is in the first person, third person,
    or cannot be determined. If the subject includes pronouns such as "I", "we",
    "he", "she", "it", "they", or relevant noun tags, the function categorizes
    them. Returns a string indicating "first" for first-person, "third" for
    third-person, or "NA" if the grammatical person cannot be determined.

    :param sentence:
        The input sentence to analyze for determining the person's subject.
        It is expected as a string.
    :return:
        The grammatical person of the subject found in the sentence. Possible
        return values are:
        - "first": If the subject is in the first person, e.g., "I" or "we".
        - "third": If the subject is in the third person, e.g., "he", "she",
          "it", "they", or singular/plural nouns.
        - "NA": If no valid grammatical person for the subject was determined.
    """
    # Parse the sentence using spaCy
    doc = nlp(sentence)

    for token in doc:
        # Look for the subject (nsubj = nominal subject)
        if token.dep_ == "nsubj" or token.dep_ == 'ROOT':
            # First person singular: 'I'
            if token.text.lower() in {"i","we"}:
                return "first"
            # Third person singular: 'he', 'she', 'it', or a singular noun
            elif token.text.lower() in {"he", "she", "it","they"} or 'NN' in token.tag_:
                return "third"
            # Plural: 'we', 'they', or plural nouns

    return "NA"


def remove_number(text):
    """
    Remove picture numbers from target labels.
    
    Handles formats like:
    - "word + pic_number" -> "word" (e.g., "bank5" -> "bank")
    - "word + meaning_number + pic_number" -> "word + meaning_number" 
      (e.g., "date15" -> "date1", "fan210" -> "fan2")
    
    Args:
        text (str): Input text with picture numbers
        
    Returns:
        str: Text with only the final picture number removed
    """
    if not isinstance(text, str):
        return text
    
    # Extract the word part (letters) and number part
    word_part = ''.join([char for char in text if char.isalpha()])
    number_part = ''.join([char for char in text if char.isdigit()])
    
    if not number_part:
        # No numbers found, return as is
        return text
    
    # For single digit (1-9): it's just pic_number, remove it entirely
    if len(number_part) == 1:
        return word_part
    
    # For two digits (10, 11-19, 21-29): 
    # - 10: pic_number only, remove entirely -> word
    # - 11-19: meaning_number=1, pic_number=1-9 -> word1
    # - 21-29: meaning_number=2, pic_number=1-9 -> word2
    elif len(number_part) == 2:
        if number_part == '10':
            return word_part  # Just pic_number 10
        elif number_part.startswith('1'):
            return word_part + '1'  # meaning_number 1
        elif number_part.startswith('2'):
            return word_part + '2'  # meaning_number 2
        else:
            return word_part  # Other two-digit numbers are just pic_numbers
    
    # For three digits (110, 210): meaning_number + pic_number=10
    elif len(number_part) == 3:
        if number_part.startswith('1'):
            return word_part + '1'  # meaning_number 1
        elif number_part.startswith('2'):
            return word_part + '2'  # meaning_number 2
        else:
            return word_part  # Fallback
    
    # For other cases, return the word part
    else:
        return word_part
