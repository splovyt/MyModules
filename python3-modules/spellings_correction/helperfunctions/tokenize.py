import re

def tokenize(sentence: str) -> list:
    '''Convert a string into a bag of words, removing all punctuation.
    
    Args:
        sentence: a string
    
    Returns:
        tokens: bag of words in a list
    
    '''
    assert isinstance(sentence, str)

    TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
    tokens = TOKENIZER_RE.findall(sentence.lower())
    
    return tokens