# author: Simon Plovyt (simon.plovyt@ml6.eu)
# date: September 2018
# python version: 3.5.2

# description:
# This module contains a Spelling Corrector to be used with word embeddings.
#
# algorithm:
# 1. build a database with correct sentences in the context of the application
# 2. identify words for which no embedding is found (using a word2vec model)
# 3. compare ngrams containing unrecognized words with correct ngrams in the database
#    if no close 3-gram is found, we will look at a close 2-gram, and finally an 1-gram by giving preference to the most occurring word
# 4. substitute unrecognized ngram with correct ngram if one is present

# standard library
from collections import Counter

import logging # save logs
# https://docs.python.org/3/library/logging.html#logrecord-attributes

# GENERAL LOG
logger = logging.getLogger(__name__) # create a separate log for this module
logger.setLevel(logging.INFO) # set logger level to INFO
file_handler = logging.FileHandler(filename='logs/spellingcorrector/spellingcorrector.log') # output file
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s') # output format
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# UNRECOGNIZED WORDS LOG
loggerUW = logging.getLogger('unrecognized_words') # create a separate log for this module
loggerUW.setLevel(logging.WARNING) # set logger level to INFO
file_handler = logging.FileHandler(filename='logs/spellingcorrector/unrecognized_words.log') # output file
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s') # output format
file_handler.setFormatter(formatter)
loggerUW.addHandler(file_handler)

# other requirements:
import numpy as np
from gensim.models import KeyedVectors # word2vec
from to_tokens import tokenize
from levenshtein_distance import levenshtein_distance

class SpellingCorrector:
    
    def __init__(self, sentence_database, trained_word2vec_model,
                 mismatches_allowed, vocabulary = None):
        '''Initialize the Spelling Corrector
        
        Args:
            sentence_database (list): list of correct database sentences (str)
            trained_word2vec_model (KeyedVectors): a trained word2vec instance
            
        Kwargs:
            mismatches_allowed (int): the fuzzy match threshold to determine whether the ngram matches a db ngram
            vocabulary (set):
            
        Returns: initalized spelling corrector instance
        '''
        assert isinstance(sentence_database, list), 'pass a list of sentences instead'
        assert isinstance(mismatches_allowed, int), 'the amount of mismatches allowed should be an integer'
        assert isinstance(vocabulary, (set, type(None))), 'the vocabulary must be a set of words'
        
        self.word2vec_model = trained_word2vec_model
        self.mismatches_allowed = mismatches_allowed
        self.vocabulary = vocabulary
        if self.vocabulary is not None:
            print('''A vocabulary list has been specified, which means that only fully recognized ngrams 
            will be kept. Keep in mind that this is only beneficial if the vocabulary list is complete.''')
        
        
        # sentence_database -> tokens; we call the sentence database the 'training data'
        # because we will train the spellings corrector using the ngrams of these sentences
        self.training_data_tokens = [tokenize(sentence) for sentence in sentence_database]
                
        # convert the tokens to n-grams
        self.training_data_3grams = self.__tokens_list_to_ngrams(self.training_data_tokens, n=3)
        self.training_data_2grams = self.__tokens_list_to_ngrams(self.training_data_tokens, n=2)
        self.training_data_1grams = self.__tokens_list_to_ngrams(self.training_data_tokens, n=1)
        
        # intialize dictionary to enable lookup previously handled cases
        self.best_db_ngram_match_dict = dict()

        # logging
        logger.info('SpellingsCorrector was started')
        return
    
    def __tokens_list_to_ngrams(self, tokens_list, n):
        '''Generate a ngram set using a list of sentence tokens.
        
        Args:
            tokens_list (list): list of sentences split in tokens
            n (int): n in ngrams
            
        Returns:
            a dictionary with all ngrams as keys and the occurrence of the ngram in the database sentences
        
        '''
        
        assert isinstance(tokens_list, list), 'tokens_list must be a list of tokens'
        assert isinstance(n, int), 'the n in ngrams must be an integer'
        # generate ngrams
        complete_ngram_list = []
        for tokens in tokens_list:
            ngram_list = [' '.join(tokens[idx:idx+n]) for idx, value in enumerate(tokens) if idx+n <= len(tokens)]
            for ngram in ngram_list:
                complete_ngram_list.append(ngram)
        
        # count the occurrences of the ngrams
        ngram_counts = Counter(complete_ngram_list)
        
        # remove ngrams not recognized in word2vec
        for ngram in ngram_counts.copy().keys():
            if n > 1:
                ngram_tokens = ngram.split(' ')
            else:
                ngram_tokens = [ngram]
            for word in ngram_tokens:
                try:
                    self.word2vec_model[word]
                except:
                    try: del ngram_counts[ngram]
                    except: pass
        
        # if we have specified a vocabulary list, we will remove the ngrams 
        # with words that are not in the vocabulary list
        if self.vocabulary is not None:
            for ngram in ngram_counts.copy().keys():
                if n > 1:
                    ngram_tokens = ngram.split(' ')
                else:
                    ngram_tokens = [ngram]
                for word in ngram_tokens:
                    if word not in self.vocabulary:
                        try: del ngram_counts[ngram]
                        except: pass
                    
        return ngram_counts
        
    def best_ngram_match(self, tokens, ngram_database, max_length_difference = 10):
        '''Find the best ngram match in the database and return the Levenshtein distance.
        
        Args:
            tokens (list): bag of words
            database (dict): dictionary with ngram as key and value the count
        
        Kwargs:
            max_length_difference (int): the maximum length difference for which we will calculate
                the Levenshtein difference
            
        Returns:
            (Levenshtein distance of closest match, closest)
        
        '''
        assert isinstance(tokens, list)
        assert isinstance(ngram_database, dict)
        
        input_ngram = ' '.join(tokens)
        
        # check if we have already handled this case
        try: 
            if input_ngram in self.best_db_ngram_match_dict: return self.best_db_ngram_match_dict[input_ngram]
        except: 
            self.best_db_ngram_match_dict = dict() # initialize dictonary if we had not
    
    
        best_ngram = (np.Inf, []) # (distance, match_list)

        # look for the closest ngram in the database
        # sort the ngrams by occurrence in descending order
        ngram_database_sorted = sorted(ngram_database.items(), key=lambda kv: kv[1], reverse=True)
        for ngram, _ in ngram_database_sorted:
            
            # if the length difference between both strings is smaller than the maximum allowed difference,
            # we will calculate the Levenshtein distance and save the best result
            if abs(len(input_ngram)-len(ngram)) < max_length_difference:
                dist = levenshtein_distance(input_ngram, ngram)
                
                # if the current db ngram is a better match, overwrite the last
                # if it scores the same, we will take the most occurring ngram simply because
                # the list is sorted by occurrence
                if dist < best_ngram[0]:
                    best_ngram = (dist, [ngram])
        
        # save the result for future lookups
        self.best_db_ngram_match_dict[input_ngram] = best_ngram
                    
        return best_ngram
    
    def unrecognized_words(self, tokens):
        '''Returns a list of unrecognized tokens from a bag of tokens.
        
        Args:
            tokens (list): list of tokens
            
        Returns:
            unrecognized (list): list of unrecognized tokens
        
        '''
        assert isinstance(tokens, list)
        unrecognized = []
        for word in tokens:
            # make sure we ignore integers
            try: int(word)
            except:
                try: self.word2vec_model[word] # success if recognized
                except: 
                    unrecognized.append(word) # fail if unrecognized
                    loggerUW.warning(word)
        return unrecognized
    
    def __correct_ngram_with_database(self, ngram, database):
        ''' Compare the ngram with the database and return the spell corrected ngram.
        
        Args:
            ngram (list): ngram as list (example: ['hello', 'world'])
            database (dict): keys = ngrams as string (ex. 'hello world'), value = occurrence in database (int)
        
        Returns:
            ngram as list
        '''
        assert isinstance(ngram, list)
        assert isinstance(database, dict)
        
        best_database_score, best_database_matches = self.best_ngram_match(ngram, database, self.mismatches_allowed)
        if best_database_score <= self.mismatches_allowed:
            return best_database_matches[0].split(' ')
        return ngram
        
    def __ngram_spelling_proofer(self, tokens, unrecognized_tokens):
        '''Compare the ngram containing unrecognized words with ngrams in the database 
        and select the most similar correct ngram if one is present.
        
        Args:
            tokens (list): bag of words
            unrecognized_tokens (list): bag of unrecognized words from the tokens list
            
        Returns:
            proofed bag of words (list)
        '''
        assert isinstance(tokens, list)
        assert isinstance(unrecognized_tokens, list)

        proofed_tokens = tokens
        
        for unrecognized in unrecognized_tokens:
            indices_of_unrecognized = [i for i, x in enumerate(tokens) if x == unrecognized]
            
            # go over every occurrence of the unrecognized word in the tokens
            for idx in indices_of_unrecognized:
                ###############
                ### 3-grams ###
                ###############
                
                # if we can find a 3gram SURROUNDING the unrecognized word
                if idx-1 >= 0 and idx +2 <= len(proofed_tokens):
                    ngram_surrounding3 = proofed_tokens[idx-1:idx+2]
                    new_ngram = self.__correct_ngram_with_database(ngram = ngram_surrounding3,
                                              database = self.training_data_3grams)
                    if new_ngram != ngram_surrounding3:
                        # if a correction was made, we will update the tokens
                        proofed_tokens[idx] = new_ngram[1]
                        continue # go to the next occurrence
                
                # if we didnt make a correction, look at the other option of 3-grams, which is LEFT
                if idx-2 >= 0 and idx +1 <= len(proofed_tokens):
                    ngram_left3 = proofed_tokens[idx-2:idx+1]
                    
                    new_ngram = self.__correct_ngram_with_database(ngram = ngram_left3,
                                              database = self.training_data_3grams)
                    
                    if new_ngram != ngram_left3:
                        # if a correction was made, we will update the tokens
                        proofed_tokens[idx] = new_ngram[2]
                        continue # go to the next occurrence
                
                # if we didnt make a correction, look at the other option of 3-grams, which is RIGHT
                if idx >= 0 and idx +3 <= len(proofed_tokens):
                    ngram_right3 = proofed_tokens[idx:idx+3]
                    
                    new_ngram = self.__correct_ngram_with_database(ngram = ngram_right3,
                                              database = self.training_data_3grams)
                    if new_ngram != ngram_right3:
                        # if a correction was made, we will update the tokens
                        proofed_tokens[idx] = new_ngram[0]
                        continue # go to the next occurrence
                        
                ###############
                ### 2-grams ###
                ###############
                
                # RIGHT 
                if idx >= 0 and idx +2 <= len(proofed_tokens): 
                    ngram_right2 = proofed_tokens[idx:idx+2]
                    
                    new_ngram = self.__correct_ngram_with_database(ngram = ngram_right2,
                                              database = self.training_data_2grams)
                    if new_ngram != ngram_right2:
                        # if a correction was made, we will update the tokens
                        proofed_tokens[idx] = new_ngram[0]
                        continue # go to the next occurrence
                
                # LEFT
                if idx-1 >= 0 and idx +1 <= len(proofed_tokens):
                    ngram_left2 = proofed_tokens[idx-1:idx+1]
                    
                    new_ngram = self.__correct_ngram_with_database(ngram = ngram_left2,
                                              database = self.training_data_2grams)
                    if new_ngram != ngram_left2:
                        # if a correction was made, we will update the tokens
                        proofed_tokens[idx] = new_ngram[1]
                        continue # go to the next occurrence
                
                ###############
                ### 1-grams ###
                ###############
                new_ngram = self.__correct_ngram_with_database(ngram = [proofed_tokens[idx]],
                                              database = self.training_data_1grams)
                if new_ngram != proofed_tokens[idx]:
                    # if a correction was made, we will update the tokens
                    proofed_tokens[idx] = new_ngram[0]
                    continue # go to the next occurrence
                
                # 1grams not implemented because bad performance when no context

        # logging
        logger.info('proofread "{}" and outputted "{}"'.format(' '.join(tokens), ' '.join(proofed_tokens)))
        return proofed_tokens
    
    def correct(self, sentence):
        '''
        Correct the sentence using ngrams in the database.
        
        Args: 
            sentence (str): an input sentence
        
        Returns:
            (str): corrected input sentence, but with punctuation removed and lowercase
        
        '''
        assert isinstance(sentence, str)
        
        
        # sentence -> tokens
        tokens = tokenize(sentence)
        
        # check if there are any words for which word2vec is not trained
        unrecognized_tokens = self.unrecognized_words(tokens)
        
        # correct the sentence if required
        if len(unrecognized_tokens) > 0:
            tokens = self.__ngram_spelling_proofer(tokens, unrecognized_tokens)
        
        return ' '.join(tokens)