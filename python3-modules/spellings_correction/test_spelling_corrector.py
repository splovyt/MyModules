import unittest
from spelling_corrector import SpellingCorrector # module to test

# non standard library
import pandas as pd
from gensim.models import KeyedVectors

class TestSpellingCorrector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''Code to run BEFORE everything else'''
        pass
    
    @classmethod
    def tearDownClass(cls):
        '''Code to run AFTER everything else'''
        pass


    def setUp(self):
        '''Code to run BEFORE EVERY test'''
        pass

    def tearDown(self):
        '''Code to run AFTER EVERY test'''
        pass

    def test_spelling_corrector(self):
        ''' Test the spelling correction.
        NOTE: requires sentence data and a word2vec model
        '''
        # load 
        print('loading word2vec model..')
        model = KeyedVectors.load_word2vec_format("/Users/simonplovyt/Desktop/wiki.nl.vec", binary=False) # word2vec model
        print('word2vec model successfully loaded!')

        # load data
        data = pd.read_csv('data/nl-Belfius_Train_Data.csv').drop('Entities ', axis=1).dropna() 
        sentences = list(data['Text']) # sentences

        # initialize spelling corrector
        SC = SpellingCorrector(sentence_database = sentences,
                                trained_word2vec_model = model,
                                mismatches_allowed = 3,
                                vocabulary = None)

        self.assertEqual(SC.correct('Kan je even mijn kredietkaartl imiet opvragen??'), 'kan je even mijn kredietkaart limiet opvragen')
        self.assertEqual(SC.correct('Kan je even mijn kreditekaart llimiet opvragen?'), 'kan je even mijn kredietkaart limiet opvragen')
        self.assertEqual(SC.correct('kreditekaart'), 'kredietkaart')
        self.assertEqual(SC.correct('Ik heb hlup nodig!'), 'ik heb hulp nodig')
        self.assertEqual(SC.correct('beste, ik ben deze ochtend mn creditcart verloren'), 'beste ik ben deze ochtend mn creditcard verloren')



# to run as "python3 test_spelling_corrector.py"
if __name__ == '__main__':
    unittest.main()