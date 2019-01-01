import unittest
from preclassifier import Preclassifier  # module to test

# non standard library
import pandas as pd


class TestPreclassifier(unittest.TestCase):

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

    def test_list1(self):
        '''
        Test the module
        '''

        # the data
        test_X = ['monkey', 'monkey', 'monkey', 'monkey', 'monkey']
        test_y = ['see', 'see', 'see', 'do', 'do']

        # initialize the preclassifier
        pc = Preclassifier(test_X, test_y)

        # the tests
        self.assertEqual(pc.single_query('monkey'), "see")
        self.assertEqual(pc.single_query('ape'), None)

    def test_list2(self):
        '''
        Test the module
        '''

        # the data
        test_X = [[1,2,3,4,'a'], [1,2,3,4,"a"], [1,2,3,4,'a'], [1,2,3,4,"a"], [1,2,3,4,'a']]
        test_y = ['see', 'do', 'see', 'do', 'see']

        # initialize the preclassifier
        pc = Preclassifier(test_X, test_y)

        # the tests
        self.assertEqual(pc.single_query([1,2,3,4,"a"]), "see")
        self.assertEqual(pc.single_query('ape'), None)

        self.assertEqual(pc.multiple_query([[1,2,3,4,"a"], [1,2,3,4,"a"], [1,2,3,4,'a']]), ["see", "see", "see"])
        self.assertEqual(pc.multiple_query(['ape', "ape"]), [None, None])

        self.assertEqual(pc.multiple_query([[1, 2, 3, 4, "a"], [1, 2, 3, 4, "a"], [1, 2, 3, 4, 'a'], "Monkey"]),
                         ["see", "see", "see", None])
        self.assertEqual(pc.multiple_query(['ape', "ape", [1, 2, 3, 4, "a"]]), [None, None, "see"])

    def test_pandas1(self):
        '''
        Test the module
        '''

        # the data
        df = pd.DataFrame()
        df['X'] = ['monkey', 'monkey', 'monkey', 'monkey', 'monkey']
        df['y'] = ['see', 'see', 'see', 'do', 'do']


        # initialize the preclassifier
        pc = Preclassifier(df['X'].tolist(), df['y'].tolist())

        # the tests
        self.assertEqual(pc.single_query('monkey'), "see")
        self.assertEqual(pc.single_query('ape'), None)

# to run as "python3 test_preclassifier.py"
if __name__ == '__main__':
    unittest.main()