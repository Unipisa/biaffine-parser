import shutil
import os
import argparse
import unittest

from tokenizer.cmds.predict import Predict

class TestTokenizer(unittest.TestCase):

    MODEL_DIR = '/tmp/stanza_models'
    
    def setUp(self):
        self.parser = argparse.ArgumentParser(
            description='Test argument parser'
        )
        self.parser.add_argument('--lang')
        self.parser.add_argument('--dir')
        self.parser.add_argument('--verbose', default=True)
        self.parser.add_argument('--device', default=1)

    def tearDown(self):
        shutil.rmtree(self.MODEL_DIR)
        
    def test_download_resources(self):
        self.assertTrue(not os.path.exists(self.MODEL_DIR))
        
        args = self.parser.parse_args(['--lang', 'it', '--dir', self.MODEL_DIR])
        tokenizer = Predict()
        tokenizer(args)
        
        self.assertTrue(os.path.exists(args.dir) and not os.path.isfile(args.dir))
        self.assertTrue(os.path.exists(os.path.join(args.dir, args.lang)))
        self.assertTrue(os.path.exists(os.path.join(args.dir, args.lang, 'tokenize')))

        sentences = tokenizer.predict('Domani vorrei andare al mare.Speriamo faccia bel tempo.')

    
    def test_tokenize(self):
        args = self.parser.parse_args(['--lang', 'it', '--dir', '/tmp/stanza_models'])
        tokenizer = Predict()
        tokenizer(args)

        doc = tokenizer.predict('Domani vorrei andare al mare.Speriamo faccia bel tempo.')

        self.assertEqual(len(doc.sentences), 2)
        
