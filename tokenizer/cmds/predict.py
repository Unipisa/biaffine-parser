# -*- coding: utf-8 -*-

import stanza

# check https://github.com/stanfordnlp/stanza/blob/master/stanza/utils/prepare_tokenizer_data.py

class Predict:
    def __call__(self, args):
        stanza.download(args.lang, dir=args.dir, verbose=args.verbose)
        self.pipeline = stanza.Pipeline(args.lang, dir=args.dir, processors='tokenize', verbose=args.verbose, use_gpu=args.device != -1)

    def predict(self, text):
        return self.pipeline(text)

        
