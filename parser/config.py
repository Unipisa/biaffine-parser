# -*- coding: utf-8 -*-

from ast import literal_eval
from configparser import ConfigParser
from argparse import Namespace


class Config(ConfigParser):

    def __init__(self, path):
        super(Config, self).__init__()

        self.read(path)
        self.namespace = Namespace()
        self.update(dict((name, literal_eval(value))
                         for section in self.sections()
                         for name, value in self.items(section)))

    def __repr__(self):
        s = line = "-" * 20 + "-+-" + "-" * 25 + "\n"
        s += f"{'Param':20} | {'Value':^25}\n" + line
        for name, value in vars(self.namespace).items():
            s += f"{name:20} | {str(value):^25}\n"
        s += line

        return s

    def __getattr__(self, attr):
        return getattr(self.namespace, attr)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, kwargs):
        for name, value in kwargs.items():
            setattr(self.namespace, name, value)

        return self
