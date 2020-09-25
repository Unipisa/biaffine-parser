# -*- coding: utf-8 -*-

from ast import literal_eval
from configparser import ConfigParser


class Config():

    def __init__(self, path):
        super().__init__()

        config = ConfigParser()
        config.read(path)
        self.update(dict((name, literal_eval(value))
                         for section in config.sections()
                         for name, value in config.items(section)))

    def __repr__(self):
        s = line = "-" * 20 + "-+-" + "-" * 25 + "\n"
        s += f"{'Param':20} | {'Value':^25}\n" + line
        for name, value in vars(self).items():
            s += f"{name:20} | {str(value):^25}\n"
        s += line

        return s

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, kwargs):
        for key in ('self', 'cls', '__class__'):
            kwargs.pop(key, None)
        kwargs.update(kwargs.pop('kwargs', dict()))
        for name, value in kwargs.items():
            setattr(self, name, value)

        return self
