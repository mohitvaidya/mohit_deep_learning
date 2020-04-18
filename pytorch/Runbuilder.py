from itertools import product

from collections import namedtuple
class Runbuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs =[]
        
        for i in product(*params.values()):
            runs.append(Run(*i))
            
        return runs
    