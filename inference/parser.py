from argparse import ArgumentParser

class Parser(): 
    def __init__(self, parser = ArgumentParser(), arg_list = []):
        self.parser = parser
        self.arg_list = arg_list
    def apply_args(self):
        for arg in self.arg_list: self.parser.add_argument(*arg)
        return self
    def return_args(self): 
        return self.parser.parse_args()
    