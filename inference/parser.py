from argparse import ArgumentParser

class Parser(): 
    def __init__(self, parser = ArgumentParser(), arg_list = []):
        self.parser = parser
        self.arg_list = arg_list
    def apply_args(self):
        for arg in self.arg_list: self.parser.add_argument(*arg)
        return self
    def assert_model_parallel(self):
        assert self.return_args.n_experts % self.return_args().model_parallel == 0
        return self
    def assert_interactive():
        assert self.return_args().input_file or self.return_args().interactive
        return self
    def return_args(self): 
        return self.parser.parse_args()
    