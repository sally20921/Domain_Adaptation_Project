from torch import optim

class Adagrad(optim.Adagrad):
    @classmethod 
    def resolve_args(cls, args, params):

