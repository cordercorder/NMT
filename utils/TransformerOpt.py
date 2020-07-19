from torch import optim


class TransformerOpt(optim.Adam):

    def __init__(self, params, init_lr):

        super().__init__(params, lr=init_lr)
        self.lr = init_lr
        self.step_num = 0

    def get_lr(self):
        return self.lr / (self.step_num ** 0.2)

    def step(self, closure=None):

        self.step_num += 1
        new_lr = self.get_lr()
        for param_group in self.param_groups:
            param_group['lr'] = new_lr

        super().step(closure)
