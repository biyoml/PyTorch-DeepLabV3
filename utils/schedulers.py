from torch.optim.lr_scheduler import LambdaLR


class PolynomialLR(LambdaLR):
    def __init__(self, optim, power, total_steps, *args, **kwargs):
        lr_lambda = lambda step: max(1 - step / total_steps, 0) ** power   # noqa: E731
        super().__init__(optim, lr_lambda, *args, **kwargs)
