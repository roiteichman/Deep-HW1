import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # ====== YOUR CODE: ======
        N = x.shape[0]

        x = torch.cat([torch.ones(N, 1), x], dim=1)

        margins = x_scores - x_scores[torch.arange(N), y].view(-1, 1) + self.delta

        margins[torch.arange(N), y] = 0

        loss_i = torch.sum(torch.maximum(torch.zeros_like(margins), margins), dim=1)

        loss = torch.mean(loss_i)

        # ========================

        # Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        self.grad_ctx['x_scores'] = x_scores
        self.grad_ctx['margins'] = margins
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """

        # ====== YOUR CODE: ======
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']
        x_scores = self.grad_ctx['x_scores']
        margins = self.grad_ctx['margins']

        N = x.shape[0]
        G = torch.zeros_like(x_scores)

        G[margins > 0] = 1

        row_sum = torch.sum(G, dim=1)

        G[torch.arange(N), y] = -row_sum

        grad = x.t().mm(G) / N

        return grad