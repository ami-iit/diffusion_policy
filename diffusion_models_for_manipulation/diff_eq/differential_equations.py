# ODE/SDE
# euler
import numpy as np
import torch
from abc import ABC, abstractmethod


class ODE(ABC):
    """
    dX_t=drift(x,t)*dt
    """

    @abstractmethod
    def drift(self, x, t):
        raise NotImplementedError


class SDE(ABC):
    """
    dX_t=drift(x,t)*dt+diffusion(x,t)*dW_t
    """

    @abstractmethod
    def drift(self, x, t):
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x, t):
        raise NotImplementedError


class LearnedODE(ODE):
    def __init__(self, vec_field):
        self.vec_field = vec_field

    def drift(self, x, t):
        return self.vec_field(x, t)


class FullyLearnedSDE(SDE):
    """
    Langevin dynamics SDE
    Here both the vector field and the score function are learned
    """

    def __init__(self, vec_field, score, sigma):
        self.vec_field = vec_field
        self.score = score
        self.sigma = sigma

    def drift(self, x, t):
        return self.vec_field(x, t) + 0.5 * self.sigma**2 * self.score(x, t)

    def diffusion(self, x):
        return self.sigma * torch.rand_like(x)


class LearnedScoreSDE(SDE):
    """
    Langevin dynamics SDE
    Here only the score function are learned and it's assumed there exists a closed formula
    connecting the vector field and the score function as it's true for e.g. gaussian conditional paths
    """

    def __init__(self, score, sigma, path):
        self.path = path
        self.score = score
        self.sigma = sigma

    def drift(self, x, t):
        score_x_t = self.score(x, t)
        vec_field_x_t = self.path.vec_field_from_score(score_x_t, t, x, self.sigma)
        return vec_field_x_t + 0.5 * self.sigma**2 * score_x_t

    def diffusion(self, x, t):
        return self.sigma * torch.randn_like(x)
