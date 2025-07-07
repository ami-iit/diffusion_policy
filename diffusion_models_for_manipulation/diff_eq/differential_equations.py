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
    def diffusion(self, x, t=None):
        raise NotImplementedError


class LearnedODE(ODE):
    def __init__(self, vec_field):
        self.vec_field = vec_field

    def drift(self, x, t=None):
        return self.vec_field(x, t)


class FullyLearnedSDE(SDE):
    """
    Langevin dynamics SDE
    Here both the vector field and the score function are learned
    """

    def __init__(self, concat, sigma):
        self.concat = concat
        self.sigma = sigma

    def drift(self, x, t):
        vec_field_x_t, score_x_t = torch.split(self.concat(x, t), 2, dim=-1)
        return vec_field_x_t + 0.5 * self.sigma**2 * score_x_t

    def diffusion(self, x, t=None):
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

    def drift(self, x, t, cond=None):
        if cond is None:
            score_x_t = self.score(x, t)
        else:
            score_x_t = self.score(x, t, cond)
        vec_field_x_t = self.path.vec_field_from_score(score_x_t, t, x, self.sigma)
        return vec_field_x_t + 0.5 * self.sigma**2 * score_x_t

    def diffusion(self, x, t=None):
        return self.sigma * torch.randn_like(x)


class LearnedVectorFieldSDE(SDE):
    """
    Langevin dynamics SDE
    Here only the score function are learned and it's assumed there exists a closed formula
    connecting the vector field and the score function as it's true for e.g. gaussian conditional paths
    """

    def __init__(self, vec_field, sigma, path):
        self.path = path
        self.vec_field = vec_field
        self.sigma = sigma

    def drift(self, x, t, cond=None):
        if cond is None:
            vec_field_x_t = self.vec_field(x, t)
        else:
            vec_field_x_t = self.vec_field(x, t, cond)
        score_x_t = self.path.score_from_vec_field(vec_field_x_t, t, x, self.sigma)
        return vec_field_x_t + 0.5 * self.sigma**2 * score_x_t

    def diffusion(self, x, t=None):
        return self.sigma * torch.randn_like(x)
