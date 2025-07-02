import torch
from abc import ABC, abstractmethod


class ProbPath(ABC):
    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def sample_p_simple(self, batch_size) -> torch.tensor:
        """
        returns torch.tensor(batch_size x self.dim) sampled from p_simple
        """
        raise NotImplementedError

    @abstractmethod
    def sample_conditional(self, t: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        returns torch.tensor(batch_size x self.dim) sampled from p_t(.|z)
        """
        raise NotImplementedError

    @abstractmethod
    def conditional_vector_field(
        self, x: torch.tensor, t: torch.tensor, z: torch.tensor
    ) -> torch.tensor:
        """
        x: torch.tensor(batch_size, self.dim)
        t: torch,tensor(batch_size, 1)
        z: torch.tensor(batch_size, self.dim)
        returns target u^*_t(x|z):
        """
        pass

    @abstractmethod
    def conditional_score_fun(
        self, x: torch.tensor, t: torch.tensor, z: torch.tensor
    ) -> torch.tensor:
        """
        x: torch.tensor(batch_size, self.dim)
        t: torch.tensor(batch_size, 1)
        z: torch.tensor(batch_size, self.dim)
        returns the gradient of log(p_t(x|z))
        """
        pass
