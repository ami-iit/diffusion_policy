import torch
from abc import ABC, abstractmethod
from diffusion_models_for_manipulation.paths.prob_paths import ProbPath


class FunctionWithDerivative(ABC):
    @abstractmethod
    def __call__(self, t):
        raise NotImplementedError

    @abstractmethod
    def dot(self, t):
        raise NotImplementedError


class LinearWithDerivative(FunctionWithDerivative):
    def __call__(self, t):
        return t

    def dot(self, t):
        return torch.ones_like(t)


class SqrtWithDerivative(FunctionWithDerivative):
    def __call__(self, t):
        return torch.sqrt(1 - t)

    def dot(self, t):
        return -1 / (2 * torch.sqrt(1 - t))


class GaussianConditionalProbPath(ProbPath):
    """
    {p_t(.!z)}_t gaussian conditional probability path
    where
    p_t(.|z) = N(alpha(t)*z, beta(t)^2 * I )

    The "noise schedulers" alpha and beta are functions with a first order derivative and must satisfy alpha(0)=beta(1)=0 and alpha(1)=beta(0)=1
    so that
    p_0(.|z) = N(0,I)
    p_1(.|z) = dirac(z)
    """

    def __init__(
        self, dim, alpha: FunctionWithDerivative, beta: FunctionWithDerivative
    ):
        super().__init__(dim)
        self.alpha = alpha
        self.beta = beta

    def sample_p_simple(self, batch_size=None):
        if batch_size is None:
            return torch.randn(self.dim)
        else:
            return torch.randn((batch_size, self.dim[0], self.dim[1]))

    def sample_conditional(self, t: torch.tensor, z: torch.tensor):
        return torch.einsum("ij,ikl->ikl", (self.alpha(t), z)) + torch.einsum(
            "ij,ikl->ikl", (self.beta(t), torch.randn_like(z))
        )

    def conditional_vector_field(self, x, z, t):
        alpha_t = self.alpha(t)
        alpha_t_dot = self.alpha.dot(t)
        beta_t = self.beta(t)
        beta_t_dot = self.beta.dot(t)
        return torch.einsum(
            "ij,ikl->ikl", ((alpha_t_dot - beta_t_dot / beta_t * alpha_t), z)
        ) + torch.einsum("ij,ikl->ikl", (beta_t_dot / beta_t, x))

    def conditional_score_fun(self, x, z, t):
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return torch.einsum(
            "ij,ikl->ikl",
            (1 / (beta_t**2), (torch.einsum("ij,ikl->ikl", ((alpha_t, z))) - x)),
        )

    def vec_field_from_score(self, score_x_t, t, x, sigma):
        a = (
            self.beta(t) ** 2 * self.alpha.dot(t) / (self.alpha(t) + 1e-3)
            - self.beta(t) * self.beta.dot(t)
            + sigma**2 / 2
        )
        return torch.einsum(
            "ij,ikl->ikl",
            (
                (
                    self.beta(t) ** 2 * self.alpha.dot(t) / (self.alpha(t) + 1e-3)
                    - self.beta(t) * self.beta.dot(t)
                    + sigma**2 / 2
                ),
                score_x_t,
            ),
        ) + torch.einsum("ij,ikl->ikl", (self.alpha.dot(t) / (self.alpha(t) + 1e-3), x))

    def score_from_vec_field(self, score_x_t, t, x, sigma):
        pass
