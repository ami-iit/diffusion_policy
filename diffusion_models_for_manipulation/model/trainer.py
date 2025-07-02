import torch
from abc import ABC, abstractmethod
from src.paths.prob_paths import ProbPath


class Trainer(ABC):

    def __init__(self, model: torch.nn.Module, path: ProbPath):
        self.model = model
        self.path = path

    @abstractmethod
    def get_loss(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_optimizer(self, learning_rate: float):
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs: int, device, learning_rate: float = 1e-3, **kwargs):
        self.model.to(device)
        opt = self.get_optimizer(learning_rate=learning_rate)
        self.model.train()
        for idx, _ in enumerate(range(num_epochs)):
            opt.zero_grad()
            loss = self.get_loss(**kwargs)
            loss.backward()
            opt.step()
        self.model.eval()

    def train(self, num_epochs: int, device, learning_rate: float = 1e-3, **kwargs):
        self.model.to(device)
        opt = self.get_optimizer(learning_rate=learning_rate)
        self.model.train()
        for idx, _ in enumerate(range(num_epochs)):
            opt.zero_grad()
            loss = self.get_loss(**kwargs)
            loss.backward()
            opt.step()
        self.model.eval()


class TrainerVectorField(Trainer):
    def get_loss(self, batch_size: int):
        """
        returns 1/N *sum_i((|| ut_i_th(x_i) - ut_i_target(x_i|z_i) ||^2))
        """
        z = self.path.sample_data(batch_size)
        t = torch.rand(batch_size, 1).to(z)
        x = self.path.sample_conditional(t=t, z=z)
        ut_theta = self.model(x, t)
        ut_target = self.path.conditional_vector_field(x=x, z=z, t=t)
        return torch.mean(torch.sum((ut_theta - ut_target) ** 2, dim=-1))


class TrainerScoreFun(Trainer):
    def get_loss(self, batch_size: int):
        """
        returns 1/N *sum_i((|| score_t_i_th(x_i) - score_t_i_target(x_i|z_i) ||^2))
        """
        z = self.path.sample_data(batch_size)
        t = torch.rand(batch_size, 1).to(z)
        x = self.path.sample_conditional(t=t, z=z)
        scoret_theta = self.model(x, t)
        scoret_target = self.path.conditional_score_fun(x=x, z=z, t=t)
        return torch.mean(torch.sum((scoret_theta - scoret_target) ** 2, dim=-1))


class TrainerComplete(Trainer):
    def get_loss(self, batch_size: int):
        """
        returns the sum of the two losses above
        """
        z = self.path.sample_data(batch_size)
        t = torch.rand(batch_size, 1).to(z)
        x = self.path.sample_conditional(t=t, z=z)
        complete_t_theta = self.model(x, t)
        ut_target = self.path.conditional_vector_field(x=x, z=z, t=t)
        scoret_target = self.path.conditional_score_fun(x=x, z=z, t=t)
        complete_t_target = torch.concat(ut_target, scoret_target)
        return torch.mean(
            torch.sum((complete_t_theta - complete_t_target) ** 2, dim=-1)
        )
