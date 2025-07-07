import torch
from diffusion_models_for_manipulation.diff_eq.differential_equations import ODE, SDE

"""
Numerical methods used to simulate the ODE/SDE in the sampling phase ("forward process")
"""


def euler(x0: torch.tensor, ode: ODE, num_timesteps) -> torch.tensor:
    """
    ODE:            dX_t = drift(x,t) * dt
    Euler method:   x_(t+h)= x_t + h * drift(x_t,t)

    x0: torch.tensor((num_samples, dim))
    ode: ODE to simulate
    num_timesteps: integer representing the number of steps to simulate the SDE over between t=0 and t=1

    returns x_1 of the same size as x0
    """
    t = torch.zeros((x0.shape[0], 1)).to(x0.device)
    x = x0
    h = 1 / num_timesteps
    for i in range(1, num_timesteps + 1):
        t = t + h
        x = x + h * ode.drift(x, t)
    return x


def euler_with_steps(x0: torch.tensor, ode: ODE, num_timesteps):
    """
    Same as above.
    Returns both x_1 and the tensor containing x_t for each t
    """
    t = torch.zeros((x0.shape[0], 1)).to(x0.device)
    x = x0
    h = 1 / num_timesteps
    xts = torch.zeros((num_timesteps + 1, x.shape[0], x.shape[1]))
    xts[0, :, :] = x
    for i in range(1, num_timesteps + 1):
        t = t + h
        x = x + h * ode.drift(x, t)
        xts[i, :, :] = x
    return x, xts


def euler_maruyama(
    x0: torch.tensor,
    sde: SDE,
    num_timesteps: int,
    cond: torch.tensor = None,
) -> torch.tensor:
    """
    SDE:                    dX_t = drift(x,t)*dt+diffusion(t)*dB_t
    Euler-Maruyama method:  x_(t+h) = x_t + h * drift(x_t,t) + sqrt(h) * diffusion(x_t,t)

    x0: torch.tensor((num_samples, dim))
    cond: torch.tensor((num_samples, cond_dim))
    sde: SDE to simulate
    num_timesteps: integer representing the number of steps to simulate the SDE over between t=0 and t=1

    returns x_1 of the same size as x0
    """
    t = torch.zeros((x0.shape[0], 1)).to(x0.device)
    x = x0
    h = torch.tensor(1 / num_timesteps)
    root_h = torch.sqrt(h)
    if cond is None:
        for i in range(1, num_timesteps + 1):
            t = t + h
            x = x + sde.drift(x, t) * h + root_h * sde.diffusion(x)
    else:
        for i in range(1, num_timesteps + 1):
            t = t + h
            x = x + sde.drift(x, t, cond) * h + root_h * sde.diffusion(x)
    return x


def euler_maruyama_with_steps(x0: torch.tensor, sde: SDE, num_timesteps):
    """
    Same as above.
    Returns both x_1 and the tensor containing x_t for each t
    """
    t = torch.zeros((x0.shape[0], 1)).to(x0.device)
    x = x0
    h = torch.tensor(1 / num_timesteps)
    root_h = torch.sqrt(h)
    xts = torch.zeros((num_timesteps + 1, x.shape[0], x.shape[1]))
    xts[0, :, :] = x
    for i in range(1, num_timesteps + 1):
        t = t + h
        x = x + sde.drift(x, t) * h + root_h * sde.diffusion(x)
        xts[i, :, :] = x
    return x, xts
