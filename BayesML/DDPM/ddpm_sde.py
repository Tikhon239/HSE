import torch


class DDPM_SDE:
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.N = config.sde.N
        self.beta_0 = config.sde.beta_min
        self.beta_1 = config.sde.beta_max

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """
        Calculate drift coeff. and diffusion coeff. in forward SDE
        """
        beta_t = (self.beta_1 - self.beta_0) * t + self.beta_0
        drift = -0.5 * beta_t.reshape(-1, 1, 1, 1) * x
        diffusion = beta_t.sqrt().reshape(-1, 1, 1, 1)
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        integral = 0.5 * (self.beta_1 - self.beta_0) * t ** 2 + self.beta_0 * t
        mean = torch.exp(-0.5 * integral).reshape(-1, 1, 1, 1) * x_0
        std = torch.sqrt(1 - torch.exp(-integral)).reshape(-1, 1, 1, 1)
        return mean, std

    def marginal_std(self, t):
        """
        Calculate marginal q(x_t|x_0)'s std
        """
        integral = 0.5 * (self.beta_1 - self.beta_0) * t ** 2 + self.beta_0 * t
        std = torch.sqrt(1 - torch.exp(-integral)).reshape(-1, 1, 1, 1)
        return std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def reverse(self, score_fn, ode_sampling=False):
        """Create the reverse-time SDE/ODE.
        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          ode_sampling: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde

        # Build the class for reverse-time SDE.
        class RSDE:
            def __init__(self):
                self.N = N
                self.ode_sampling = ode_sampling

            @property
            def T(self):
                return T

            def sde(self, x, t, y=None):
                """
                Create the drift and diffusion functions for the reverse SDE/ODE.
                
                
                y is here for class-conditional generation through score SDE/ODE
                """
                
                """
                Calculate drift and diffusion for reverse SDE/ODE
                
                
                ode_sampling - True -> reverse SDE
                ode_sampling - False -> reverse ODE
                """
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t, y)

                deterministic_part = diffusion ** 2 * score

                if self.ode_sampling:
                    drift = drift - deterministic_part
                else:
                    drift = drift - 0.5 * deterministic_part
                    diffusion = torch.zeros_like(diffusion)

                return drift, diffusion

        return RSDE()


class EulerDiffEqSolver:
    def __init__(self, sde, score_fn, ode_sampling=False):
        self.sde = sde
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling
        self.rsde = sde.reverse(score_fn, ode_sampling)

    def step(self, x, t, y=None):
        """
        Implement reverse SDE/ODE Euler solver
        """
        
        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        dt = -1. / self.rsde.N
        drift, diffusion = self.rsde.sde(x, t, y)

        x_mean = x + drift * dt

        eps = torch.randn_like(x)
        x = x_mean + diffusion * abs(dt) ** 0.5 * eps
        return x, x_mean
