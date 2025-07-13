"""Probability distributions."""

from typing import Any, Dict, Optional, Tuple, TypeVar

from gymnasium import spaces
import torch as th
from torch import nn
from torch.distributions import Categorical
from stable_baselines3.common.distributions import Distribution

SelfCategoricalDistribution = TypeVar("SelfCategoricalDistribution", bound="CategoricalDistribution")


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(
        self, 
        action_dim: int, 
        mask_strategy: str = "minus",
        mask_minus_coef: int = 15,
        mask_replace_coef: int = -1000000,
        update_actor_stratrgy: str = "masked",
        entropy_strategy: str = "naive",
        invalid_probs_strategy: str = "naive",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.mask_strategy = mask_strategy
        self.mask_minus_coef = mask_minus_coef
        self.mask_replace_coef = mask_replace_coef
        self.update_actor_stratrgy = update_actor_stratrgy
        self.entropy_strategy = entropy_strategy
        self.invalid_probs_strategy = invalid_probs_strategy

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self: SelfCategoricalDistribution, action_logits: th.Tensor, mask: Optional[th.Tensor] = None) -> SelfCategoricalDistribution:
        if mask is None:
            self.distribution = Categorical(logits=action_logits)
        else:
            inver_mask = 1 - mask
            if self.mask_strategy == "minus":
                m_action_logits = action_logits - self.mask_minus_coef * inver_mask
                self.distribution = Categorical(logits=m_action_logits)
            elif self.mask_strategy == "replace":
                m_action_logits = th.where(mask.bool(), action_logits, th.tensor(self.mask_replace_coef))
                self.distribution = Categorical(logits=m_action_logits)
            else:
                raise ValueError(f"mask_strategy {self.mask_strategy} is not supported")
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
    


def make_proba_distribution(
    action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(int(action_space.n), **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )