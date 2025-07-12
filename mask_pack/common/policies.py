import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, ClassVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.distributions import Distribution

from mask_pack.common.torch_layers import (
    BaseNetwork,
    CnnMlpNetwork1, 
    CnnMlpNetwork2, 
    CnnMlpNetwork3, 
    TransfromerNetwork1,
    TransfromerNetwork3,
)
from mask_pack.common.preprocessing import preprocess_obs
from mask_pack.common.distributions import (
    CategoricalDistribution, 
    make_proba_distribution,
)
from mask_pack.common.constants import BIN, MASK

class CustomActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)

    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """   

    network_aliases: ClassVar[Dict[str, Type[BaseNetwork]]] = {
        "CnnMlpNetwork1": CnnMlpNetwork1,
        "CnnMlpNetwork2": CnnMlpNetwork2,
        "CnnMlpNetwork3": CnnMlpNetwork3,
        "TransfromerNetwork1": TransfromerNetwork1,
        "TransfromerNetwork3": TransfromerNetwork3,
    }

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        # net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        # features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        # features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        # share_features_extractor: bool = True,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        network: Union[str, Type[BaseNetwork]] = CnnMlpNetwork1,
        network_kwargs: Optional[Dict[str, Any]] = None,
        dist_kwargs: Optional[Dict[str, Any]] = None,
        mask_type: str = "truth",
    ):
        if isinstance(network, str):
            self.network_class = self._get_network_from_name(network)
        else:
            self.network_class = network
        self.network_kwargs = network_kwargs
        self.mask_type = mask_type

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            # features_extractor_class,
            # features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.log_std_init = log_std_init
        assert not (squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_network_from_name(self, network_name: str) -> Type[BaseNetwork]:
        """
        Get a network class from its name representation.

        :param network_name: Alias of the network
        :return: A network class (type)
        """
        if network_name in self.network_aliases:
            return self.network_aliases[network_name]
        else:
            raise ValueError(f"Policy {network_name} unknown")

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                # net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                # features_extractor_class=self.features_extractor_class,
                # features_extractor_kwargs=self.features_extractor_kwargs,
                network_class=self.network_class,
                network_kwargs=self.network_kwargs,
            )
        )
        return data
    
    def _build_network(self) -> None:
        self.network = self.network_class(
            observation_space=self.observation_space[BIN], 
            action_dim=int(self.action_space.n), 
            **self.network_kwargs
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_network()

        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.network.actor_net: 0.01,
                self.network.critic_net: 1,
            }
            if self.network.share_extractor is not None:
                module_gains[self.network.share_extractor] = np.sqrt(2)

            if self.network.mask_net is not None:
                module_gains[self.network.mask_net] = 0.01

            if self.network.attention is not None:
                module_gains[self.network.attention] = np.sqrt(2)

            if self.network.mask_extractor is not None:
                module_gains[self.network.mask_extractor] = np.sqrt(2)
                module_gains[self.network.actor_extractor] = np.sqrt(2)
                module_gains[self.network.critic_extractor] = np.sqrt(2)

            if self.network.transformer_encoder is not None:
                module_gains[self.network.transformer_encoder] = np.sqrt(2)
                module_gains[self.network.l1] = np.sqrt(2)
                
            if self.network.transformer is not None:
                module_gains[self.network.transformer] = np.sqrt(2)
                module_gains[self.network.l1] = np.sqrt(2)
                module_gains[self.network.l2] = np.sqrt(2)
                if self.network.l3 is not None:
                    module_gains[self.network.l3] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def binary(self, input:th.Tensor):
        """
        Convert input tensor to binary tensor.\n
        If the value of input tensor is greater than 0.5, the output tensor will be 1, otherwise 0.\n

        Parameters
        ----------
        input : torch.Tensor
            The input tensor
        
        Returns
        -------
        out: torch.Tensor
            The binary tensor
        """
        a = th.ones_like(input)
        b = th.zeros_like(input)
        output = th.where(input >= 0.5, a, b)
        return output
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, log probability of the action and predicted mask probabilities
        """
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        mask_probs, action_logits, values = self.network(obs[BIN], obs[MASK])
        pred_mask = self.binary(mask_probs)
        
        if self.mask_type == "truth":
            distribution = self._get_action_dist_from_latent(action_logits, obs[MASK])
        elif self.mask_type == "predict":
            distribution = self._get_action_dist_from_latent(action_logits, pred_mask)
        elif self.mask_type == "none":
            distribution = self._get_action_dist_from_latent(action_logits, None)
        else:
            raise ValueError("Unknown mask type")

        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_probs, mask_probs
    
    def _get_action_dist_from_latent(self, mean_actions: th.Tensor, mask: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution by given the latent codes.

        :param mean_actions: The latent code
        :return: Action distribution
        """
        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions, mask=mask)
        else:
            raise ValueError("Invalid action distribution")
        
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        return actions, state  # type: ignore[return-value]

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        obs = preprocess_obs(observation, self.observation_space, normalize_images=self.normalize_images)
        mask_probs, action_logits, values = self.network(obs[BIN], obs[MASK])
        pred_mask = self.binary(mask_probs)

        if self.mask_type == "truth":
            distribution = self._get_action_dist_from_latent(action_logits, obs[MASK])
        elif self.mask_type == "predict":
            distribution = self._get_action_dist_from_latent(action_logits, pred_mask)
        elif self.mask_type == "none":
            distribution = self._get_action_dist_from_latent(action_logits, None)
        else:
            raise ValueError("Unknown mask type")
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            , entropy of the action distribution and the probability of invalid actions.
        """
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        mask_probs, action_logits, values = self.network(obs[BIN], obs[MASK])
        pred_mask = self.binary(mask_probs)

        if self.action_dist.update_actor_stratrgy == "naive":
            distribution = self._get_action_dist_from_latent(action_logits, mask=None)
        elif self.action_dist.update_actor_stratrgy == "masked":
            if self.mask_type == "truth":
                distribution = self._get_action_dist_from_latent(action_logits, obs[MASK])
            elif self.mask_type == "predict":
                distribution = self._get_action_dist_from_latent(action_logits, pred_mask)
            elif self.mask_type == "none":
                distribution = self._get_action_dist_from_latent(action_logits, None)
            else:
                raise ValueError("Unknown mask type")
        else:
            raise ValueError("Unknown update actor strategy")
        log_prob = distribution.log_prob(actions)

        if self.action_dist.entropy_strategy == "naive":
            distribution = self._get_action_dist_from_latent(action_logits, mask=None)
        elif self.action_dist.entropy_strategy == "masked":
            if self.mask_type == "truth":
                distribution = self._get_action_dist_from_latent(action_logits, obs[MASK])
            elif self.mask_type == "predict":
                distribution = self._get_action_dist_from_latent(action_logits, pred_mask)
            elif self.mask_type == "none":
                distribution = self._get_action_dist_from_latent(action_logits, None)
            else:
                raise ValueError("Unknown mask type")
        else:
            raise ValueError("Unknown entropy strategy")
        entropy = distribution.entropy()

        if self.action_dist.invalid_probs_strategy == "naive":
            distribution = self._get_action_dist_from_latent(action_logits, mask=None)
        elif self.action_dist.invalid_probs_strategy == "masked":
            if self.mask_type == "truth":
                distribution = self._get_action_dist_from_latent(action_logits, obs[MASK])
            elif self.mask_type == "predict":
                distribution = self._get_action_dist_from_latent(action_logits, pred_mask)
            elif self.mask_type == "none":
                distribution = self._get_action_dist_from_latent(action_logits, None)
            else:
                raise ValueError("Unknown mask type")
        else:
            raise ValueError("Unknown invalid probs strategy")
        invalid_probs = (distribution.distribution.probs * (1 - obs[MASK])).sum(dim=-1)

        return values, log_prob, entropy, invalid_probs

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.network.forward_critic(obs[BIN], obs[MASK])
    
    def predict_masks(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated mask probabilities according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated mask probabilities.
        """
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.network.forward_mask_probs(obs[BIN], obs[MASK])
