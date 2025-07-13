import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import gymnasium as gym
import numpy as np
import copy

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

from mask_pack.common.constants import BIN, MASK

class CustomDummyVecEnv(VecEnv):
    """
    Custom is refer to to enable different observation space for each env.
    
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.all_obs_space = [None] * self.num_envs     # Per-environment observation spaces
        self.all_action_space = [None] * self.num_envs  # Per-environment action spaces
        for idx, env in enumerate(self.envs):
            obs_space = env.observation_space
            action_space = env.action_space
            self.all_obs_space[idx] = obs_space
            self.all_action_space[idx] = action_space
                
        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            if self.all_obs_space[env_idx] != self.observation_space:
                assert self.all_action_space[env_idx].n <= self.action_space.n, "The action space should be the same or larger than the target action space."
                self.actions[env_idx] = self._downscale_action(self.actions[env_idx], self.all_obs_space[env_idx])
                
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            
            # # upscale reward
            # if self.all_obs_space[env_idx] != self.observation_space:
            #     if self.buf_rews[env_idx] > 0:
            #         ratio = self.action_space.n / self.all_action_space[env_idx].n 
            #         self.buf_rews[env_idx] *= ratio
            
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None: 
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                if self.all_obs_space[env_idx] != self.observation_space:
                    assert self.all_obs_space[env_idx][BIN].shape[1] <= self.observation_space[BIN].shape[1], "The width of observation space should be the same or larger than the target observation space."
                    assert self.all_obs_space[env_idx][BIN].shape[2] <= self.observation_space[BIN].shape[2], "The height of observation space should be the same or larger than the target observation space."
                    obs = self._upscale_obs(obs)
                self.buf_obs[key][env_idx] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    def _upscale_obs(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        bin_data = obs[BIN]
        mask_data = obs[MASK] 
        
        # bin
        input_bin_c, input_bin_w, input_bin_h = obs[BIN].shape
        target_bin_c, target_bin_w, target_bin_h = self.observation_space[BIN].shape
        target_bin = np.zeros((target_bin_c, target_bin_w, target_bin_h), dtype=bin_data.dtype)
        target_bin[0, :input_bin_w, :input_bin_h] = bin_data[0] 
        fill_values_w = bin_data[1, 0, 0]
        target_bin[1] = fill_values_w[None, None] * np.ones((target_bin_w, target_bin_h), dtype=bin_data.dtype)
        fill_values_h = bin_data[2, 0, 0]
        target_bin[2] = fill_values_h[None, None] * np.ones((target_bin_w, target_bin_h), dtype=bin_data.dtype)
        
        # mask
        target_mask_shape = self.observation_space[MASK].shape
        mask_data = mask_data.reshape(input_bin_w, input_bin_h)
        target_mask = np.zeros((target_bin_w, target_bin_h), dtype=bin_data.dtype)
        target_mask[:input_bin_w, :input_bin_h] = mask_data
        target_mask = target_mask.reshape(target_mask_shape[0])
        
        return {BIN: target_bin, MASK: target_mask}

    def _downscale_action(
        self, 
        actions: np.ndarray, 
        orig_obs_sapce: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Downscale the action to the original shape.

        :param actions: The action to be downscaled
        :return: The downscaled action
        """
        input_bin_c, input_bin_w, input_bin_h = orig_obs_sapce[BIN].shape
        target_bin_c, target_bin_w, target_bin_h = self.observation_space[BIN].shape
        scale_i = actions // target_bin_h
        scale_j = actions % target_bin_h
        orig_action = scale_i * input_bin_h + scale_j
        orig_action = np.where(orig_action < (input_bin_w * input_bin_h), orig_action, (input_bin_w * input_bin_h - 1))
        return orig_action
    
    def rebuild_obs_buf(
        self, 
        obs_sapce: Union[np.ndarray, Dict[str, np.ndarray]], 
        action_space: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> None:
        self.observation_space = obs_sapce
        self.action_space = action_space
        obs_space = self.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
