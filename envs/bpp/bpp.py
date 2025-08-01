from typing import Optional, Tuple, Union, Any, Dict

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from pat.common.constants import BIN, MASK
from envs.bpp.creator import ItemsCreator
from envs.bpp.bin import Bin

__all__ = ["BppEnv"]


class BppEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        bin_w: int = 5, 
        bin_h: int = 5, 
        bin_channels: int = 3,
        min_items_per_bin: int = 10,
        max_items_per_bin: int = 15,
        area_reward_coef: float = 0.4,
        constant_penalty: float = -5.0,
        action_fail: str = "continue",
        reward_type: str = "area",
    ):
        self.render_mode = render_mode
        self.bin_w = bin_w
        self.bin_h = bin_h
        self.bin_channels = bin_channels
        self.min_items_per_bin = min_items_per_bin
        self.max_items_per_bin = max_items_per_bin
        self.action_dim = bin_w * bin_h
        self.action_space = spaces.Discrete(bin_w * bin_h)
        self.observation_space = spaces.Dict({
            MASK: spaces.Box(low=0, high=1, shape=(bin_w*bin_h,), dtype=np.uint8),
            BIN: spaces.Box(low=0, high=max(bin_w, bin_h), 
                              shape=(self.bin_channels, bin_w, bin_h), dtype=np.uint8),                        
        })
        self.bin = Bin(width=bin_w, height=bin_h)
        self.items_creator = ItemsCreator(
            bin_w=bin_w, 
            bin_h=bin_h, 
            min_items_per_bin=min_items_per_bin,
            max_items_per_bin=max_items_per_bin,
        )
        
        self.area_reward_coef = area_reward_coef
        self.constant_penalty = constant_penalty
        self.action_fail = action_fail
        self.reward_type = reward_type

    def step(
        self, 
        action: Union[int, np.int64, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """   
        Take a step in the environment. \n
        The environment will put the item into the bin according to the action. \n
        If the action is invalid, the environment will continue to operate until using up all item data of the current episode. \n

        Parameters
        ----------
        action: Union[int, np.int64, np.ndarray]
            The action to take. \n

        Returns
        -------
        observation: Dict[str, np.ndarray]
            The observation after taking the action.
        reward: float
            The reward after taking the action.
        terminated: bool
            Whether the episode is done.
        truncated: bool
            Whether the episode is truncated.
        info: Dict[str, Any]
            The information of the current episode.
        """
        if isinstance(action, np.ndarray):
            idx = action[0]
        elif isinstance(action, (np.int64, int)):
            idx = action
        else:
            raise TypeError("Action type error, action should be a int or np.ndarray.")
        
        assert idx < self.action_dim, "Action index is out of range, please check the action index." 

        is_success = self.bin.put_item(self._get_next_item(), idx)

        if is_success:
            if self.reward_type == "area":
                reward = self.area_reward_coef * self._get_next_item_area()
            elif self.reward_type == "compactness":
                reward = self._compute_compactness_reward(action=idx)
            terminated = False
        else:
            reward = self.constant_penalty
            if self.action_fail == "continue":
                terminated = False
            elif self.action_fail == "terminate":
                terminated = True
            else:
                raise ValueError("Invalid `action_fail` parameter, action_fail should be 'continue' or 'terminate'.")

        self.items_creator.drop_item()
        if self.items_creator.items_list_length() == 0:
            terminated = True  # used up all items
            self.items_creator._add_unit_item() # ensure next state is able to be observed

        truncated = False
        info = {'bin': self._get_bin(), 'SU': self._get_bin_SU(), 'next_item': self._get_next_item(), 'mask': self._get_mask_obs()}
        return self._get_obs(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment. \n
        The environment will reset the bin and the items creator. \n
        Make sure to call this function before starting a new episode. \n

        Parameters
        ----------
        seed: int (default=None)
            The random seed for the environment. \n
            If None, the environment will not set the random seed. \n
        options: dict (default=None)
            The options for the environment. \n
        
        Returns
        -------
        observation: np.ndarray
            The observation after resetting the environment.
        info: dict
            The information of the current episode.
        """
        super().reset(seed=seed)
        self.bin = Bin(width=self.bin_w, height=self.bin_h)
        self.items_creator.reset(np_random=self.np_random)
        return self._get_obs(), {}
        
    def render(self) -> None:
        lx, ly = self.bin.items_list[-1].lx, self.bin.items_list[-1].ly
        item_width, item_height = self.bin.items_list[-1].width, self.bin.items_list[-1].height
        image = self.bin2image(self._get_bin(), lx, ly, item_width, item_height)
        image.figure.savefig('test.png')
        plt.pause(self.metadata["render_fps"])  
        plt.close(image.figure)

    def close(self) -> None:
        pass

    def _get_bin(self) -> np.ndarray:
        """
        Return the `bin`.

        Returns
        -------
        out: np.ndarray
            A 2D np.ndarray with the size of (bin_w, bin_h), \n
            where 1 represents empty and 0 represents occupied.
        """
        return self.bin.get_bin()

    def _get_bin_SU(self) -> float:
        """
        Return the space utilization of the `bin`.
        
        Returns
        -------
        out: float
            The space utilization of the `bin`.
        """
        return self.bin.get_bin_SU()

    def _get_next_item(self) -> Tuple[int, int]:
        """
        Return the next item size. 

        Returns
        -------
        item_size: Tuple[int, int]
            The size of the next item.
            Format: (item_width, item_height)
        """
        item = self.items_creator.preview(length=1)[0]
        item_width, item_height = item
        return (item_width, item_height)
    
    def _get_next_item_area(self) -> int:
        """
        Return the area of the next item.
        
        Returns
        -------
        item_area: int
            The area of the next item.
        """
        item_width, item_height = self._get_next_item()
        return (item_width * item_height)
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Return the observation of the environment.
        
        Returns
        -------
        obs: dict[str, np.ndarray]
            The observation of the environment. \n
            Format: {"mask": mask_obs, "bin": bin_obs}
        """
        bin_obs = self._get_bin_obs()
        mask_obs = self._get_mask_obs()
        obs = dict({MASK: mask_obs, BIN: bin_obs})
        return obs

    def _get_bin_obs(self) -> np.ndarray:
        """
        Return the observation of the `bin`.
        
        Returns
        -------
        bin_obs: np.ndarray
            A 3D np.ndarray with the size of (3, bin_w, bin_h), \n
            where the first channel represents the bin, \n
            the second channel represents the width of the next item, \n
            the third channel represents the height of the next item.
        """
        bin_channel = self.bin.get_bin()
        item_width, item_height = self._get_next_item()
        width_channel = np.full((self.bin_w, self.bin_h), item_width, dtype=np.uint8)
        height_channel = np.full((self.bin_w, self.bin_h), item_height, dtype=np.uint8)
        bin_obs = np.stack([bin_channel, width_channel, height_channel], axis=0)
        return bin_obs

    def _get_mask_obs(self) -> np.ndarray:
        """
        Return the observation of the action mask based on the current `bin` and the next item.

        Returns
        -------
        action_mask: np.ndarray
            A 1D np.ndarray with the size of (`action_dim`), \n
            where 1 represents valid action and 0 represents invalid action.
        """
        current_bin = self.bin.get_bin()
        item_width, item_height = self._get_next_item()
        action_mask = np.zeros(self.action_dim, dtype=np.uint8)  # default: invalid action
        x_positions = np.arange(self.bin_w - item_width + 1)
        y_positions = np.arange(self.bin_h - item_height + 1)
        for x in x_positions:
            for y in y_positions:
                # np.all(): AND operation for all elements in the array, if one element is False (0), return False.
                if np.all(current_bin[x:x+item_width, y:y+item_height] == 1):
                    action_mask[(self.bin_h * x) + y] = 1  # valid action
        return action_mask

    def _compute_compactness_reward(self, action: int) -> float:
        """
        Reward function based on the compactness of the cluster used in Deep-Pack.\n
        Paper: https://ieeexplore.ieee.org/document/8956393\n
        Source code: https://github.com/JeepWay/DeepPack\n
        """
        lx, ly = self.bin.index_to_location(action)
        state_np = (1 - self.bin.get_bin())
        labeled_array, num_features = label(state_np, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        cluster_sizes = np.bincount(labeled_array.ravel())
        item_label = labeled_array[lx][ly]
        cluster_size = cluster_sizes[item_label]
        cluster_indices = np.argwhere(labeled_array == item_label)
        top_left = cluster_indices.min(axis=0)
        bottom_right = cluster_indices.max(axis=0)
        bounding_box_size = (bottom_right[0] - top_left[0] + 1) * (bottom_right[1] - top_left[1] + 1)
        # compute compactness
        compactness = cluster_size / bounding_box_size
        reward = cluster_size * compactness
        return reward

    def bin2image(
        self, 
        state: np.ndarray, 
        x:int, 
        y:int, 
        item_w:int, 
        item_h:int
    ) -> plt.imshow:
        """
        Convert the bin state to an image for visualization. \n

        Parameters
        ----------
        state: np.ndarray
            The bin state.
        x: int
            The x position of the width of the item.
        y: int
            The y position of the height of the item.
        item_w: int
            The width of the item.
        item_h: int
            The height of the item.
        
        Returns
        -------
        image: matplotlib.image.AxesImage
            The image of the bin state.

        Examples
        --------
        >>> image = bin2image(state, x, y, item_w, item_h)
        >>> image.figure.savefig('result.png')
        >>> plt.close(image.figure)  
        """
        plt.figure(figsize=(8, 8))
        state_ = np.flipud(np.rot90(state, k=1))
        image = plt.imshow(state_, cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.title(f'{state.shape[0]}x{state.shape[1]} bin', fontsize=20)
        plt.xlabel('width', fontsize=16)
        plt.ylabel('height', fontsize=16)
        plt.scatter(-0.5+x, -0.5+y, s=150, color='red', marker='o')
        if self.bin_w == 5:
            plt.text(-0.5, 4.8, f'item:{item_w}x{item_h}', fontsize=18, color='red', ha='center', va='center')
            plt.xticks(np.arange(-0.5, state.shape[0]), np.arange(0, state.shape[0]+1), size=10)
            plt.yticks(np.arange(-0.5, state.shape[1]), np.arange(0, state.shape[1]+1), size=10)
        elif self.bin_w == 6:
            plt.text(-0.5, 10, f'item:{item_w}x{item_h}', fontsize=18, color='red', ha='center', va='center')
            plt.xticks(np.arange(-0.5, state.shape[0]), np.arange(0, state.shape[0]+1), size=10)
            plt.yticks(np.arange(-0.5, state.shape[1]), np.arange(0, state.shape[1]+1), size=10)
        elif self.bin_w == 12: 
            plt.text(-0.5, 21, f'item:{item_w}x{item_h}', fontsize=18, color='red', ha='center', va='center')
            plt.xticks(np.arange(-0.5, state.shape[0]), np.arange(0, state.shape[0]+1), size=8)
            plt.yticks(np.arange(-0.5, state.shape[1]), np.arange(0, state.shape[1]+1), size=8)
        elif self.bin_w == 23: 
            plt.text(-0.5, 38, f'item:{item_w}x{item_h}', fontsize=18, color='red', ha='center', va='center')
            plt.xticks(np.arange(-0.5, state.shape[0]), np.arange(0, state.shape[0]+1), size=8)
            plt.yticks(np.arange(-0.5, state.shape[1]), np.arange(0, state.shape[1]+1), size=8)
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=2)
        return image
    