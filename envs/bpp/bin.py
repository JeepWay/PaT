from typing import List, Tuple

import numpy as np  
import copy
from envs.bpp.item import Item

__all__ = ["Bin"]


class Bin(object):
    def __init__(
        self, 
        width: int = 5, 
        height: int = 5
    ):
        """
        Record the space information of the bin in the current episode.
        Bin object contains the following attributes:
        - bin_w: int, the width of the bin.
        - bin_h: int, the height of the bin.
        - bin_size: np.ndarray, the size of the bin in tuple format.
        - bin: np.ndarray, the bin with the size of (bin_w, bin_h), where 1 represents empty and 0 represents occupied.
        - items_list: list of Item objects, including successful and failed placements.
        
        Parameters
        ----------
        width: int (default=5)
            The width of the bin.
        height: int (default=5)
            The height of the bin.
        """
        self.bin_w = width
        self.bin_h = height
        self.bin_size = np.array([width, height])
        self.bin = np.ones(shape=(width, height), dtype=np.uint8)
        self.items_list: List[Item] = list()  

    def print_bin(self) -> None:
        """
        Print the `bin` to the console. 
        """
        print(self.bin)

    def get_bin(self) -> np.ndarray:
        """
        Return the `bin`.

        Returns
        -------
        out: np.ndarray
            A 2D np.ndarray with the size of (bin_w, bin_h).

        Examples
        --------
        >>> bin = Bin(width=5, height=5)
        >>> print(bin.get_bin())
        [[1 1 1 1 1]
         [1 1 1 1 1]
         [1 1 1 1 1]
         [1 1 1 1 1]
         [1 1 1 1 1]]
        """
        return copy.deepcopy(self.bin)

    def get_bin_SU(self) -> float:
        """
        Return the space utilization of the `bin`.
        
        Returns
        -------
        out: float
            The space utilization of the `bin`.
        """
        return ((1-self.bin).sum()) / (self.bin_w * self.bin_h)
    
    def get_items_list(self) -> List[Item]:
        """
        Return the `items_list`.

        Returns
        -------
        out: List[Item]
            A list of Item objects, where each object contains the information of a item.
        """
        return copy.deepcopy(self.items_list)
    
    def get_items_list_info(self) -> List[Tuple[int, int, int, int, bool]]:
        """
        Return the `items_list` information.

        Returns
        -------
        out: Dict[int, Tuple[int, int, int, int, bool]]
            A dictionary with the index as the key and the information of the item as the value. \n
            The information includes the width, height, lx, ly, and isplaced of the item.

        Examples
        --------
        >>> bin = Bin(width=5, height=5)
        >>> some operation to place items in the bin
        >>> print(bin.get_items_list_info())
        >>> {0: {'width': 1, 'height': 2, 'lx': 0, 'ly': 0, 'isplaced': True}, 
            1: {'width': 2, 'height': 3, 'lx': 1, 'ly': 0, 'isplaced': False}}
        """
        items_info = dict()
        for index, item in enumerate(self.items_list):
            items_info[index] = item.info
        return copy.deepcopy(items_info)
        
    def _update_bin_graph(self, item: Item) -> np.ndarray:
        """
        Update the `bin` with the item information.
        
        Parameters
        ----------
        item: Item
            The Item object to be placed in the `bin`.
        
        Returns
        -------
        out: np.ndarray
            The updated `bin` with new address.
            A 2D np.ndarray with the size of (bin_w, bin_h).

        Examples
        --------
        >>> bin = bin(width=5, height=5)
        >>> item = Item(1, 2, 0, 0)
        >>> print(bin._update_bin_graph(item))
        [[0 1 1 1 1]
         [0 1 1 1 1]
         [1 1 1 1 1]
         [1 1 1 1 1]
         [1 1 1 1 1]]
        """
        bin = copy.deepcopy(self.bin)
        bin[item.lx:(item.lx+item.width), item.ly:(item.ly+item.height)] = 0
        return bin

    def _check_item(self, item: Item) -> bool:
        """
        Check whether the item can be placed in the `bin`.
        
        Parameters
        ----------
        item: Item
            The Item object to be placed in the `bin`.
        
        Returns
        -------
        out: bool
            True if the item can be placed in the `bin`, otherwise False.
        
        Examples
        --------
        >>> bin = Bin(width=5, height=5)
        >>> item = item(width=5, height=2, lx=1, ly=0)
        >>> print(bin._check_item(item))
        False
        """
        if ((item.lx + item.width) > self.bin_w) or ((item.ly + item.height) > self.bin_h):
            return False
        elif (self.bin[item.lx:(item.lx + item.width), item.ly:(item.ly + item.height)].sum()) \
                    != (item.width * item.height):
            return False
        else:
            return True

    def index_to_location(self, index: int) -> Tuple[int, int]:
        """
        Convert the action index to the location of the item in the bin.
        
        Parameters
        ----------
        index: int
            The action index.
        
        Returns
        -------
        out: Tuple[int, int]
            The location of the item in the bin.
        """
        lx = index // self.bin_h
        ly = index % self.bin_h
        return (lx, ly)
    
    def location_to_index(self, lx:int, ly:int) -> int:
        """
        Convert the item location in the bin to the action index.
        
        Parameters
        ----------
        lx: int
            The x-coordinate of the wigth of the item.
        ly: int
            The y-coordinate of the height of the item.
            
        Returns
        -------
        out: int
            The action index.
        """
        index = (self.bin_h * lx) + ly
        return index

    def put_item(self, item_size: Tuple[int, int], index: int) -> bool:
        """
        Put a coming item in the bin according to the action index.
        Return True if the item is successfully placed, otherwise False.
        
        Parameters
        ----------
        item_size: tuple[int, int]
            The size of the item in tuple format.
            tuple: (item_width, item_height)
        index: int
            The action index.
        
        Returns
        -------
        out: bool
            True if the item is successfully placed, otherwise False.

        Examples
        --------
        >>> bin = Bin(width=6, height=10)
        >>> item_size = (1, 2)
        >>> action = 0
        >>> is_success = bin.put_item(item_size, action)
        >>> print(is_success)
        True
        """
        lx, ly = self.index_to_location(index)
        coming_item = Item(width=item_size[0], height=item_size[1], lx=lx, ly=ly)
        if self._check_item(coming_item) is True:
            coming_item.isplaced = True
            self.items_list.append(coming_item)
            self.bin = self._update_bin_graph(coming_item)
            return True
        else:
            coming_item.isplaced = False
            self.items_list.append(coming_item)
            return False

