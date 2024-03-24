
# %%
import numpy as np
from typing import List, Tuple, Callable
import pickle
import matplotlib.pyplot as plt
import itertools

# %%
def find_seed(matrix: np.ndarray) -> Tuple[int, int]:
    """finds first non-zero entry in the matrix

    Args:
        matrix (np.ndarray): 2-D matrix of binary non overlapping regions

    Returns:
        Tuple[int, int]: index of first non-zero entry
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                return i, j

# %%
def flood(mask: np.ndarray, 
          seed_x: int,
          seed_y: int,
          value: int = 2) -> None:
    """flood fills the area for given seed

    Args:
        mask (np.ndarray): 2-D matrix with regions
        seed_x (int): x index of seed
        seed_y (int): y index of seed
        value (int, optional): fill value. Defaults to 2.
    """
    flood_area = mask[seed_x, seed_y]
    fill_pixels = [(seed_x, seed_y)]
    
    # track visited pixels
    filled = np.zeros_like(mask)
    filled[seed_x, seed_y] = 1
    
    def is_valid(i, j):
        return (i >= 0
                and j >= 0
                and i < mask.shape[0]
                and j < mask.shape[1])
    
    while fill_pixels:
        x, y = fill_pixels.pop()    
        mask[x, y] = value
        
        for dx, dy in itertools.product([0,1,-1], [0, 1, -1]):
            if (is_valid(x+dx, y+dy) 
                and mask[x+dx, y+dy] == flood_area 
                and filled[x+dx, y+dy] == 0):
                fill_pixels.append((x+dx, y+dy))
                filled[x+dx, y+dy] = 1
                    

# %%
def flood_fill(mask: np.ndarray) -> int:
    index = find_seed(mask)
    mask_counter = 2
    while index:
        flood(mask, index[0], index[1], mask_counter)
        index = find_seed(mask)
        mask_counter += 1
        
    return mask, mask_counter-2

# %%
def binary_mask_to_children(mask: np.ndarray) -> List[np.ndarray]:
    """Converts a compund binary mask to list of individual masks

    Args:
        mask (torch.tensor): binary mask of objects
    """
    
    mask, n_children = flood_fill(mask.copy())
        
    children = []
    for i in range(n_children):
        children.append(mask == i+2)
        
    return children

# %%
class Tree:
    def __init__(self, data, level=0):
        self.data = data
        self.level = level
        self.entropy = 1
        self.children: List[Tree] = []
        self.match = None
        
    def traverse_inorder(self, func: Callable=lambda x, l: print("  "*l, x.data)):
        func(self, self.level)
        for child in self.children:
            child.traverse_inorder(func)
            
    def _assign_r_entropy(self, entropy=1):
        self.entropy = (2*entropy)/(len(self.children)+2) if len(self.children) else entropy
        for child in self.children:
            child._assign_r_entropy(entropy/(len(self.children)+2))
            
    def assign_entropy(self):
        self._assign_r_entropy()
        def reciprocal(x, l):
            x.entropy = 1/x.entropy
            
        self.traverse_inorder(reciprocal)
        
    def total_entropy(self):
        total = 0
        def get_total(x, l):
            nonlocal total
            total += x.entropy
            
        self.traverse_inorder(get_total)
        return total
    
    def copy(self):
        root = Tree(self.data, self.level)
        
        for child in self.children:
            root.children.append(child.copy())
        
        return root
    
    def find_matching(self, masks: np.ndarray, threshold=0.9) -> "Tree":
        tree = self.copy()
        
        def matcher(x, l):
            for m in masks:
                if np.sum(x.data & m) / np.sum(x.data | m) >= threshold:
                    x.match = True
                    
        tree.traverse_inorder(matcher)
        return tree
    
    def _flat_list(self) -> List["Tree"]:
        children = []
        for child in self.children:
            if child.match == True:
                children.append(child)
            else:
                children.extend(child._flat_list())
                
        return children
        
        
    def prune(self, level=0):
        root = Tree(self.data, level)
        children = self._flat_list()
        
        for child in children:
            root.children.append(child.prune(level+1))
            
        return root
        
# %%
def is_contained(parent: np.ndarray, child: np.ndarray):
    return np.all((parent & child) == child)

# %%
def _generate_tree(root: Tree, mask_groups: dict):
    if root.level + 1 > len(mask_groups):
        return
    
    for mask in mask_groups[root.level+1]:
        if is_contained(root.data, mask):
            root.children.append(Tree(mask, root.level+1))
            
    for child in root.children:
        _generate_tree(child, mask_groups)

# %%
def generate_mask_tree(masks: List[np.ndarray]) -> Tree:
    root = Tree(np.ones(masks[0].shape, dtype=bool))
    mask_groups = {i+1: binary_mask_to_children(mask) for i, mask in enumerate(masks)}
    
    # generate mask hierarchy tree
    _generate_tree(root, mask_groups)
    
    return root

# %%
def get_mask_list(masks: List[np.ndarray]) -> List[np.ndarray]:
    mask_list = []
    for mask in masks:
        mask_list.extend(binary_mask_to_children(mask))
        
    return mask_list

# %%
def hierarchical_metrics(truth: List[np.ndarray], pred: List[np.ndarray]) -> dict:    
    
    # ground truth tree
    gt_tree = generate_mask_tree(truth)
    
    # prediction list
    pred_tree = generate_mask_tree(pred)
    preds = get_mask_list(pred)
    
    # get matching tree
    pruned_tree = gt_tree.find_matching(preds).prune()
    
    # assign entropy
    gt_tree.assign_entropy()
    pred_tree.assign_entropy()
    pruned_tree.assign_entropy()
    
    # compute entropy
    gt_e = gt_tree.total_entropy()
    pred_e = pred_tree.total_entropy()
    pruned_e = pruned_tree.total_entropy()
    
    # precision, recall , f1
    precision = pruned_e / pred_e
    recall = pruned_e / gt_e
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "hierarchical-precision": precision,
        "hierarchical-recall": recall,
        "hierarchical-f1": f1
    }