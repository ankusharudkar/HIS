# %%
# laoding the mat annotation in python
import scipy.io
from functools import reduce, partial
import numpy as np

# %%
from collections import namedtuple

# type for annotation
# Annotation = namedtuple("Annotation", ["name", "mask", "children"])
Annotation = lambda name, mask, children: (name, mask, children)

# %%
def getAnnotationTree(mat) -> list:
    """Generate annotation tree from PASCAL mat file

    Args:
        mat (dict): loaded mat file dict of pascal parts

    Returns:
        list: [(name:str, mask:numpy.array), None|List]
    """
    
    ann = []
    root = None
    
    # annotation
    for elem in mat["anno"][0][0][1]:    
        # objects
        for nelem in elem:
            nest = []
            # nested parts
            for part in nelem[3]:
                part = part[0]
                nest.append(Annotation(part[0][0], part[1] == 1, []))
            
            if root is None:
                root = np.ones_like(nelem[2] == 1)
                # root[root>=1] = 1
            # else:
                # root = np.copy(nelem[2])
                
            ann.append(Annotation(nelem[0][0], nelem[2] == 1, nest))
    
    return Annotation("root", root, ann)

# %%
def printTree(tree, tab=0):
    print("\t"*tab, tree[0])
    for t in tree[2]:
        printTree(t, tab+1)

# %%
def generateAnnotationMask(annotation: tuple, attentionMask: np.ndarray, threshold: float=0.1) -> np.ndarray:
    
    attentionMaskSum = np.sum(attentionMask)
    annotationMask = np.zeros_like(attentionMask)
    
    def _inorder(ann):
        mask = attentionMask & ann[1]
        intersection = np.sum(mask)
        if (attentionMaskSum-intersection)/attentionMaskSum > threshold:
            nonlocal annotationMask
            annotationMask = annotationMask | mask
            return
        
        for child in ann[2]:
            _inorder(child)
            
    _inorder(annotation)
    
    return annotationMask

# %%
import cv2

def dictToAnnotation(data: dict):
    size = (data["height"], data["width"])
    image = data["Image"]
    
    def _genAnnotations(image):
        mask = np.zeros(size)
        # getting the polygon points
        vertices = []
        for i in range(0, len(image["mask"]), 2):
            vertices.append([
                round(image["mask"][i] * (size[1]-1)),
                round(image["mask"][i+1] * (size[0]-1))
            ])
        
        vertices = np.array([vertices], dtype=np.int32)

        cv2.fillPoly(mask, vertices, color=1)
        
        children = []
        for child in image["children"]:
            children.append(_genAnnotations(child))
            
        return Annotation(image["name"], mask == 1, children)
    
    return _genAnnotations(image)

# %%
import cv2

def dictToRNNAnnotation(data: dict):
    size = (data["height"], data["width"])
    image = data["Image"]
    
    queue = [(1, c)for c in [*image["children"]]]
    masks = []
    while queue:
        child = queue.pop(0)
        
        # process
        if len(masks) < child[0]:
            masks.append(np.zeros(size))
            
        vertices = []
        for i in range(0, len(child[1]["mask"]), 2):
            vertices.append([
                round(child[1]["mask"][i] * (size[1]-1)),
                round(child[1]["mask"][i+1] * (size[0]-1))
            ])
        
        vertices = np.array([vertices], dtype=np.int32)

        cv2.fillPoly(masks[-1], vertices, color=1)
        # end process
          
        # add children
        for c in child[1]['children']:
            queue.append((child[0]+1, c))
    
    return masks

