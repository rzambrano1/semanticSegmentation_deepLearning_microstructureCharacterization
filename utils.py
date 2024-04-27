#!/usr/bin/python3

#################
# Session Setup #
#################

# Standard Libraries
import argparse
import os
import sys
from tqdm import tqdm

# Python Modules
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple
from collections import namedtuple
import gc

# Type Hint Libraries
from typing import Optional, Tuple, Union, TypeVar, List, Type, TypeVar, Generic
import numpy.typing as npt
import matplotlib.figure
from torch import Tensor

# Math and Data Science Libraries
import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve

# Plot Libraries
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# Machine Learning Libraries
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image Libraries
import cv2 

import skimage as ski
from skimage import io
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb

###############################
# Data Structures and Classes #
###############################

A = TypeVar('A')
I = TypeVar('I')

@dataclass(frozen=True)
class annotated_polygon:
    center: npt.NDArray
    nodes: List[npt.NDArray]
        
    def __str__(self):
        return f'Polygon:\ncenter -> {self.center}\nnodes  -> {self.nodes}'
    
    def __repr__(self):
        return f'Polygon:\ncenter -> {self.center}\nnodes  -> {self.nodes}'
    
    def __getitem__(self,position):
        return self.nodes[position]
    
    def __len__(self):
        return len(self.nodes)
    
    def get_center(self):
        return self.center
    
    def get_nodes(self):
        return self.nodes
    

@dataclass
class img_mask:
    img_name: str
    polygons: List[annotated_polygon]
        
    def __str__(self):
        return f'img_mask:\nimg_name -> {self.img_name}\npolygons -> {self.num_polygons()}'
    
    def __repr__(self):
        return f'img_mask:\nimg_name -> {self.img_name}\npolygons -> {self.num_polygons()}'
    
    def num_polygons(self):
        return len(self.polygons)
    
    def add_polygon(self,new_polygon):
        self.polygons.append(new_polygon)
        
    def __getitem__(self,position):
        return self.polygons[position]
    
    def __len__(self):
        return len(self.polygons)
    
    def get_polygons(self):
        return self.polygons
    
    def get_name(self):
        return self.img_name
    
####################
# Helper Functions #
####################

def parse_line(line_item: str)->list:
    """
    Helper function to process a line with a poligon in the csv file with annotated polygons
    """
    line = line_item
    polygon_points = []
    
    line_items = line.split(';')
    line_items[1] = np.array([float(x) for x in line_items[1][1:-1].split(',')])
    
    line_items[2] = line_items[2].strip()[1:-1].split('),')
    line_items[2] = line_items[2][0:-1]
    for item in line_items[2]:
        remove_space = item.replace(' ','')
        remove_parenthesis = item.replace('(','')
        points_str = remove_parenthesis.split(',')
        points_float = [float(x) for x in points_str]
        polygon_points.append(np.array(points_float))
        
    line_items[2] = polygon_points
    
    return line_items

def lines_to_polygons(line:str)->Tuple[str,Type[annotated_polygon]]:
    """
    Helper function to build a polygon out of a line with data.
    """
    parsed_line = parse_line(line)
    image_name = parsed_line[0]
    polygon_struc = annotated_polygon(parsed_line[1],parsed_line[2])
    
    return image_name, polygon_struc     

def build_mask_dataset(document_lines: List[str])->List[Type[img_mask]]:
    """
    Helper function to build a dataset of img_mask objects based on a parsed .csv file with annotated coordinates
    associated to a given image.
    """
    
    masks = []
    
    for line in document_lines[1:]: # Omits first line which is a comment
        im_name, polygon_structure = lines_to_polygons(line)
        im_name = im_name.replace('.png','')
        try:
            expr1 = 'isinstance({}, img_mask)'.format(im_name)
            instance_test = eval(expr1)
        except:
            instance_test = False
        if instance_test:
            eval('{}.add_polygon(polygon_structure)'.format(im_name))
        else:
            expr2 = '{} = img_mask("{}",[polygon_structure])'.format(im_name,im_name)
            exec(expr2)
            eval('masks.append({})'.format(im_name))
    
    return masks

def mask_in_set(name:str,dataset:List[Type[img_mask]]):
    """
    Helper function to retrieve a mask by name. returns None if not in dataset.
    """
    for mask in dataset:
        if name == mask.get_name():
            return mask
    print('Mask not found in dataset...')
    return None

def nodes_to_points(nodes:List[npt.NDArray])->Tuple[npt.NDArray,npt.NDArray]:
    """
    Helper function to turn nodes of a polygon into points easy to plot in matplotlib.
    """
    x = []
    y = []
    
    for coord in nodes:
        x.append(coord[0])
        y.append(coord[1])
        
    return np.array(x),np.array(y)

def plot_image_and_mask(img: npt.NDArray, mask:Type[img_mask], plotCenters: Optional[bool]=False)->matplotlib.figure:
    """
    Helper function to plot image and all the polygons in its mask
    """
    if len(img.shape) != 2:
        try:
            curr_img = rgb2gray(img)
        except:
            raise ValueError('Please provide either an RGB image or a grayscale image')
    else:
        curr_img = img
        
    polygons_in_mask = mask.get_polygons()
    
    fig, ax = plt.subplots()
    
    ax.imshow(curr_img, cmap=plt.cm.gray)
    
    for curr_poly in polygons_in_mask:
        if plotCenters:
            # Get the center
            x = curr_poly.get_center()[0]
            y = curr_poly.get_center()[1]
        # Get polygon nodes as coordinates
        x_node_coords,y_node_coords =  nodes_to_points(curr_poly.get_nodes())
        # Plotting the polygon
        ax.fill(x_node_coords,y_node_coords,c='lime',alpha=0.4)
        if plotCenters:
            # Plotting polygon center
            ax.scatter(x,y,c='r',marker='*')
    
    return ax

def create_binary_mask(mask:Type[img_mask])->matplotlib.figure:
    """
    Helper function to plot the image's binary mask
    """
        
    polygons_in_mask = mask.get_polygons()
    
    my_dpi = 96
    fig, ax = plt.subplots(figsize=(1322/my_dpi, 998/my_dpi), dpi=my_dpi)
    
    ax.set_facecolor('black')
    
    for curr_poly in polygons_in_mask:
        # Get polygon nodes as coordinates
        x_node_coords,y_node_coords =  nodes_to_points(curr_poly.get_nodes())
        # Plotting the polygon
        ax.fill(x_node_coords,-y_node_coords,c='white',alpha=1)
    
    # Set the range of x-axis to match figure size
    plt.xlim(0, 1024)
    plt.ylim(-769,0)
    
   # Removing frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Turning off ticks and labels
    ax.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False) 
    
    plt.close()
    
    return fig

