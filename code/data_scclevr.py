'''
Generate scientific CLEVR dataset (still to be completed).
Clone git repo https://gitlab.lrz.de/scocl/scclevr and install package via pip install -e .
Output of these classes are not optimal for our syntax that's why we need to modify them here quickly.

Sara Aumiller
Nov 2023
'''

import scclevr
import torch
import numpy as np

def makeRings(N_img=1000, N_obj=2, device='cpu'):
    '''
    What to adjust from the repo:
    - reshape arrays
    - convert np arrays into pytorch tensors
    - create masks instead of images of individual objects -> values between 0 and 1 between shared pixels
    
    Paramter:
    - N_img: number of images to generate
    - N_obj: number of object per image
    ------------------------------------------------------------------------
    returns:
    - event_images: (N_img, 1, 31, 31)
    - object_images: (N_img, N_obj, 31, 31)
    - n_objects: (N_img) e.g. array of only 2 if every image has two rings
    - object_features: (N_img, N_obj, 3), 3 for x,y, radius
    '''
    
    
    rings = scclevr.RingsBinaryUniform(N_obj) # two rings per imagne
    event_images, object_images, n_objects, object_features =  rings.gen_events(N_img)
    
    # convert object_images into attention masks
    object_images = _att_masks(event_images, object_images, N_obj)
    # change dimentions to be using in SlotAttention modules
    object_images, event_images = _change_dims(event_images, object_images)
    # convert to Pytorch Tensors
    return _convert_into_pytorch_tensors(event_images, object_images, n_objects, object_features, device)
    

def _att_masks(event_images, object_images, n_objects):
    '''
    n_obejcts: int
    --------------
    returns attention masks, fraction of hits per pixel over slots
    '''
    multi_hits = np.sum(object_images, axis=1) - event_images
    i_multi_hits = np.where(multi_hits!=0)
    val_multi_hits = multi_hits[i_multi_hits]
    val_multi_hits_replace = 1/(val_multi_hits+1)
    for i in range(n_objects):
        object_images[i_multi_hits[0], 0, i_multi_hits[1], i_multi_hits[2]] = np.expand_dims(val_multi_hits_replace, axis=(1))
        object_images[i_multi_hits[0], 1, i_multi_hits[1], i_multi_hits[2]] = np.expand_dims(val_multi_hits_replace, axis=(1))
    
    return object_images
    
    
def _change_dims(event_images, object_images):
    '''
    input shape:
    event_images.shape = (N, 31, 31, 1)
    object_images.shape = (N, 2, 31, 31, 1)
    '''
    object_images = np.squeeze(object_images, axis=4) # (N, 2, 31, 31)
    event_images = np.expand_dims(np.squeeze(event_images, axis=3), axis=1) # (N, 1, 31, 31)
    return object_images, event_images    
    
    
def _convert_into_pytorch_tensors(event_images, object_images, n_objects, object_features, device):
    return torch.FloatTensor(event_images).to(device), \
               torch.FloatTensor(object_images).to(device), \
               torch.FloatTensor(n_objects).to(device), \
               torch.FloatTensor(object_features).to(device)
   