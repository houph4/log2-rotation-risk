"""Mesh and RGB-D input helpers for CPU ICP regeneration."""
from pathlib import Path
import os
import numpy as np
import cv2
from plyfile import PlyData
def mesh(path,scale):
    z=PlyData.read(path);v=np.column_stack([z['vertex'][k] for k in ['x','y','z']]).astype(float)*scale
    f=np.asarray(list(z['face']['vertex_indices']),dtype=np.int64)
    return np.ascontiguousarray(v),np.ascontiguousarray(f)
def crop(row):
    depth=cv2.imread(row['depth_path'],cv2.IMREAD_UNCHANGED);mask=cv2.imread(row['mask_path'],cv2.IMREAD_UNCHANGED)>0
    yy,xx=np.where(mask);side=max(np.ptp(xx)+1,np.ptp(yy)+1)*1.3;left=(xx.min()+xx.max())/2-side/2;top=(yy.min()+yy.max())/2-side/2
    coords=(np.arange(128)+.5)*side/128;mx,my=np.meshgrid((left+coords).astype(np.float32),(top+coords).astype(np.float32))
    observed=cv2.remap(depth.astype(np.float32),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)*row['depth_scale_to_m']
    valid=cv2.remap(mask.astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0;valid&=observed>0
    K=np.asarray(row['cam_K']).reshape(3,3).copy();K[0,0]*=128/side;K[1,1]*=128/side;K[0,2]=(K[0,2]-left)*128/side;K[1,2]=(K[1,2]-top)*128/side
    return observed.astype(float),valid,K
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0]))
        if p.name.startswith('scene_gt') or 'evaluation_private' in p.parts:raise RuntimeError('GT access in action generation')
