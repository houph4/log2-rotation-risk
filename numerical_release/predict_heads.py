"""Replay the supplied frozen visual heads from standardized confirmation features."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent;S=ROOT/'data/learning'
data=np.load(S/'planar_data_001/confirmation.npz');expected=np.load(S/'planar_evaluation_confirmation_001/predictions.npz');out={};checks={}
for name in expected.files:
    z=torch.load(S/'planar_fit_001'/name,map_location='cpu',weights_only=False);dim=len(z['center']);width=z['width']
    model=torch.nn.Sequential(torch.nn.Linear(z['input_dim'],width),torch.nn.ReLU(),torch.nn.Linear(width,width),torch.nn.ReLU(),torch.nn.Linear(width,dim))
    model.load_state_dict(z['state_dict']);model.eval()
    with torch.no_grad():pred=model(torch.tensor((data['features']-z['xmean'])/z['xstd'],dtype=torch.float32)).numpy()*z['scale']+z['center']
    err=float(np.max(abs(pred-expected[name])));assert err<2e-6,(name,err);out[name]=pred;checks[name]=err
path=ROOT/'results/learning';path.mkdir(parents=True,exist_ok=True)
np.savez_compressed(path/'replayed_predictions.npz',**out);(path/'PREDICTION_REPLAY.json').write_text(json.dumps(checks,indent=2));print('FROZEN_HEADS_REPLAYED',len(checks),'max abs difference',max(checks.values()))
