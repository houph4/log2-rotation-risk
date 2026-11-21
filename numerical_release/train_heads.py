"""Reproduce the controlled-head recipe from the supplied training features."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
def targets(d,kind,supervision):
    if kind=='Centered2':
        if supervision!='conditional_mean':raise ValueError('Centered covariance uses conditional-teacher supervision')
        return d['centered_mean']
    if kind=='Matrix9':return d['matrix_mean' if supervision=='conditional_mean' else 'matrix_gt']
    x=d['full_mean' if supervision=='conditional_mean' else 'full_gt']
    return x[:,:3 if kind=='Mean3' else 9]
def main(args):
    source=ROOT/'data/learning/planar_data_001';tr=dict(np.load(source/'train.npz'));dv=dict(np.load(source/'development.npz'))
    xm=tr['features'].mean(0);xs=np.maximum(tr['features'].std(0),1e-6)
    X=torch.tensor((tr['features']-xm)/xs,dtype=torch.float32);XD=torch.tensor((dv['features']-xm)/xs,dtype=torch.float32)
    Y=targets(tr,args.kind,args.supervision);YD=targets(dv,args.kind,args.supervision);center=Y.mean(0);scale=np.ones(Y.shape[1])
    blocks=[slice(0,Y.shape[1])] if args.kind in ['Matrix9','Mean3'] else [slice(0,3),slice(3,9)]
    for b in blocks:scale[b]=max(float(np.sqrt(np.mean(np.sum((Y[:,b]-center[b])**2,axis=1)))),1e-4)
    TY=torch.tensor((Y-center)/scale,dtype=torch.float32);TD=torch.tensor((YD-center)/scale,dtype=torch.float32)
    dest=ROOT/'results/retrained_heads';dest.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed);gen=torch.Generator().manual_seed(seed+1)
        model=torch.nn.Sequential(torch.nn.Linear(145,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,Y.shape[1]))
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf')
        for step in range(1,args.steps+1):
            ids=torch.randint(len(X),(512,),generator=gen);loss=(model(X[ids])-TY[ids]).square().sum(1).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%150==0 or step==args.steps:
                with torch.no_grad():val=float((model(XD)-TD).square().sum(1).mean())
                if val<best:
                    best=val;torch.save(dict(state_dict=model.state_dict(),input_dim=145,width=64,kind=args.kind,center=center,scale=scale,xmean=xm,xstd=xs,seed=seed,step=step),dest/f'{args.supervision}_{args.kind}_{seed}.pt')
        print(args.supervision,args.kind,seed,'best normalized development MSE',best,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['Matrix9','Mean3','Log2','Centered2'],default='Centered2');p.add_argument('--supervision',choices=['conditional_mean','single_GT'],default='conditional_mean');p.add_argument('--steps',type=int,default=3000);p.add_argument('--seeds',nargs='+',type=int,default=[640401,640402,640403]);main(p.parse_args())
