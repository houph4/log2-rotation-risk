"""Frozen head errors and explicit mean/covariance risk contributions."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,itertools
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as Rot
from geometry import point_losses

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;S=BASE/'ambiguity_stage64'
if (ROOT/'data/learning').exists():S=ROOT/'data/learning'
def tensor(c):
    out=np.zeros(c.shape[:-1]+(3,3))
    for j,(a,b) in enumerate(itertools.combinations_with_replacement(range(3),2)):
        out[...,a,b]=out[...,b,a]=c[...,j]*np.pi**2/(1 if a==b else np.sqrt(2))
    return out
def metric_project(Z,cap):
    e,U=np.linalg.eigh((Z+Z.swapaxes(-1,-2))/2);v=np.maximum(e,0);s=np.sort(e,axis=-1)[...,::-1]
    t=(np.cumsum(s,axis=-1)-cap)/np.arange(1,4);rho=np.maximum((s-t>0).sum(-1)-1,0)
    theta=np.take_along_axis(t,rho[...,None],axis=-1)[...,0];theta=np.where(v.sum(-1)>cap,theta,0.)
    return (U*np.maximum(e-theta[...,None],0)[...,None,:])@U.swapaxes(-1,-2)
OUT=ROOT/'results/learning';OUT.mkdir(parents=True,exist_ok=True)
data=dict(np.load(S/'planar_data_001/confirmation.npz'));contexts=json.loads((S/'planar_data_001/CONTEXTS.json').read_text())
regime=np.asarray([contexts[int(v)]['regime'] for v in data['context']]);m=data['full_mean'][:,:3]*np.pi;second=tensor(data['full_mean'][:,3:9]);cov=second-m[:,:,None]*m[:,None,:]
pred=np.load(S/'planar_evaluation_confirmation_001/predictions.npz');vec=np.load(S/'planar_evaluation_confirmation_001/vectors.npz')
print('VECTORS',vec.files[:8],flush=True)
rows=[];scales=[];risks=[]
def cap(a):return a*np.minimum(1,2*np.deg2rad(29)/np.maximum(np.linalg.norm(a,axis=1),1e-300))[:,None]
def expected_risk(a):
    risk=np.zeros(len(a))
    for ci,c in enumerate(contexts):
        ids=np.flatnonzero(data['context']==ci)
        x=np.load(S/'planar_data_001'/Path(c['bank_path']).name)['x']
        risk[ids]=np.sum(point_losses(a[ids],x)*data['posterior'][ids],axis=1)
    return risk
for name in pred.files:
    p=pred[name];sup,rest=name.split('_',1);kind=name.split('_')[-2];seed=int(name.split('_')[-1].split('.')[0])
    supervision='single_GT' if name.startswith('single_GT') else 'conditional_mean'
    checkpoint=torch.load(S/'planar_fit_001'/name,map_location='cpu',weights_only=False)
    scales.append({'model':name,'center':np.asarray(checkpoint['center']).tolist(),'scale':np.asarray(checkpoint['scale']).tolist(),'checkpoint_step':int(checkpoint['step'])})
    if kind=='Matrix9':
        U,_,VH=np.linalg.svd(p.reshape(-1,3,3)*np.sqrt(3));D=np.broadcast_to(np.eye(3),(len(p),3,3)).copy();D[:,2,2]=np.linalg.det(U@VH)
        action=cap(Rot.from_matrix(U@D@VH).as_rotvec());rr=expected_risk(action)
        for reg in ['orthographic','perspective']:
            ix=regime==reg;risks.append({'model':name,'supervision':supervision,'kind':kind,'seed':seed,'regime':reg,'mean_risk_deg2':np.nan,'decoded_risk_deg2':float(rr[ix].mean()*(180/np.pi)**2),'same_mean_gain_deg2':np.nan})
        continue
    mh=p[:,:3]*np.pi
    if kind=='Mean3':pc=None;sh=None
    else:
        block=tensor(p[:,3:9])
        if kind=='Centered2':pc=metric_project(block,np.deg2rad(29)**2);sh=metric_project(pc+mh[:,:,None]*mh[:,None,:],np.deg2rad(29)**2)
        else:sh=metric_project(block,np.deg2rad(29)**2);pc=metric_project(sh-mh[:,:,None]*mh[:,None,:],np.deg2rad(29)**2)
    rm=expected_risk(cap(mh))
    if pc is None:ra=rm.copy()
    else:
        BB=np.eye(3)-(np.trace(sh,axis1=1,axis2=2)[:,None,None]*np.eye(3)-sh)/12
        if kind=='Centered2':BB=np.eye(3)-(np.trace(pc,axis1=1,axis2=2)[:,None,None]*np.eye(3)-pc)/12
        ra=expected_risk(cap(np.linalg.solve(BB,mh[...,None])[...,0]))
    for reg in ['orthographic','perspective']:
        ix=regime==reg;em=np.linalg.norm(mh[ix]-m[ix],axis=1)
        risks.append({'model':name,'supervision':supervision,'kind':kind,'seed':seed,'regime':reg,
            'mean_risk_deg2':float(rm[ix].mean()*(180/np.pi)**2),'decoded_risk_deg2':float(ra[ix].mean()*(180/np.pi)**2),'same_mean_gain_deg2':float((rm[ix]-ra[ix]).mean()*(180/np.pi)**2)})
        rows.append({'model':name,'supervision':supervision,'kind':kind,'seed':seed,'regime':reg,'targets':int(ix.sum()),
            'mean_RMSE_rad':float(np.sqrt(np.mean(em**2))),
            'second_RMSE_rad2':float(np.sqrt(np.mean(np.sum((sh[ix]-second[ix])**2,axis=(1,2))))) if sh is not None else np.nan,
            'covariance_RMSE_rad2':float(np.sqrt(np.mean(np.sum((pc[ix]-cov[ix])**2,axis=(1,2))))) if pc is not None else np.nan})
pd.DataFrame(rows).to_csv(OUT/'head_errors.csv',index=False)
pd.DataFrame(risks).to_csv(OUT/'head_risks.csv',index=False)
pd.DataFrame(risks).groupby(['supervision','kind','regime']).mean(numeric_only=True).reset_index().to_csv(OUT/'head_risks_summary.csv',index=False)
summary=pd.DataFrame(rows).groupby(['supervision','kind','regime']).mean(numeric_only=True).reset_index();summary.to_csv(OUT/'head_errors_summary.csv',index=False)
(OUT/'target_normalization.json').write_text(json.dumps(scales,indent=2))
(OUT/'TRAINING_PROTOCOL.txt').write_text('''Physical target blocks are m/pi and svec(S)/pi^2, or m/pi and svec(Sigma)/pi^2. Matrix9 target is vec(R)/sqrt(3).
Each model centres targets using its own training mean c. For each block B, s_B=max(sqrt(E_train ||y_B-c_B||^2),1e-4). All coordinates in a block share this scale.
The loss is E_batch sum_B ||(yhat_B-y_B)/s_B||^2. Thus every normalized block has coefficient one; no division by output dimension is applied.
Matrix9 has one nine-coordinate block. Mean3 has one three-coordinate block. Log2 and Centered2 have blocks of widths three and six.
The svec off-diagonal entries use sqrt(2); the Euclidean second-block norm equals the Frobenius matrix norm.
Inputs are centred/scaled coordinate-wise using training mean and standard deviation, floor1e-6. Architecture145->64->64->output, AdamW lr0.001, weight decay0.0001, batch512, 3000 steps, seeds640401/640402/640403.
Own normalized development-target MSE is checked every150 steps. The lowest such value chooses each checkpoint.
For conditional-teacher heads, targets are exact conditional moments under the known pixel model. SingleGT uses x and xx^T targets from the sampled latent pose. Centered2 uses conditional covariance supervision.
PSD/trace repair projects onto {Z >=0, tr Z <=r^2}; raw and centered output actions use the corresponding repaired blocks. Mean errors above are computed before action clipping and reported in physical radians.
''',encoding='utf8')
print(summary.to_string(index=False),flush=True)
