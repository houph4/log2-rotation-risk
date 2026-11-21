"""One metric definition for absolute, delta and acquisition-exclusion tables."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json, re, hashlib, argparse
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
SYNC=BASE.parent/'results_sync_20261024'
OUT=ROOT/'results/budget'
DS=['lmo','icbin','tudl','tless','ycbv']
METHODS={'Native confidence':'score_Native','Depth residual':'score_ICP_residual',
         'Matrix9':'score_Matrix9_restore','Log mean':'score_Mean3',
         'Diagonal':'score_Log2_diagonal','Log2':'score_Log2','FullPool':'score_FullPool'}
FRACTIONS=[0.,.1,.25,.5,.75,1.]
COLS=['sample_id','dataset','scene_id','im_id','cluster','rgb_path','gate_deg','pair_valid','eligible','visibility',
      'mssd_before','mssd_after','mspd_before_px','mspd_after_px']+list(METHODS.values())

def resolve_rgb(source,ds):
    p=Path(source)
    if p.exists():return p
    s=source.replace('\\','/');tail=s.split(f'/{ds}/')[-1]
    for root in [BASE/'data/bop_stage73',BASE/'data/bop_stage74',BASE/'tpami_full_stage77/data/public']:
        p=root/ds/tail
        if p.exists():return p
    raise FileNotFoundError(source)

def recall(x,thresholds):return (x[:,None]<=np.asarray(thresholds)).mean(1)*100

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=ROOT/'data/source_config.json'
    cfg=json.loads(config.read_text()) if config.exists() else None
    exposed=set(cfg['development_ids']) if cfg else {r['sample_id'] for r in json.loads((BASE/'tpami_full_stage77/results/dense_development/TEMPERATURE_LOCK.json').read_text())['case_diagnostics']}
    protocol={'version':'stage82','mspd_input':'raw official-toolkit pixels','mspd_normalization':'e640=e_raw*640/image_width',
              'surface_thresholds':(np.arange(1,11)*.05).tolist(),'projection_thresholds':list(range(5,51,5)),
              'threshold_operator':'<=','aggregation':'equal weight per target within each dataset',
              'fractions':FRACTIONS,'harm_tolerance_diameter':.001,'seed':821106,
              'ranking':'ascending score; sample-ID stable ties; recompute budget within each declared population',
              'interval':'2000 acquisition-group resamples of fixed observed acceptance allocation',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    rows=[];groups=[];splits=[];checks=[];inputs=[];widths={}
    for ds in DS:
        node='node2' if ds=='ycbv' else 'node1';frames=[]
        portable=ROOT/'data/budget'/(ds+'_inputs.csv.gz')
        if portable.exists():frames=[pd.read_csv(portable)];inputs.append(portable.relative_to(ROOT).as_posix())
        else:
            for p in sorted((SYNC/node/'results/metrics/shards'/ds).glob('[0-9][0-9][0-9][0-9]/cases.csv')):
                f=pd.read_csv(p,usecols=COLS);f=f[(f.gate_deg==29)&f.pair_valid&f.eligible&(f.visibility>=.1)]
                frames.append(f);inputs.append(str(p))
        f=pd.concat(frames,ignore_index=True)
        assert not f.sample_id.duplicated().any(),ds
        finite=np.isfinite(f[[*METHODS.values(),'mssd_before','mssd_after','mspd_before_px','mspd_after_px']].to_numpy(float)).all(1)
        f=f.loc[finite].sort_values('sample_id',kind='stable').reset_index(drop=True)
        scene_width={}
        for scene,g in f.groupby('scene_id',sort=True):
            if cfg:width=cfg['image_widths'][ds][str(int(scene))]
            else:
                p=resolve_rgb(g.rgb_path.iloc[0],ds)
                with Image.open(p) as im:width,height=im.size
            scene_width[int(scene)]=width
        scale=f.scene_id.map(scene_width).to_numpy(float)
        f['mspd_before_640']=f.mspd_before_px*640/scale;f['mspd_after_640']=f.mspd_after_px*640/scale
        widths[ds]=scene_width
        # Preserve raw and normalized deltas to identify the reported conflict.
        raw0=recall(f.mspd_before_px.to_numpy(),np.arange(5,51,5)).mean()
        norm0=recall(f.mspd_before_640.to_numpy(),np.arange(5,51,5)).mean()
        checks.append({'dataset':ds,'targets':len(f),'widths':str(sorted(set(scene_width.values()))),
                       'initial_AR_P_raw':raw0,'initial_AR_P_640':norm0})
        dev=f.sample_id.isin(exposed);devgroups=set(f.loc[dev,'cluster']);devscenes=set(f.loc[dev,'scene_id'])
        scopes={'all':f,'exclude_dev_targets':f[~dev],'exclude_dev_clusters':f[~f.cluster.isin(devgroups)],'exclude_dev_scenes':f[~f.scene_id.isin(devscenes)]}
        # A compact, self-contained input permits release without images or old CSV files.
        keep=['sample_id','dataset','scene_id','cluster','mssd_before','mssd_after','mspd_before_px','mspd_after_px','mspd_before_640','mspd_after_640']+list(METHODS.values())
        compact=f[keep].copy();compact['development_target']=dev;compact['development_group']=f.cluster.isin(devgroups)
        compact.to_csv(OUT/(ds+'_inputs.csv.gz'),index=False,compression='gzip')
        for scope,g in scopes.items():
            n=len(g);splits.append({'dataset':ds,'scope':scope,'targets':n,'clusters':g.cluster.nunique(),'scenes':g.scene_id.nunique()})
            if not n:continue
            am0=recall(g.mssd_before.to_numpy(),np.arange(1,11)*.05);am1=recall(g.mssd_after.to_numpy(),np.arange(1,11)*.05)
            ap0=recall(g.mspd_before_640.to_numpy(),np.arange(5,51,5));ap1=recall(g.mspd_after_640.to_numpy(),np.arange(5,51,5))
            delta=g.mssd_after.to_numpy()-g.mssd_before.to_numpy();harm=delta>.001;benefit=delta<-.001
            cl,cix=np.unique(g.cluster.to_numpy(str),return_inverse=True)
            for method,col in METHODS.items():
                order=np.argsort(g[col].to_numpy(float),kind='stable')
                for frac in FRACTIONS:
                    take=np.zeros(n,bool);take[order[:int(n*frac)]]=True;accepted=int(take.sum())
                    rec={'dataset':ds,'scope':scope,'method':method,'fraction':frac,'targets':n,'accepted':accepted,
                         'AR_M_initial':am0.mean(),'AR_M':(am0+(am1-am0)*take).mean(),
                         'AR_P_initial':ap0.mean(),'AR_P':(ap0+(ap1-ap0)*take).mean(),
                         'harm_count':int((take&harm).sum()),'harm_percent':100*(take&harm).sum()/accepted if accepted else np.nan,
                         'benefit_count':int((take&benefit).sum()),'benefit_recall':100*(take&benefit).sum()/benefit.sum() if benefit.sum() else np.nan,
                         'net_mssd_gain':-np.mean(delta*take)}
                    rec['delta_AR_M']=rec['AR_M']-rec['AR_M_initial'];rec['delta_AR_P']=rec['AR_P']-rec['AR_P_initial']
                    rows.append(rec)
                    for j,c in enumerate(cl):
                        ii=cix==j
                        groups.append({'dataset':ds,'scope':scope,'method':method,'fraction':frac,'cluster':c,'targets':int(ii.sum()),'accepted':int((ii&take).sum()),
                            'harm':int((ii&take&harm).sum()),'benefit':int((ii&take&benefit).sum()),'total_benefit':int((ii&benefit).sum()),
                            'gain_sum':float(-np.sum(delta[ii]*take[ii])),'AR_M_gain_sum':float(np.sum((am1-am0)[ii]*take[ii])),
                            'AR_P_gain_sum':float(np.sum((ap1-ap0)[ii]*take[ii]))})
        print('BUDGET',ds,len(f),'widths',sorted(set(scene_width.values())),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'metrics.csv',index=False)
    gc=pd.DataFrame(groups);gc.to_csv(OUT/'cluster_contributions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(splits).to_csv(OUT/'populations.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'normalization_check.csv',index=False)
    rng=np.random.default_rng(821106);intervals=[]
    for (ds,scope,frac),g in gc.groupby(['dataset','scope','fraction']):
        clusters=sorted(g.cluster.unique());n=len(clusters);boot=rng.multinomial(n,np.ones(n)/n,size=2000)
        for method in METHODS:
            a=g[g.method==method].set_index('cluster').loc[clusters]
            numer=boot@a.harm.to_numpy();denom=boot@a.accepted.to_numpy()
            h=np.divide(100*numer,denom,out=np.full(2000,np.nan),where=denom>0)
            intervals.append({'dataset':ds,'scope':scope,'fraction':frac,'method':method,
                 'harm_low':float(np.nanquantile(h,.025)) if frac else np.nan,'harm_high':float(np.nanquantile(h,.975)) if frac else np.nan})
    pd.DataFrame(intervals).to_csv(OUT/'intervals.csv',index=False)
    for (ds,scope),g in df.groupby(['dataset','scope']):
        q=g[g.fraction==1]
        assert q.AR_M.max()-q.AR_M.min()<1e-12
        assert q.AR_P.max()-q.AR_P.min()<1e-12
        z=g[g.fraction==0];assert (z.delta_AR_M==0).all() and (z.delta_AR_P==0).all()
    protocol['image_widths']=widths
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    (OUT/'INPUTS.json').write_text(json.dumps(inputs,indent=2),encoding='utf8')
    print(df[(df.scope=='all')&(df.fraction==.5)&(df.method=='Log2')][['dataset','AR_M','AR_P_initial','AR_P','delta_AR_P']].to_string(index=False),flush=True)

if __name__=='__main__':main()
