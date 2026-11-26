import numpy as np,pandas as pd
from pathlib import Path
P=Path(__file__).parent
A=pd.read_csv(P/'full_angle_trajectories_105(1).csv'); G=pd.read_csv(P/'../data/repetition_counting_dataset.csv').set_index('video_name'); rng=np.random.default_rng(20261124)

rel=np.array([(a/100,b/100) for a in range(20,51,5) for b in range(55,86,5)])
absolute=np.array([(a,b) for a in range(5,180,5) for b in range(a+5,181,5)])
def count(x,ds,us,minimum):
 n=np.zeros(len(ds),int); down=np.zeros(len(ds),bool); last=np.full(len(ds),-999)
 if np.ptp(x)<minimum:return n
 for i,z in enumerate(x):
  hit=down & (z>us) & (i-last>=8); start=~down & (z<ds); n+=hit;last[hit]=i;down=(down|start)&~hit
 return n
rows=[]
for e,d in A.groupby('exercise'):
 vids=[]; ys=[]; rr=[]; ff=[]; oo=[]; minimum={'pushup':15,'squat':20,'situp':12}[e]; default=(.30,.70) if e=='situp' else (.35,.75)
 for v,t in d.groupby('video_name'):
  x=t.sort_values('retained_sample_index').smoothed_angle_deg.to_numpy(); assert np.isfinite(x).all(); lo=x.min(); r=np.ptp(x); vids.append(v);ys.append(G.loc[Path(v).stem,'ground_truth']);rr.append(count(x,lo+rel[:,0]*r,lo+rel[:,1]*r,minimum));ff.append(count(x,absolute[:,0],absolute[:,1],minimum));oo.append(count(x,np.array([lo+default[0]*r]),np.array([lo+default[1]*r]),minimum)[0])
 y=np.array(ys); rr=np.array(rr);ff=np.array(ff); er=abs(rr-y[:,None]);ef=abs(ff-y[:,None]);
 for i,v in enumerate(vids):
  j=np.argmin(er.sum(0)-er[i]);k=np.argmin(ef.sum(0)-ef[i]);rows.append([e,v,y[i],oo[i],rr[i,j],ff[i,k],*rel[j],*absolute[k]])
f=pd.DataFrame(rows,columns=['exercise','video','ground_truth','original_ratio_replay','adaptive_loo','fixed_loo','alpha','beta','down_deg','up_deg']);assert len(f)==105 and f.video.nunique()==105;f.to_csv(P/'threshold_cv_verified_per_video.csv',index=False)
summary=[]; comparisons=[]
for e,s in list(f.groupby('exercise'))+[('overall',f)]:
 idx=rng.integers(0,len(s),(10000,len(s))); y=s.ground_truth.to_numpy(); errors={}
 for m in ['original_ratio_replay','adaptive_loo','fixed_loo']:
  err=abs(s[m].to_numpy()-y);errors[m]=err
  for metric,values,root in [('MAE',err,False),('RMSE',err**2,True),('exact_pct',100*(err==0),False),('within1_pct',100*(err<=1),False)]:
   fun=np.sqrt if root else lambda x:x; ci=np.quantile(fun(values[idx].mean(1)),[.025,.975]);summary.append([e,len(s),m,metric,fun(values.mean()),*ci])
 diff=errors['fixed_loo']-errors['adaptive_loo']; comparisons.append([e,len(s),diff.mean(),*np.quantile(diff[idx].mean(1),[.025,.975])])
pd.DataFrame(summary,columns=['exercise','n','method','metric','estimate','ci_low','ci_high']).to_csv(P/'threshold_cv_verified_summary.csv',index=False)
pd.DataFrame(comparisons,columns=['exercise','n','fixed_minus_adaptive_mae','ci_low','ci_high']).to_csv(P/'threshold_cv_verified_paired.csv',index=False)
None;None
