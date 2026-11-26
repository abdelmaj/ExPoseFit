"""Downsample cached raw angle traces; no pose inference. Requires previous two scripts."""
import argparse
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,sosfilt_zi
from controlled_threshold_updating import Counter,norm
import online_smoothing_loocv as base

def evaluate(raw,fps,exercise,method):
    s=np.full(len(raw),np.nan); valid=np.isfinite(raw)
    if method=='gaussian':
        buf=deque(maxlen=base.SMOOTHING_BUFFER)
        for k,x in enumerate(raw):
            if np.isfinite(x): buf.append(x); s[k]=base.causal_smoothed_last(buf)
    elif valid.any():
        sos=butter(2,1,fs=fps,btype='lowpass',output='sos')
        s[valid],_=sosfilt(sos,raw[valid],zi=sosfilt_zi(sos)*raw[valid][0])
    c=Counter(exercise,fps,False)
    for k,x in enumerate(s): c.update(x,k)
    return c

def main(a):
    labels=pd.read_csv(a.labels); labels.exercise=labels.exercise.map(norm); labels.video_name=labels.video_name.str.removesuffix('.mp4')
    assert len(labels)==105 and not labels.duplicated(['exercise','video_name']).any()
    traces={}
    for p in sorted(Path(a.traces).rglob('*.csv')):
        _,e,v=p.stem.split('__',2); key=(norm(e),v.removesuffix('.mp4')); d=pd.read_csv(p).sort_values('frame_index').reset_index(drop=True)
        if key in traces: pd.testing.assert_frame_equal(traces[key],d)
        else: traces[key]=d
    assert set(traces)==set(zip(labels.exercise,labels.video_name))
    ref=pd.read_csv(a.reference); ref=ref[ref.method=='butterworth_1hz_o2_original'].set_index(['exercise','video']); assert len(ref)==105 and ref.index.is_unique
    rows=[]
    for r in labels.itertuples(index=False):
        e,v=r.exercise,r.video_name; d=traces[e,v]; assert np.array_equal(d.frame_index,np.arange(len(d)))
        raw=d.raw_angle_deg.to_numpy(float); fps=float(r.source_fps); assert np.isfinite(fps) and fps>=15-1e-6
        assert ref.loc[(e,v),'ground_truth']==r.ground_truth
        for rate in ['native','15','10']:
            target=fps if rate=='native' else min(fps,float(rate))
            
            idx=np.arange(len(raw)) if target==fps else np.floor(np.arange(int(np.floor((len(raw)-1)*target/fps+1e-9))+1)*fps/target+1e-9).astype(int)
            assert len(np.unique(idx))==len(idx) and idx[-1]<len(raw)
            x=raw[idx]
            for method in ['gaussian','butterworth_1hz']:
                c=evaluate(x,target,e,method)
                if rate=='native':
                    expected=r.online_reps if method=='gaussian' else ref.loc[(e,v),'predicted']
                    assert c.reps==expected,(e,v,method,c.reps,expected)
                rows.append(dict(exercise=e,video=v,ground_truth=r.ground_truth,filter=method,rate=rate,source_fps=fps,target_fps=target,samples=len(x),valid_samples=np.isfinite(x).sum(),predicted=c.reps,calibrated=c.calibrated,calibration_seconds=(c.calibration_frame+1)/target if c.calibration_frame is not None else np.nan))
    f=pd.DataFrame(rows); f['absolute_error']=(f.predicted-f.ground_truth).abs(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); f.to_csv(out/'fps_per_video.csv',index=False)
    rng=np.random.default_rng(42); summary=[]; paired=[]
    for method,g in f.groupby('filter'):
        for e,h in list(g.groupby('exercise'))+[('overall',g)]:
            w=h.pivot(index='video',columns='rate',values='absolute_error'); n=len(w); ix=rng.integers(n,size=(10000,n))
            for rate in ['native','15','10']:
                q=h[h.rate==rate]; err=q.absolute_error.to_numpy(); summary.append(dict(filter=method,exercise=e,rate=rate,n=n,mae=err.mean(),rmse=np.sqrt((err**2).mean()),exact=(err==0).mean(),within1=(err<=1).mean(),calibrated=q.calibrated.sum()))
                if rate!='native':
                    diff=(w[rate]-w['native']).to_numpy(); ci=np.quantile(diff[ix].mean(1),[.025,.975]); paired.append(dict(filter=method,exercise=e,rate=rate,n=n,mae_increase=diff.mean(),ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(summary).to_csv(out/'fps_summary.csv',index=False); pd.DataFrame(paired).to_csv(out/'fps_paired.csv',index=False); None; None
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--traces',required=True); p.add_argument('--labels',required=True); p.add_argument('--reference',required=True); p.add_argument('--out',required=True); main(p.parse_args())
