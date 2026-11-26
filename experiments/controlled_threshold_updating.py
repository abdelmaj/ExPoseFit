"""Controlled causal extrema-updating ablation. Requires online_smoothing_loocv.py alongside this file."""
import argparse
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,sosfilt_zi
import online_smoothing_loocv as base

class Counter(base.OnlineRepCounter):
    def __init__(self,exercise,fps,adaptive):
        super().__init__(exercise,fps); self.adaptive=adaptive; self.updates=0; self.calibration_frame=None
    def _finish_calibration(self):
        ok=super()._finish_calibration()
        if ok:
            self.mn=min(self.calibration_angles); self.mx=max(self.calibration_angles)
        return ok
    def update(self,x,k):
        if not np.isfinite(x): return
        if not self.calibrated:
            self.calibration_angles.append(x)
            if k+1>=self.calibration_frames and self._finish_calibration(): self.calibration_frame=k
            return
        if self.adaptive and (x<self.mn or x>self.mx):
            self.mn=min(self.mn,x); self.mx=max(self.mx,x); self.motion_range=self.mx-self.mn
            self.down_th=self.mn+base.DOWN_RATIO[self.exercise]*self.motion_range
            self.up_th=self.mn+base.UP_RATIO[self.exercise]*self.motion_range
            self.updates+=1
        self._state_update(x)

def norm(e): return {'pushup':'push_up','push_up':'push_up','situp':'situp','sit_up':'situp','squat':'squat'}[str(e).lower().replace('-','_')]
def main(args):
    labels=pd.read_csv(args.labels); labels.exercise=labels.exercise.map(norm); labels.video_name=labels.video_name.str.removesuffix('.mp4')
    assert len(labels)==105 and labels.video_name.nunique()==105
    traces={}
    for p in sorted(Path(args.traces).rglob('*.csv')):
        _,e,v=p.stem.split('__',2); key=(norm(e),v.removesuffix('.mp4')); d=pd.read_csv(p).sort_values('frame_index').reset_index(drop=True)
        if key in traces: pd.testing.assert_frame_equal(traces[key],d)
        else: traces[key]=d
    assert set(traces)==set(zip(labels.exercise,labels.video_name))
    reference=pd.read_csv(args.reference); reference=reference[reference.method=='butterworth_1hz_o2_original'].set_index(['exercise','video'])
    rows=[]
    for r in labels.itertuples(index=False):
        e,v=r.exercise,r.video_name; d=traces[e,v]; assert np.array_equal(d.frame_index,np.arange(len(d)))
        raw=d.raw_angle_deg.to_numpy(float); valid=np.isfinite(raw); fps=float(r.source_fps); fps=fps if np.isfinite(fps) and fps>0 else 30.
        g=np.full(len(raw),np.nan); buf=deque(maxlen=base.SMOOTHING_BUFFER)
        for k,x in enumerate(raw):
            if np.isfinite(x): buf.append(x); g[k]=base.causal_smoothed_last(buf)
        if 'smoothed_angle_deg' in d: assert np.allclose(g,d.smoothed_angle_deg,atol=1e-6,rtol=0,equal_nan=True),v
        b=np.full(len(raw),np.nan); sos=butter(2,1,fs=fps,btype='lowpass',output='sos')
        if valid.any(): b[valid],_=sosfilt(sos,raw[valid],zi=sosfilt_zi(sos)*raw[valid][0])
        for name,s in [('gaussian',g),('butterworth_1hz',b)]:
            for adaptive in [False,True]:
                c=Counter(e,fps,adaptive)
                for k,x in enumerate(s): c.update(x,k)
                if not adaptive:
                    expected=r.online_reps if name=='gaussian' else reference.loc[(e,v),'predicted']
                    assert c.reps==expected,(e,v,name,c.reps,expected)
                rows.append(dict(exercise=e,video=v,ground_truth=r.ground_truth,filter=name,adaptive=adaptive,predicted=c.reps,calibrated=c.calibrated,calibration_frame=c.calibration_frame,threshold_updates=c.updates))
    f=pd.DataFrame(rows); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); f.to_csv(out/'updating_per_video.csv',index=False)
    rng=np.random.default_rng(42); results=[]
    for name,g in f.groupby('filter'):
        for e,h in list(g.groupby('exercise'))+[('overall',g)]:
            w=h.pivot(index=['video','ground_truth'],columns='adaptive',values='predicted'); y=w.index.get_level_values('ground_truth').to_numpy(); a=abs(w[False].to_numpy()-y); b=abs(w[True].to_numpy()-y); n=len(a); idx=rng.integers(n,size=(10000,n)); diff=a-b
            results.append(dict(filter=name,exercise=e,n=n,frozen_mae=a.mean(),adaptive_mae=b.mean(),frozen_rmse=np.sqrt((a*a).mean()),adaptive_rmse=np.sqrt((b*b).mean()),frozen_exact=(a==0).mean(),adaptive_exact=(b==0).mean(),frozen_within1=(a<=1).mean(),adaptive_within1=(b<=1).mean(),frozen_minus_adaptive=diff.mean(),ci_low=np.quantile(diff[idx].mean(1),.025),ci_high=np.quantile(diff[idx].mean(1),.975)))
    q=pd.DataFrame(results); q.to_csv(out/'updating_summary.csv',index=False); pass; pass
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--traces',required=True); p.add_argument('--labels',required=True); p.add_argument('--reference',required=True); p.add_argument('--out',required=True); main(p.parse_args())
