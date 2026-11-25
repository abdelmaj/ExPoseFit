# Run with Python or in Colab after mounting Drive.
import numpy as np, pandas as pd
from pathlib import Path
from collections import deque
from scipy.ndimage import gaussian_filter
import argparse
CALIBRATION_SECONDS = 3.0
SMOOTHING_BUFFER = 31
MIN_FRAMES_BETWEEN_REPS = 8
DOWN_RATIO = {'push_up':0.35,'squat':0.35,'situp':0.30}
UP_RATIO = {'push_up':0.75,'squat':0.75,'situp':0.70}
MIN_MOTION_RANGE = {'push_up':15.0,'squat':20.0,'situp':12.0}

def gaussian_smoothing(data, sigma=3):
    return gaussian_filter(data, sigma=sigma)

def causal_smoothed_last(raw_buffer):
    if len(raw_buffer)==0: return np.nan
    vals=list(raw_buffer)
    if len(vals)<3: return float(vals[-1])
    try:
        sm=gaussian_smoothing(vals)
        return float(sm[-1]) if len(sm) else float(vals[-1])
    except Exception:
        return float(vals[-1])


class OnlineRepCounter:
    def __init__(self, exercise, fps):
        self.exercise = exercise
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.calibration_frames = max(1, int(round(CALIBRATION_SECONDS*self.fps)))
        self.raw_angles = deque(maxlen=SMOOTHING_BUFFER)
        self.calibration_angles = []
        self.calibrated = False
        self.calibration_status = "collecting"
        self.down_th = np.nan
        self.up_th = np.nan
        self.motion_range = np.nan
        self.state = "UP"
        self.rep_bottom = float("inf")
        self.frames_since_last_rep = 999
        self.reps = 0
        self.valid_angle_frames = 0
        self.invalid_angle_frames = 0

    def _state_update(self, a):
        if not np.isfinite(a):
            return False

        self.frames_since_last_rep += 1
        rep_completed = False

        if self.state == "UP":
            if a < self.down_th:
                self.state = "DOWN"
                self.rep_bottom = a

        elif self.state == "DOWN":
            self.rep_bottom = min(self.rep_bottom, a)

            if a > self.up_th and self.frames_since_last_rep >= MIN_FRAMES_BETWEEN_REPS:
                if self.rep_bottom < self.down_th:
                    self.reps += 1
                    rep_completed = True

                self.state = "UP"
                self.rep_bottom = float("inf")
                self.frames_since_last_rep = 0

        return rep_completed

    def _finish_calibration(self):
        valid = np.asarray([a for a in self.calibration_angles if np.isfinite(a)], dtype=float)

        if len(valid) == 0:
            self.calibration_status = "no_valid_angles"
            return False

        mn = float(valid.min())
        mx = float(valid.max())
        self.motion_range = mx - mn

        if self.motion_range < MIN_MOTION_RANGE[self.exercise]:
            self.calibration_status = "insufficient_motion"
            return False

        self.down_th = mn + DOWN_RATIO[self.exercise]*self.motion_range
        self.up_th   = mn + UP_RATIO[self.exercise]*self.motion_range

        self.calibrated = True
        self.calibration_status = "ok"

        # replay once so calibration-period reps are not lost
        for a in self.calibration_angles:
            self._state_update(a)

        return True

    def update(self, raw_angle, frame_index):
        if not np.isfinite(raw_angle):
            self.invalid_angle_frames += 1
            return np.nan, False

        self.valid_angle_frames += 1
        self.raw_angles.append(float(raw_angle))
        smoothed = causal_smoothed_last(self.raw_angles)

        if not self.calibrated:
            self.calibration_angles.append(smoothed)
            if frame_index + 1 >= self.calibration_frames:
                self._finish_calibration()
            return smoothed, False

        return smoothed, self._state_update(smoothed)


from scipy.signal import butter, sosfilt, sosfilt_zi

def main(trace_dir, labels_csv, out):
    global causal_smoothed_last
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv(labels_csv)
    def norm(x):
        return {'pushup':'push_up','push_up':'push_up','situp':'situp','sit_up':'situp','squat':'squat'}[str(x).lower().replace('-','_')]
    labels['exercise']=labels.exercise.map(norm)
    labels['video_name']=labels.video_name.str.strip().str.replace(r"\.mp4$", "", regex=True)
    traces={}
    for p in sorted(Path(trace_dir).rglob('*.csv')):
        _,ex,v=p.stem.split('__',2); key=(norm(ex),v.removesuffix('.mp4'))
        d=pd.read_csv(p).sort_values('frame_index').reset_index(drop=True)
        if key in traces: pd.testing.assert_frame_equal(traces[key],d,check_exact=True)
        else: traces[key]=d
    assert len(labels)==105 and not labels.duplicated(['exercise','video_name']).any()
    assert set(zip(labels.exercise,labels.video_name))==set(traces), 'Trace/label mismatch'
    assert labels.video_name.str[:11].nunique()==105, 'Source IDs not unique'
    grid=[(a/100,b/100) for a in range(20,51,5) for b in range(55,86,5)]
    methods=['raw','gaussian','butterworth_1hz_o2','butterworth_2hz_o2']
    defaults={e:(DOWN_RATIO[e],UP_RATIO[e]) for e in DOWN_RATIO}
    gaussian=causal_smoothed_last
    rows=[]
    for e,g in labels.groupby('exercise'):
        g=g.sort_values('video_name').reset_index(drop=True)
        pred=np.zeros((len(g),len(methods),len(grid)),int); base=np.zeros((len(g),len(methods)),int)
        for i,r in enumerate(g.itertuples(index=False)):
            d=traces[e,r.video_name]; assert np.array_equal(d.frame_index,np.arange(len(d)))
            raw=d.raw_angle_deg.to_numpy(float); valid=np.isfinite(raw)
            fps=float(r.source_fps); fps=fps if np.isfinite(fps) and fps>0 else 30.
            buf=deque(maxlen=SMOOTHING_BUFFER); sm=np.full(len(raw),np.nan)
            for k,x in enumerate(raw):
                if np.isfinite(x): buf.append(x); sm[k]=gaussian(buf)
            if 'smoothed_angle_deg' in d:
                assert np.allclose(sm,d.smoothed_angle_deg.to_numpy(float),atol=1e-6,rtol=0,equal_nan=True),r.video_name
            signals=[raw,sm]
            for cutoff in [1.,2.]:
                assert cutoff<fps/2
                sos=butter(2,cutoff,fs=fps,btype='lowpass',output='sos'); f=np.full(len(raw),np.nan)
                if valid.any(): f[valid],_=sosfilt(sos,raw[valid],zi=sosfilt_zi(sos)*raw[valid][0])
                signals.append(f)
            def run(signal,a,b):
                DOWN_RATIO[e]=a; UP_RATIO[e]=b
                c=OnlineRepCounter(e,fps)
                for k,x in enumerate(signal): c.update(float(x),k)
                return c.reps
            causal_smoothed_last=lambda buf:float(buf[-1])
            try:
                for m,signal in enumerate(signals):
                    base[i,m]=run(signal,*defaults[e]); pred[i,m]=[run(signal,a,b) for a,b in grid]
            finally: causal_smoothed_last=gaussian
            assert base[i,1]==r.online_reps,(e,r.video_name,base[i,1],r.online_reps)
        y=g.ground_truth.to_numpy(int); err=np.abs(pred-y[:,None,None]); total=err.sum(0)
        for i,v in enumerate(g.video_name):
            train=total-err[i]
            for m,name in enumerate(methods):
                j=int(np.argmin(train[m])); rows.append([e,v,y[i],name+'_original',base[i,m],*defaults[e],name]); rows.append([e,v,y[i],name+'_loo',pred[i,m,j],*grid[j],name])
            m,j=np.unravel_index(np.argmin(train[2:]),train[2:].shape); m+=2
            rows.append([e,v,y[i],'butterworth_selected_loo',pred[i,m,j],*grid[j],methods[m]])
        DOWN_RATIO[e],UP_RATIO[e]=defaults[e]
    f=pd.DataFrame(rows,columns=['exercise','video','ground_truth','method','predicted','alpha','beta','selected_filter'])
    f.to_csv(out/'smoothing_per_video.csv',index=False)
    rng=np.random.default_rng(20261124); summary=[]; paired=[]
    for e,g in list(f.groupby('exercise'))+[('overall',f)]:
        wide=g.pivot(index=['exercise','video','ground_truth'],columns='method',values='predicted'); y=wide.index.get_level_values('ground_truth').to_numpy(); n=len(y); idx=rng.integers(0,n,(10000,n)); errors={}
        for m in wide:
            z=np.abs(wide[m].to_numpy()-y); errors[m]=z
            for metric,vals,fn in [('MAE',z,lambda x:x),('RMSE',z*z,np.sqrt),('exact_pct',100*(z==0),lambda x:x),('within1_pct',100*(z<=1),lambda x:x)]:
                summary.append([e,n,m,metric,fn(vals.mean()),*np.quantile(fn(vals[idx].mean(1)),[.025,.975])])
        for m in ['raw_loo','butterworth_1hz_o2_loo','butterworth_2hz_o2_loo','butterworth_selected_loo']:
            diff=errors['gaussian_loo']-errors[m]; paired.append([e,n,m,diff.mean(),*np.quantile(diff[idx].mean(1),[.025,.975])])
    q=pd.DataFrame(summary,columns=['exercise','n','method','metric','estimate','ci_low','ci_high']); q.to_csv(out/'smoothing_summary.csv',index=False)
    pd.DataFrame(paired,columns=['exercise','n','method','gaussian_loo_minus_method','ci_low','ci_high']).to_csv(out/'smoothing_paired.csv',index=False)
    print(q[(q.exercise=='overall')&(q.metric=='MAE')].to_string(index=False))
    print(f.groupby(['exercise','method','selected_filter','alpha','beta']).size().to_string())
    print('PASS: original Gaussian counts reproduced for all 105 videos')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--traces',required=True); p.add_argument('--labels',required=True); p.add_argument('--out',required=True); a=p.parse_args(); main(a.traces,a.labels,a.out)
