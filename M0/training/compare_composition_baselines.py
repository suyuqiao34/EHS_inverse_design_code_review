"""Reproduce the submitted Table S4 baseline analysis (185/183 records).
An optional paired audit uses --data-dir data/round0/evaluation (186/184 records).
See reproduction_status.md for the distinction between the archived snapshots.

Default reruns do not overwrite the supplied results. Run from any directory:
    python compare_composition_baselines.py --out results_rerun/composition_baselines
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE=Path(__file__).resolve().parent
TASKS={'tm':'Tm (°C)','dh':'ΔHm (J/g)'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=HERE.parents[1]/'data/round0/manuscript_baseline')
    ap.add_argument('--out',type=Path,default=HERE/'results_rerun/composition_baselines')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    all_runs=[]; pred_rows=[]; splits=[]; provenance={}
    for task,prop in TASKS.items():
        path=args.data_dir/f'round0_{task}.csv'; df=pd.read_csv(path)
        y=pd.to_numeric(df[prop],errors='raise').to_numpy(float)
        weights=df[['ID1','ID2','ID3']].to_numpy(float)
        values=df[[f'{k}-{prop}' for k in (1,2,3)]].to_numpy(float)
        x=(weights*values).sum(axis=1)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        for seed in range(100):
            tr,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=seed)
            model=LinearRegression().fit(x[tr,None],y[tr]); p=model.predict(x[te,None])
            for name,pred in [('weighted',x[te]),('calibrated',p)]:
                all_runs.append({'task':task,'seed':seed,'method':name,'n':len(df),'test_n':len(te),
                    'r2':r2_score(y[te],pred),'rmse':mean_squared_error(y[te],pred)**.5,'mae':mean_absolute_error(y[te],pred)})
            for j,idx in enumerate(te):
                pred_rows.append({'task':task,'seed':seed,'row_index':int(idx),'y_true':y[idx],
                                  'weighted_prediction':x[idx],'calibrated_prediction':p[j]})
            splits.extend({'task':task,'seed':seed,'row_index':int(idx),'partition':label}
                          for label,ids in [('train',tr),('test',te)] for idx in ids)
        provenance[task]={'input_file':path.name,'n':len(df),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    runs=pd.DataFrame(all_runs);summary=runs.groupby(['task','method'])[['r2','rmse','mae']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    runs.to_csv(args.out/'baseline_per_split_metrics.csv',index=False)
    summary.to_csv(args.out/'baseline_summary.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(args.out/'baseline_per_sample_predictions.csv',index=False)
    pd.DataFrame(splits).to_csv(args.out/'split_membership.csv',index=False)
    provenance.update({'n_splits':100,'split_seeds':list(range(100)),'test_size':.2,'standard_deviation_ddof':1,
                       'calibration':'intercept and slope fitted on each training subset only'})
    (args.out/'protocol.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__': main()
