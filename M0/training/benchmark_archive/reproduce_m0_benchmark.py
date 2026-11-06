"""Portable checks/replay of recovered M0 benchmark inputs and original run metrics.
All paths are local to this archive; original Desktop files are never opened or changed.
"""
from pathlib import Path
import argparse,ast,json,sys,hashlib,warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parent
OUTPUT_DIR=ROOT/'results_rerun'
warnings.filterwarnings('ignore')
LABELS={'tm':'tm','dh':'dh'}
def data(task):
    d=pd.read_csv(ROOT/'inputs'/f'original_m0_{task}.csv')
    X=pd.get_dummies(d.iloc[:,1:-1],drop_first=True).apply(pd.to_numeric,errors='coerce')
    y=pd.to_numeric(d.iloc[:,-1],errors='coerce')
    mask=y.notna() & ~X.isna().any(axis=1)
    return X.loc[mask].reset_index(drop=True),y.loc[mask].reset_index(drop=True)
def params(task,model):
    df=pd.read_csv(ROOT/'original_results'/f'{LABELS[task]}_model_comparison.csv')
    return ast.literal_eval(df.loc[df.model==model,'best_params'].iloc[0])
def estimator(model,p,seed):
    p=p.copy()
    if model=='GBDT':return GradientBoostingRegressor(**p,random_state=seed)
    if model=='RF':return RandomForestRegressor(**p,random_state=seed,n_jobs=1)
    if model=='DT':return DecisionTreeRegressor(**p,random_state=seed)
    if model=='SVM':return SVR(**p)
    if model=='XGBoost':return XGBRegressor(**p,random_state=seed,n_jobs=1)
    raise ValueError('LR reconstruction uses a different search/standardization script. Use --lr-family and --alpha explicitly.')
def linear_est(family,alpha=1.,l1_ratio=.5):
    if family=='linear':m=LinearRegression()
    elif family=='ridge':m=Ridge(alpha=alpha,random_state=42)
    elif family=='lasso':m=Lasso(alpha=alpha,max_iter=20000,random_state=42)
    else:m=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,max_iter=20000,random_state=42)
    return make_pipeline(StandardScaler(),m)
def metrics(m,X,y,seed):
    tr,te,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed)
    m.fit(tr,ytr);a=m.predict(tr);b=m.predict(te)
    return {'run':seed,'r2_train':r2_score(ytr,a),'r2_test':r2_score(yte,b),'rmse_train':mean_squared_error(ytr,a)**.5,'rmse_test':mean_squared_error(yte,b)**.5}
def aggregate():
    rows=[]
    for task,ch in LABELS.items():
        for m in ['LR','SVM','DT','RF','GBDT','XGBoost']:
            f=ROOT/'original_results'/f'{ch}_{m}_runs_metrics_100.csv';d=pd.read_csv(f)
            rec={'task':task,'model':m,'n_runs':len(d),'source_file':f.relative_to(ROOT).as_posix()}
            for field in ['r2_train','r2_test','rmse_train','rmse_test']:
                rec[field+'_mean']=d[field].mean();rec[field+'_std_ddof1']=d[field].std(ddof=1)
            rows.append(rec)
    out=pd.DataFrame(rows);dest=OUTPUT_DIR/'verified_results';dest.mkdir(parents=True,exist_ok=True)
    out.to_csv(dest/'regression_summary_from_original_100runs.csv',index=False,encoding='utf-8-sig')
    for task,ch in LABELS.items():
        old=pd.read_csv(ROOT/'original_results'/f'{ch}_model_comparison.csv')
        for i,r in old.iterrows():
            q=out[(out.task==task)&(out.model==r.model)].iloc[0]
            for col,key in [('r2_mean','r2_test_mean'),('r2_std','r2_test_std_ddof1'),('rmse_mean','rmse_test_mean'),('rmse_std','rmse_test_std_ddof1'),('train_r2_mean','r2_train_mean'),('train_r2_std','r2_train_std_ddof1'),('train_rmse_mean','rmse_train_mean'),('train_rmse_std','rmse_train_std_ddof1')]:old.loc[i,col]=q[key]
            old.loc[i,'repeated_result_source']=q.source_file
            old.loc[i,'fixed_split_status']='historical original comparison row retained; independent repeated LR rebuild uses different model family' if r.model=='LR' else 'historical original comparison row retained'
        old.to_csv(dest/f'{task}_comparison_verified_repeated_metrics.csv',index=False,encoding='utf-8-sig')
    print(out.to_string(index=False))
def replay(task,model,n_runs,family=None,alpha=1.,l1_ratio=.5):
    X,y=data(task);p=params(task,model) if model!='LR' else None
    rows=[]
    for seed in range(n_runs):
        m=linear_est(family,alpha,l1_ratio) if model=='LR' else estimator(model,p,seed)
        rows.append(metrics(m,X,y,seed))
        if (seed+1)%20==0:print(task,model,'completed',seed+1,flush=True)
    d=pd.DataFrame(rows);dest=OUTPUT_DIR/'replayed';dest.mkdir(parents=True,exist_ok=True)
    d.to_csv(dest/f'{task}_{model}_replayed_{n_runs}.csv',index=False,encoding='utf-8-sig')
    old=pd.read_csv(ROOT/'original_results'/f'{LABELS[task]}_{model}_runs_metrics_100.csv').set_index('run')
    diffs={f:float(np.max(abs(d.set_index('run')[f]-old.loc[d.run,f].to_numpy()))) for f in ['r2_train','r2_test','rmse_train','rmse_test']}
    summary={f:{'mean':float(d[f].mean()),'std_ddof1':float(d[f].std(ddof=1))} for f in ['r2_train','r2_test','rmse_train','rmse_test']}
    result={'task':task,'model':model,'n':len(y),'n_features':X.shape[1],'test_n':int(np.ceil(.2*len(y))),'n_runs':n_runs,'model_params':p,'lr_family':family,'alpha':alpha,'l1_ratio':l1_ratio,'max_abs_delta_vs_original':diffs,'summary':summary}
    (dest/f'{task}_{model}_replay_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
def search_ridge(task):
    X,y=data(task);orig=pd.read_csv(ROOT/'original_results'/f'{LABELS[task]}_LR_runs_metrics_100.csv')
    seeds=[0,1,2]
    def fun(log_alpha):
        err=[]
        for seed in seeds:
            v=metrics(linear_est('ridge',np.exp(log_alpha)),X,y,seed)
            for f in ['r2_train','r2_test']:err.append(v[f]-orig.loc[seed,f])
        return float(np.mean(np.square(err)))
    res=minimize_scalar(fun,bounds=(np.log(1e-4),np.log(1e3)),method='bounded',options={'xatol':1e-13})
    result={'task':task,'family':'ridge','alpha_inferred_from_saved_metrics':float(np.exp(res.x)),'fit_mse_on_r2_3seeds':float(res.fun),'status':'numerically inferred, not original saved hyperparameter'}
    print(json.dumps(result,indent=2));dest=OUTPUT_DIR/'replayed';dest.mkdir(parents=True,exist_ok=True);(dest/f'{task}_ridge_parameter_inference.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['aggregate','replay','search-ridge']);ap.add_argument('--task',choices=['tm','dh'],default='tm');ap.add_argument('--model',choices=['LR','SVM','DT','RF','GBDT','XGBoost'],default='GBDT');ap.add_argument('--runs',type=int,default=100);ap.add_argument('--lr-family',choices=['linear','ridge','lasso','elasticnet']);ap.add_argument('--alpha',type=float,default=1.);ap.add_argument('--l1-ratio',type=float,default=.5);ap.add_argument('--out',type=Path,default=ROOT/'results_rerun');args=ap.parse_args();OUTPUT_DIR=args.out
    if args.command=='aggregate':aggregate()
    elif args.command=='search-ridge':search_ridge(args.task)
    else:replay(args.task,args.model,args.runs,args.lr_family,args.alpha,args.l1_ratio)
