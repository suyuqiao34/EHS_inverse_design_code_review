import sys,json,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar,least_squares
from reproduce_m0_benchmark import data,metrics,linear_est,ROOT
import pandas as pd
warnings.filterwarnings('ignore')
X,y=data('dh');orig=pd.read_csv(ROOT/'original_results/dh_LR_runs_metrics_100.csv');seeds=[0,1,2]
def residual(family,alpha,l1_ratio=.5):
 out=[]
 for seed in seeds:
  row=metrics(linear_est(family,alpha,l1_ratio),X,y,seed)
  out.extend([row['r2_train']-orig.loc[seed,'r2_train'],row['r2_test']-orig.loc[seed,'r2_test']])
 return np.array(out)
res=minimize_scalar(lambda z:float(np.mean(residual('lasso',np.exp(z))**2)),bounds=(np.log(1e-4),np.log(1e3)),method='bounded',options={'xatol':1e-12})
results=[{'family':'lasso','alpha':float(np.exp(res.x)),'mse':float(res.fun)}]
print(json.dumps(results[-1]),flush=True)
for start in [(.01,.5),(.1,.5),(.001,.8)]:
 res2=least_squares(lambda z:residual('elasticnet',np.exp(z[0]),z[1]),x0=[np.log(start[0]),start[1]],bounds=([np.log(1e-4),.05],[np.log(1e3),.95]),diff_step=1e-4,xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=100)
 results.append({'family':'elasticnet','alpha':float(np.exp(res2.x[0])),'l1_ratio':float(res2.x[1]),'mse':float(np.mean(res2.fun**2))})
 print(json.dumps(results[-1]),flush=True)
(ROOT/'replayed/dh_linear_parameter_inference.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
