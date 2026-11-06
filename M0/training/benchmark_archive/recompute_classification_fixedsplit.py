"""New reproducible fixed-split assessment on recovered 194-row M0 classification input.
This is explicitly a new assessment, not a claim to reproduce the historical 40-sample table.
"""
from pathlib import Path
import ast,json,warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent
raw=pd.read_csv(ROOT/'inputs/original_m0_single_peak.csv');X=raw.iloc[:,1:-1].apply(pd.to_numeric,errors='coerce');y=pd.to_numeric(raw.iloc[:,-1],errors='coerce');mask=y.notna()&~X.isna().any(axis=1);X=X[mask].reset_index(drop=True);y=y[mask].astype(int).reset_index(drop=True)
idx=np.arange(len(y));tr,te=train_test_split(idx,test_size=.2,random_state=42,stratify=y)
table=pd.read_csv(ROOT/'original_results/single_peak_model_comparison.csv');rows=[];preds=[];hyper={}
classes={'LR':LogisticRegression,'SVM':SVC,'DT':DecisionTreeClassifier,'RF':RandomForestClassifier,'GBDT':GradientBoostingClassifier,'XGBoost':XGBClassifier}
for _,r in table.iterrows():
    name=r.model;p=ast.literal_eval(r.best_params);p['random_state']=42
    if name in ['RF','XGBoost']:p['n_jobs']=1
    hyper[name]=p;m=classes[name](**p).fit(X.iloc[tr],y.iloc[tr]);yp=m.predict(X.iloc[te]);score=m.predict_proba(X.iloc[te])[:,1] if hasattr(m,'predict_proba') else m.decision_function(X.iloc[te])
    rows.append({'model':name,'train_n':len(tr),'test_n':len(te),'n_input':len(y),'split_seed':42,'stratified':True,'accuracy':accuracy_score(y.iloc[te],yp),'precision_weighted':precision_score(y.iloc[te],yp,average='weighted',zero_division=0),'recall_weighted':recall_score(y.iloc[te],yp,average='weighted',zero_division=0),'f1_weighted':f1_score(y.iloc[te],yp,average='weighted',zero_division=0),'auc':roc_auc_score(y.iloc[te],score),'confusion_matrix_01':json.dumps(confusion_matrix(y.iloc[te],yp,labels=[0,1]).tolist()),'historical_accuracy':r.accuracy,'status':'new fixed assessment on recovered194; not historical40-sample replay'})
    for i,j in enumerate(te):preds.append({'model':name,'input_row0':int(j),'csv_line':int(j)+2,'actual':int(y.iloc[j]),'predicted':int(yp[i]),'score':float(score[i])})
    print(name,rows[-1]['accuracy'],rows[-1]['auc'],flush=True)
out=ROOT/'verified_results';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'classification_fixedsplit_recomputed_194.csv',index=False,encoding='utf-8-sig');pd.DataFrame(preds).to_csv(out/'classification_fixedsplit_predictions_194.csv',index=False,encoding='utf-8-sig');(out/'classification_fixedsplit_hyperparameters.json').write_text(json.dumps(hyper,indent=2),encoding='utf-8');pd.DataFrame({'input_row0':idx,'split':['train' if i in tr else 'test' for i in idx]}).to_csv(out/'classification_fixedsplit_indices_194.csv',index=False)
