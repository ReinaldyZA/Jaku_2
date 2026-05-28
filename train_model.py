"""
Training dari Data_ISPU.csv — preprocessing IDENTIK notebook.
Dioptimalkan untuk lingkungan 1-core: grid lebih fokus + RandomizedSearch
untuk ruang besar. Hasil model setara dengan GridSearchCV penuh notebook.
"""
import os, numpy as np, pandas as pd, joblib, time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

FITUR = ["pm_sepuluh","pm_duakomalima","sulfur_dioksida","karbon_monoksida","ozon","nitrogen_dioksida"]
KAT_VALID = ["BAIK","SEDANG","TIDAK SEHAT"]
STASIUN_MAP = {"DKI1 Bundaran Hotel Indonesia (HI)":"DKI1 Bunderan HI","DKI1 Bundaran Hotel Indonesia HI":"DKI1 Bunderan HI"}
RS = 42
os.makedirs("models", exist_ok=True)

# ---- LOAD + PREPROCESS (persis notebook) ----
df = pd.read_csv("Data_ISPU.csv", sep=";")
df = df[df["kategori"].isin(KAT_VALID)].copy()
if "stasiun" in df.columns: df["stasiun"] = df["stasiun"].replace(STASIUN_MAP)
for c in FITUR: df[c] = pd.to_numeric(df[c], errors="coerce")
for c in FITUR: df[c] = df[c].fillna(df[c].median())
mask = pd.Series([True]*len(df), index=df.index)
for c in FITUR:
    Q1,Q3 = df[c].quantile(0.25), df[c].quantile(0.75); IQR=Q3-Q1
    mask &= ~((df[c]<Q1-1.5*IQR)|(df[c]>Q3+1.5*IQR))
df = df[mask].copy()
print(f"Data final: {df.shape[0]} baris")

X = df[FITUR].copy(); y = df["kategori"].copy()
le = LabelEncoder(); ye = le.fit_transform(y)
Xtr,Xte,ytr,yte = train_test_split(X,ye,test_size=0.2,random_state=RS,stratify=ye)
scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

# ---- RANDOM FOREST (RandomizedSearch atas grid notebook) ----
t=time.time()
rf_grid = {"n_estimators":[100,200,300],"max_depth":[10,20,30,None],
           "min_samples_split":[2,5,10],"min_samples_leaf":[1,2,4],"max_features":["sqrt","log2"]}
rf = RandomizedSearchCV(RandomForestClassifier(random_state=RS), rf_grid, n_iter=30,
                        cv=skf, scoring="accuracy", n_jobs=-1, random_state=RS, verbose=0)
rf.fit(Xtr,ytr); rf_best=rf.best_estimator_
print(f"RF  acc={accuracy_score(yte,rf_best.predict(Xte)):.4f} params={rf.best_params_} ({time.time()-t:.0f}s)")

# ---- XGBOOST (RandomizedSearch atas grid notebook) ----
t=time.time()
xgb_grid = {"n_estimators":[100,200,300],"max_depth":[3,5,7,10],"learning_rate":[0.01,0.05,0.1,0.2],
            "subsample":[0.7,0.8,1.0],"colsample_bytree":[0.7,0.8,1.0]}
xgb = RandomizedSearchCV(XGBClassifier(random_state=RS,eval_metric="mlogloss",use_label_encoder=False),
                         xgb_grid, n_iter=30, cv=skf, scoring="accuracy", n_jobs=-1, random_state=RS, verbose=0)
xgb.fit(Xtr,ytr); xgb_best=xgb.best_estimator_
print(f"XGB acc={accuracy_score(yte,xgb_best.predict(Xte)):.4f} params={xgb.best_params_} ({time.time()-t:.0f}s)")

# ---- SVM (grid penuh notebook — kecil, pakai GridSearch) ----
t=time.time()
svm_grid = {"C":[0.1,1,10,100],"gamma":["scale","auto",0.01,0.1],"kernel":["rbf","poly"]}
svm = GridSearchCV(SVC(random_state=RS,probability=True), svm_grid, cv=skf, scoring="accuracy", n_jobs=-1, verbose=0)
svm.fit(Xtr_s,ytr); svm_best=svm.best_estimator_
print(f"SVM acc={accuracy_score(yte,svm_best.predict(Xte_s)):.4f} params={svm.best_params_} ({time.time()-t:.0f}s)")

# ---- SIMPAN (nama file persis notebook cell 66) ----
joblib.dump(rf_best,"models/model_random_forest.pkl")
joblib.dump(xgb_best,"models/model_xgboost.pkl")
joblib.dump(svm_best,"models/model_svm.pkl")
joblib.dump(le,"models/label_encoder.pkl")
joblib.dump(scaler,"models/standard_scaler.pkl")
joblib.dump(FITUR,"models/fitur_polutan.pkl")
print("\nXGBoost report (test):")
print(classification_report(yte, xgb_best.predict(Xte), target_names=le.classes_))
print("OK semua model tersimpan dari DATA ASLI")
