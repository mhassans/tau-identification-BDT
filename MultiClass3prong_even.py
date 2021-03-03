import sys
#sys.path.append("/eos/home-m/mhassans/.local/lib/python2.7/site-packages")

#!echo $PYTHONPATH

import ROOT
import uproot # can also use root_pandas or root_numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
#%matplotlib inline
import time
start_time = time.time()


tree1=uproot.open("/vols/cms/mhh18/Offline/output/SM/Validation_July4_LeadStripSignalThenIsoCone_2017/trainNtuple_Signal/trainNtuple_tt_2017_LeadStripSignalThenIsoCone.root")["train_ntuple"]

tree2=uproot.open("/vols/cms/mhh18/Offline/output/SM/Validation_July4_LeadStripSignalThenIsoCone_2017/trainNtuple_Signal/trainNtuple_mt_2017_LeadStripSignalThenIsoCone.root")["train_ntuple"]


df_1 = tree1.pandas.df(["wt", # always require the event weights
                        "event",
                        "wt_cp_sm",
                        "gen_match_1", # gen match info (==5 for real tau_h)
                        "tau_decay_mode_1", # reco tau decay mode
                        "tauFlag1", # gen tau decay mode
                        "mass0_1",
                        "mass1_1",
                        "mass2_1",
                        "E1_1",
                        "E2_1",
                        "E3_1",
                        "strip_E_1",
                        "a1_pi0_dEta_1",
                        "a1_pi0_dphi_1",
                        "strip_pt_1",
                        "pt_1",
                        "eta_1",
                        "E_1",
                        "h1_h2_dphi_1",
                        "h1_h3_dphi_1",
                        "h2_h3_dphi_1",
                        "h1_h2_dEta_1",
                        "h1_h3_dEta_1",
                        "h2_h3_dEta_1", 
                        
                        "Egamma1_1",
                        "Egamma2_1",
                        "gammas_dEta_1",
                        "Mpi0_1",
                        "gammas_dphi_1",
                        "Mpi0_TwoHighGammas_1",


                      ])

# df_2 will be for the subleading tau ("*_2") in mutau input
df_2 = tree1.pandas.df(["wt", # always require the event weights
                         "event",
                        "wt_cp_sm",
                        "gen_match_2", # gen match info (==5 for real tau_h)
                        "tau_decay_mode_2", # reco tau decay mode
                        "tauFlag2", # gen tau decay mode
                        "mass0_2",
                        "mass1_2",
                        "mass2_2",
                        "E1_2",
                        "E2_2",
                        "E3_2",
                        "strip_E_2",
                        "a1_pi0_dEta_2",
                        "a1_pi0_dphi_2",
                        "strip_pt_2",
                        "pt_2",
                        "eta_2",
                        "E_2",
                        "h1_h2_dphi_2",
                        "h1_h3_dphi_2",
                        "h2_h3_dphi_2",
                        "h1_h2_dEta_2",
                        "h1_h3_dEta_2",
                        "h2_h3_dEta_2", 

                        "Egamma1_2",
                        "Egamma2_2",
                        "gammas_dEta_2",
                        "Mpi0_2",
                        "gammas_dphi_2",
                        "Mpi0_TwoHighGammas_2",

                       ])

# df_3 will be for the tau ("*_2") in mutau input
df_3 = tree2.pandas.df(["wt", # always require the event weights
                        "event",
                        "wt_cp_sm",
                        "gen_match_2", # gen match info (==5 for real tau_h)
                        "tau_decay_mode_2", # reco tau decay mode
                        "tauFlag2", # gen tau decay mode
                        "mass0_2",
                        "mass1_2",
                        "mass2_2",
                        "E1_2",
                        "E2_2",
                        "E3_2",
                        "strip_E_2",
                        "a1_pi0_dEta_2",
                        "a1_pi0_dphi_2",
                        "strip_pt_2",
                        "pt_2",
                        "eta_2",
                        "E_2",
                        "h1_h2_dphi_2",
                        "h1_h3_dphi_2",
                        "h2_h3_dphi_2",
                        "h1_h2_dEta_2",
                        "h1_h3_dEta_2",
                        "h2_h3_dEta_2", 

                        "Egamma1_2",
                        "Egamma2_2",
                        "gammas_dEta_2",
                        "Mpi0_2",
                        "gammas_dphi_2",
                        "Mpi0_TwoHighGammas_2",

                      ])





df_1 = df_1[
        #(df_1["mva_olddm_vloose_1"] > 0.5)
    #& (df_1["antiele_1"] == True)
    #& (df_1["antimu_1"] == True)
    (df_1["tau_decay_mode_1"] > 9)
    &(df_1["gen_match_1"] == 5)
]


df_2 = df_2[
        #(df_2["mva_olddm_vloose_2"] > 0.5)
    #& (df_2["antiele_2"] == True)
    #& (df_2["antimu_2"] == True)
    (df_2["tau_decay_mode_2"] >9)
    &(df_2["gen_match_2"] == 5)
]




df_3 = df_3[
        #(df_2["mva_olddm_vloose_2"] > 0.5)
    #& (df_2["antiele_2"] == True)
    #& (df_2["antimu_2"] == True)
    (df_3["tau_decay_mode_2"] >9)
    &(df_3["gen_match_2"] == 5)
]


# define some new variables

# for df_1
df_1.loc[:,"E1_overEa1"]    = df_1["E1_1"] / (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"])
df_1.loc[:,"E2_overEa1"]    = df_1["E2_1"] / (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"])
df_1.loc[:,"E1_overEtau"]   = df_1["E1_1"] / (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"] + df_1["strip_E_1"])
df_1.loc[:,"E2_overEtau"]   = df_1["E2_1"] / (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"] + df_1["strip_E_1"])
df_1.loc[:,"E3_overEtau"]   = df_1["E3_1"] / (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"] + df_1["strip_E_1"])
df_1.loc[:,"a1_pi0_dEta_timesEtau"] = df_1["a1_pi0_dEta_1"] * (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"] + df_1["strip_E_1"])
df_1.loc[:,"a1_pi0_dphi_timesEtau"] = df_1["a1_pi0_dphi_1"] * (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"] + df_1["strip_E_1"])
df_1.loc[:,"h1_h2_dphi_timesE12"] = df_1["h1_h2_dphi_1"] * (df_1["E1_1"] + df_1["E2_1"])
df_1.loc[:,"h1_h2_dEta_timesE12"] = df_1["h1_h2_dEta_1"] * (df_1["E1_1"] + df_1["E2_1"])
df_1.loc[:,"h1_h3_dphi_timesE13"] = df_1["h1_h3_dphi_1"] * (df_1["E1_1"] + df_1["E3_1"])
df_1.loc[:,"h1_h3_dEta_timesE13"] = df_1["h1_h3_dEta_1"] * (df_1["E1_1"] + df_1["E3_1"])
df_1.loc[:,"h2_h3_dphi_timesE23"] = df_1["h2_h3_dphi_1"] * (df_1["E2_1"] + df_1["E3_1"])
df_1.loc[:,"h2_h3_dEta_timesE23"] = df_1["h2_h3_dEta_1"] * (df_1["E2_1"] + df_1["E3_1"])
df_1.loc[:,"gammas_dEta_timesEtau"] = df_1["gammas_dEta_1"] * (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"] + df_1["strip_E_1"])

df_1.loc[:,"gammas_dR_timesEtau"] = np.sqrt(df_1["gammas_dEta_1"]*\
                    df_1["gammas_dEta_1"] + df_1["gammas_dphi_1"]*df_1["gammas_dphi_1"])*\
                    (df_1["E1_1"] + df_1["E2_1"] + df_1["E3_1"] + df_1["strip_E_1"])


# now the same for df_2
df_2.loc[:,"E1_overEa1"]    = df_2["E1_2"] / (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"])
df_2.loc[:,"E2_overEa1"]    = df_2["E2_2"] / (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"])
df_2.loc[:,"E1_overEtau"]   = df_2["E1_2"] / (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"] +   df_2["strip_E_2"])
df_2.loc[:,"E2_overEtau"]   = df_2["E2_2"] / (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"] +   df_2["strip_E_2"])
df_2.loc[:,"E3_overEtau"]   = df_2["E3_2"] / (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"] +   df_2["strip_E_2"])
df_2.loc[:,"a1_pi0_dEta_timesEtau"] = df_2["a1_pi0_dEta_2"] * (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"] + df_2["strip_E_2"])
df_2.loc[:,"a1_pi0_dphi_timesEtau"] = df_2["a1_pi0_dphi_2"] * (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"] + df_2["strip_E_2"])
df_2.loc[:,"h1_h2_dphi_timesE12"] = df_2["h1_h2_dphi_2"] * (df_2["E1_2"] + df_2["E2_2"])
df_2.loc[:,"h1_h2_dEta_timesE12"] = df_2["h1_h2_dEta_2"] * (df_2["E1_2"] + df_2["E2_2"])
df_2.loc[:,"h1_h3_dphi_timesE13"] = df_2["h1_h3_dphi_2"] * (df_2["E1_2"] + df_2["E3_2"])
df_2.loc[:,"h1_h3_dEta_timesE13"] = df_2["h1_h3_dEta_2"] * (df_2["E1_2"] + df_2["E3_2"])
df_2.loc[:,"h2_h3_dphi_timesE23"] = df_2["h2_h3_dphi_2"] * (df_2["E2_2"] + df_2["E3_2"])
df_2.loc[:,"h2_h3_dEta_timesE23"] = df_2["h2_h3_dEta_2"] * (df_2["E2_2"] + df_2["E3_2"])
df_2.loc[:,"gammas_dEta_timesEtau"] = df_2["gammas_dEta_2"] * (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"] + df_2["strip_E_2"])

df_2.loc[:,"gammas_dR_timesEtau"] = np.sqrt(df_2["gammas_dEta_2"]*\
                   df_2["gammas_dEta_2"] + df_2["gammas_dphi_2"]*df_2["gammas_dphi_2"])*\
                   (df_2["E1_2"] + df_2["E2_2"] + df_2["E3_2"] + df_2["strip_E_2"])


# now the same for df_3
df_3.loc[:,"E1_overEa1"]    = df_3["E1_2"] / (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"])
df_3.loc[:,"E2_overEa1"]    = df_3["E2_2"] / (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"])
df_3.loc[:,"E1_overEtau"]   = df_3["E1_2"] / (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"] +   df_3["strip_E_2"])
df_3.loc[:,"E2_overEtau"]   = df_3["E2_2"] / (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"] +   df_3["strip_E_2"])
df_3.loc[:,"E3_overEtau"]   = df_3["E3_2"] / (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"] +   df_3["strip_E_2"])
df_3.loc[:,"a1_pi0_dEta_timesEtau"] = df_3["a1_pi0_dEta_2"] * (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"] + df_3["strip_E_2"])
df_3.loc[:,"a1_pi0_dphi_timesEtau"] = df_3["a1_pi0_dphi_2"] * (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"] + df_3["strip_E_2"])
df_3.loc[:,"h1_h2_dphi_timesE12"] = df_3["h1_h2_dphi_2"] * (df_3["E1_2"] + df_3["E2_2"])
df_3.loc[:,"h1_h2_dEta_timesE12"] = df_3["h1_h2_dEta_2"] * (df_3["E1_2"] + df_3["E2_2"])
df_3.loc[:,"h1_h3_dphi_timesE13"] = df_3["h1_h3_dphi_2"] * (df_3["E1_2"] + df_3["E3_2"])
df_3.loc[:,"h1_h3_dEta_timesE13"] = df_3["h1_h3_dEta_2"] * (df_3["E1_2"] + df_3["E3_2"])
df_3.loc[:,"h2_h3_dphi_timesE23"] = df_3["h2_h3_dphi_2"] * (df_3["E2_2"] + df_3["E3_2"])
df_3.loc[:,"h2_h3_dEta_timesE23"] = df_3["h2_h3_dEta_2"] * (df_3["E2_2"] + df_3["E3_2"])
df_3.loc[:,"gammas_dEta_timesEtau"] = df_3["gammas_dEta_2"] * (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"] + df_3["strip_E_2"])

df_3.loc[:,"gammas_dR_timesEtau"] = np.sqrt(df_3["gammas_dEta_2"]*\
                   df_3["gammas_dEta_2"] + df_3["gammas_dphi_2"]*df_3["gammas_dphi_2"])*\
                   (df_3["E1_2"] + df_3["E2_2"] + df_3["E3_2"] + df_3["strip_E_2"])


for key, values in df_1.iteritems():
    if "_1" in key:
        print(key)
        df_1.loc[:,key[:-2]] = df_1[key]
        df_1 = df_1.drop(key, axis=1).reset_index(drop=True)

df_1.loc[:,"tauFlag"] = df_1["tauFlag1"]
df_1 = df_1.drop("tauFlag1", axis=1).reset_index(drop=True)

for key, values in df_2.iteritems():
    if "_2" in key:
        print(key)
        df_2.loc[:,key[:-2]] = df_2[key]
        df_2 = df_2.drop(key, axis=1).reset_index(drop=True)
        
df_2.loc[:,"tauFlag"] = df_2["tauFlag2"]
df_2 = df_2.drop("tauFlag2", axis=1).reset_index(drop=True)


for key, values in df_3.iteritems():
    if "_2" in key:
        print(key)
        df_3.loc[:,key[:-2]] = df_3[key]
        df_3 = df_3.drop(key, axis=1).reset_index(drop=True)
        
df_3.loc[:,"tauFlag"] = df_3["tauFlag2"]
df_3 = df_3.drop("tauFlag2", axis=1).reset_index(drop=True)

comb_df = pd.concat([df_1,df_2], ignore_index=True)
comb_df = pd.concat([comb_df,df_3], ignore_index=True)


df_threeprong = comb_df[
    (comb_df["tauFlag"] == 10) 
]
print(df_threeprong.shape)

df_threeprong_pi0 = comb_df[
    (comb_df["tauFlag"] == 11)
]
print(df_threeprong_pi0.shape)

df_other = comb_df[
    (comb_df["tauFlag"]!=10) & (comb_df["tauFlag"]!=11)
]
print(df_other.shape)

# prepare the target labels
y_other = pd.DataFrame(np.zeros(df_other.shape[0]))
y_threeprong = pd.DataFrame(np.zeros(df_threeprong.shape[0]))
y_threeprong_pi0 = pd.DataFrame(np.zeros(df_threeprong_pi0.shape[0]))


y_other+=0
y_threeprong+=1
y_threeprong_pi0+=2



frames = [df_other, df_threeprong, df_threeprong_pi0]

X = pd.concat(frames)
w = X["wt_cp_sm"]

y_frames = [y_other, y_threeprong, y_threeprong_pi0 ]

y = pd.concat(y_frames).reset_index(drop=True)
y.columns = ["class"]



# drop any other variables that aren't required in training
X = X.drop(["wt","gen_match","tauFlag","gammas_dphi"], axis=1).reset_index(drop=True)

print X.shape



X_odd = X[X["event"]==0]
X_even = X[X["event"]==1]

X_odd_indices = X_odd.index.tolist()
X_even_indices = X_even.index.tolist()

y_odd = y.drop(X_even_indices)
y_even = y.drop(X_odd_indices)

w_odd = X_odd["wt_cp_sm"]
w_even = X_even["wt_cp_sm"]

X_odd = X_odd.drop(["event","wt_cp_sm"], axis=1).reset_index(drop=True)
X_even = X_even.drop(["event","wt_cp_sm"], axis=1).reset_index(drop=True)

print 'variables used in training: '
print X_odd.columns.values

xgb_params = {
    "objective": "multi:softprob",
    "max_depth": 5,
    "learning_rate": 0.05,
    "silent": 1,
    "n_estimators": 2000,
    "subsample": 0.9,
    "seed": 123451,
    "n_jobs":4,
}


xgb_clf_even = xgb.XGBClassifier(**xgb_params)
xgb_clf_even.fit(
    X_even,
    y_even,
    sample_weight=w_even,
    early_stopping_rounds=100,
    eval_set=[(X_even, y_even, w_even), (X_odd, y_odd, w_odd)],
    eval_metric = "mlogloss", 
    verbose=True,
)
xgb_clf_even.get_booster().save_model('mvadm_inclusive_2fold_applytoodd.model')


with open ("mvadm_inclusive_2fold_applytoodd.pkl",'w') as f:
    pickle.dump(xgb_clf_even,f)
print 'even-model saved!'

proba_even=xgb_clf_even.predict_proba(X_odd)

with open ("proba_even.pkl",'w') as f:
    pickle.dump(proba_even,f)

with open ("w_odd.pkl",'w') as f:
    pickle.dump(w_odd,f)

with open ("X_odd.pkl",'w') as f:
    pickle.dump(X_odd,f)

with open ("y_odd.pkl",'w') as f:
    pickle.dump(y_odd,f)

predict_even=xgb_clf_even.predict(X_odd)

with open ("predict_even.pkl",'w') as f:
    pickle.dump(predict_even,f)



elapsed_time = time.time() - start_time
print "elapsed time=", elapsed_time




