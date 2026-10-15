import common as c,features_r5 as ft,raw_inputs as ri,saved_math as sm
import numpy as np
TEMP=None
def temporal():
 global TEMP
 if TEMP is None:r=ri.raw();TEMP=ft.build(r["past"],r["native"])
 return TEMP
def fold(fold,arm):
 d,m,core,con=c.inputs();tr,va=con.split_masks(d,fold);base=core.transform(d["X"],m[fold],guard=c.guard);T=temporal();stats=ft.statistics(T,tr);z=ft.transform(base,T,stats,arm)
 return z,tr,va,stats
def reproduce(model,z,d,tr,va,fold):
 oldbase=c.R3/"production_20261013/pairwise"/("fold_"+str(fold));old=c.npz(oldbase/"model.npz");pred=c.npz(oldbase/"predictions.npz")
 assert np.array_equal(model.coef,old["coef"]) and model.intercept==float(old["intercept"][0]),"STOP_CONTROL_COEFFICIENT_REPRODUCTION"
 ts=c.k.score(z[tr],model);vs=c.k.score(z[va],model)
 assert np.array_equal(ts,pred["train_scores"]) and np.array_equal(vs,pred["validation_scores"]),"STOP_CONTROL_SCORE_REPRODUCTION"
 assert np.array_equal(c.k.rank4(ts),pred["train_slots"]) and np.array_equal(c.k.rank4(vs),pred["validation_slots"]),"STOP_CONTROL_SLOT_REPRODUCTION"
 return dict(fold=fold,coef_intercept_train_OOF_scores_top4_BITWISE_EXACT=True)
def reference_slots(d):
 pol={name:d["reference_slots"][name] for name in ("linear","envelope","fixed_random","bottom")};pol["pointwise"]=d["reference_slots"]["old_rbf"];a=c.npz(c.R3/"production_20261013/pairwise/OOF.npz");assert np.array_equal(a["sample_id"],d["ids"]);pol["control"]=a["slots"];return pol
