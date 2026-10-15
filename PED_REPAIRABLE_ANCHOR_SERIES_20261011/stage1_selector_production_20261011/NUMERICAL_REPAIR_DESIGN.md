# Fixed-objective solver repair, registered before performance outcomes

Latest human: “自己处理这个问题”. Parent source/code/PREFLIGHT failure remain frozen, no rewriting old hashes or results.

## Diagnosis and selected remedy

Original standardized 513-window fixture (10260candidate rows,460positive) exhausts lbfgs500 and raises ConvergenceWarning. This establishes numerical budget failure, not absence of predictive signal. Repeated P2/native geometric coordinates may create correlated directions, but a specific conditioning cause is not asserted before measured Hessian evidence.

Use the installed sklearn1.4.1.post1 binary Newton-Cholesky solver once, fixed. It computes the120+intercept Hessian, suitable for many candidate rows and only120features. No feature whitening/PCA/feature deletion and no changed regularization metric, which would alter the statistical model. C1/L2/balanced/intercept/float64 fit normalization/tol1e-6/max_iter500 and every scientific gate unchanged.

## Objective equivalence

For n rows y_i in{0,1}, a_i=n/(2 n_class(y_i)), S=sum a_i, z_i=x_i dot w+b:

F(w,b)=sum a_i[logaddexp(0,z_i)-y_i z_i]/S + ||w||²/(2 C S).

Intercept is unpenalized. Local installed _logistic.py uses l2_reg_strength=1/(C*sum(sample_weight)) for both lbfgs and newton-cholesky. The standalone NumPy objective/gradient/Hessian reproduces this exactly. Software files are SHA-bound, not updated.

## Actual numerical verification

First actual513 fixture fitted twice with new solver and all weights/statistics/scores required BITWISE repeat. Independent weighted-gradient infinity norm<=original1e-6 in every preflight/OOF/full-A fit. Fixed independent scipy trust-exact from zero only on that fixture, max500/gtol1e-9, independently implemented objective and Hessian; stationarity<=1e-6/objective difference<=1e-9/probability max difference<=1e-3. Probability criterion is solver verification, not a relaxed evaluation epsilon. Fixed analytic gradient/Hessian directionalFD eps1e-5 must agree<=1e-8, HessianSPD/condition reported. Reference solution is never used to select/deploy anchors. No H4/GT outcome choose solver or parameters.

If new solver fails convergence/certificate/resource/planning, STOP and preserve source; no subsequent blind solver/C/tol/iteration sweep. Completed fits' exact coefficient certificates checked again in saved-data completion without new fitting.

## Scope and unchanged gates

A7334/B11575 and all trackhash/3 OOF folds/labels byte-identical to parent. OOFoverall H4 gain>=5pp over both bottom and fixedrandom with pairedtrack97.5% lower>0; change eachfold no worse than bottom. Only then full-A once; B both improvements>0/change>=bottom then freeze B/outer/inner/CAL. Scientific gate failure may be executionSUCCESS but S2 must not run. Optimistic potential is not trajectory rescue/physics/intent/generalization.

Sources: [sklearn1.4 official docs](https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html), installed _logistic.py lines355–367/451/483 (exact installed SHA manifest). Repair choice is made from numerical model structure, before OOF/B outcome. Same-assistant ARS inline checks, not external review.
