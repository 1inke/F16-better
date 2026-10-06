# Cycle7 experiment2 completion analysis

BASIS_CONDITION workerSUCCESS2026-10-02 23:38:48HK, actual35m59.9s. Production exitSUCCESS; independent auditPASS18modelSHA/18explicit1516IDfiniteK20/allstoredmetrics recomputed/frozenoriginaldecoder+encoder/source/10931oldfileSHA unchanged. SHA/source separately reread verified thisreview. OldPID33520done noquery/retrain. Results and scientificsources notmodified.

Preregistered constrained Gaussian-condition arm changeADE/FDE .37115124/.69715631 vsnative .37217474/.69632643: ADE gain.275%, FDE harm.119%. RawGaussian freshcontrol .37177491/.69538683, numericallysamepreviousGaussian architecture/seeds, notindependentreplication.

Conditioned-native ADEdelta-.00102355 CI[-.001864,-.00023255], FDEdelta+.00082998 CI[-.00036228,.00211958]. Conditioned-raw ADEdelta-.00062368 CI[-.001531,.00018503], FDEdelta+.00176947 CI[.00071624,.00286615]. Jointdelta+.00021865 CI[-.0010044,.00145398]. Exploratory FDEharmrelativecontrol; no provenoverallincrement/major/newbaseline.

Overall/stable/transition/change conditionedADE .21830256/.17433678/.27505589/.37115124; FDE .39964104/.31357729/.51240230/.69715631. Protection/coverage/duplicates/hardkinematics/samepath pass,5%/perseed/nativefreshcontrolCI/motiondistribution fail. Overall accelerationmean .10029429vsnative.09946864(+.830%); stable accelerationmean .10329703vs.10235941(+.916%); transition jerkp95 2.88868625vs2.85860052(+1.052%); changejerkp95 2.93265474vs2.88845197(+1.530%). Full allpolicy/allgroup metrics andCI retained originalsummary.json, not bestpolicyselection.

Interpretation: covariance conditioning doesnot reliablysolve GT-proximal generation. Itchangesfiniteoptimization and weightdecay prior, not unboundedlinearfunctionspan; doesnotprove cause/noise/intent/futureimpossibility. No floor/bandwidth/epochs sweep. Thirdpreregistered trial tests nonlinearcomposition withmatchedpaired4326parameterfactorizedlinearcontrol. Thischangesfunctionclass/optimization, isnot guaranteedbenefit and mustrespectallgates. Thirdcomplete mandatorycycle7fullreview including earlierMLPnegativeevidence, capacity, calibrationfitbehavior, repeatedcontrols/ETHdevlimitations.
