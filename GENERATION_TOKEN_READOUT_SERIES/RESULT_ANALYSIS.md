# Full observation token readout — cycle4 experiment1 result

NormalSUCCESS2026-10-02 02:35:33.304HK,actual36m32s. SixNEW80epochtrainings,18policyentries. Independent18SHA/18explicitID/1516x20x12x2finiteK20/allstoredmetrics recomputePASS;builtinalloriginaldecoder/scorerfreeze18PASS/SUCCESS. No rerunworker44332.

Constrained3seedaverage:
|group|nativeADE/FDE|uniform|attention|
|---|---|---|---|
|overall|.219154/.399725|.218235/.401099|.218498/.401075|
|stable|.175177/.313804|.175109/.317584|.175092/.316632|
|transition|.275791/.512714|.275005/.511410|.274931/.513222|
|change|.372175/.696326|.367050/.688901|.368992/.691457|

Attentionchange gains.855%/.699% not5%. Each3seedbothoracleimprovesnative (gatePASS) butnative FDEdeltaCI[-.012595,.001651]cross0;nativeADEdelta-.003183CI[-.005743,-.000923]. Jointnative -.005490CI[-.010469,-.000912]. Small exploratory signal underrepeateddev,notmajor/nobaseline/intentclaim. Uniform nominalchangegains1.377%/1.066%; it iscontrol, notposthoc primarypromotion.

AttentionvsfreshuniformchangeADE+.001942CI[-.000046,.004164] andFDE+.002556[-.001016,.006996], no attentionincrement. Joint+.003681CI[.000121,.008112] andtop3+.001535[.000200,.003238] adverse exploratory nearzero lowerbounds; notuniversalattentionharmproof. No extra seed/fold/generalizationvalidation.

Attentionprotectedoracle/samepath/eachseed/coverage/dup/hardmotion PASS;5%/native+controlCI/motiondistributionFAIL. ControlstableFDE+1.20%native, so controlitselfnotallprotectedgroupsPASS. Attentionoverallaccel/jerkmean+2.40%;uniform+2.53%, still>1%distributionguard despitehardthresholdPASS. Changejointnative.746631/control.737460/attention.741141. Coveragechange.905830bothvsnative.910314, highcoarsepresence notpathintentaccuracy. Constrainedepochsattention23/29/17,uniform23/29/26. Finalattentionchange.368497/.693493,uniform.367168/.694437;no devselectionfinalpromotion.

Methodboth39590nominalparams/fullpairedinit/nativeidentity;uniformq/keydata grad0 effectivecapacitydifference;originalcachedtokenscontextualmeanretainsinfo. Alloldweightsfixed/nativebase3anchorbound1.5/originalloss/cal,noGTforward. Fulltokensbranch isplausiblyuseful inputreadout,butthiscomparedoesnotisolateextra valueprojection versuslargerhead versuspriorDirectAnchor, so don'tprove64contextbottleneck.

Nextsinglefactor focusfixed temporalcontrast ratherthanattention coefficient/head count: bothsamecontext+mode128/valuemean32+extra32 zeroor value(last4mean-first4mean),shared192->128->64->6 zero-output/nativeidentity,onevalue96->32,both36454params. No query/key/position/attention innewmodel. Control extra32 firstlayercolumnsdata gradientszero so notequaleffectivecapacityproof. Tokensalreadycontextual, contrastmeans observationrepresentationchange notlearnedfutureintent. Nativeoriginalcontextcontainslast+mean, so newcontrastnotnewrawdata or guaranteednewinformation. Originalsplit/loss/cal/clip/guards6fresh80epoch3seed unchanged. AfterimplementationrealGPUpreflight/EXACTresumeonlylaunch. Cycle4completed1/3, thirdcompletionfullreviewmandatory.
