# SOCIAL_TAP audited result cycle5experiment1
NormalSUCCESS2026-10-02T04:35:21.826339+08:00 actual17m14s. Independent18SHA/18explicitIDsfiniteK20/allstoredmetrics recompute/frozen18/SUCCESS PASS. Treatmentchange .367382/.691784 vsnative.372175/.696326 gains1.29%/.65%,freshconstant .367508/.688556.
NativeADEdelta-.004793CI[-.008492,-.002183],FDE-.004542[-.012506,.001128] exploratory; freshzeroADE-.000127[-.003131,.002880],FDE+.003228[-.002125,.008960],joint+.001009[-.004921,.007056]allcross0. No provennewtapincrement. 5%/nativeANDcontrolCI/motiondistributionFAIL,othergatesPASS;overallacceljerk+2.23%. Selectedepochs13/19/11. No devfinalpromotion/major/newbaseline/intent/rankingclaim.
Social already inF16tokens; oldgenericresidualusedsocial; directtapnotnewdata, no proofsocialuseless orwholeF16bottleneck. Nextfactor both fullsocial,static vs mode-conditioned NEWsocialmodulation,freshbothseeds. Sharedheadalready nonlinear mode+social conditioning, so newfactor explicitinductivebias notmissingconditioningproof. PriorCONDITIONAL_MODE changednativequery instead; not universalgatingnovelty. Maintaingates/fullfrozen/testisolation. Completed1/3,thirdthenmandatoryreview.
```json
{
  "gates": {
    "change_ade_fde_gain_5pct": false,
    "protected_groups_guard": true,
    "each_seed_change_improves": true,
    "native_and_zero_ci": false,
    "all_seed_feasible": true,
    "coverage_guard": true,
    "duplicate_guard": true,
    "kinematic_guard": true,
    "same_path_joint_guard": true,
    "motion_distribution_no_harm_1pct": false
  },
  "metrics": [
    {
      "arm": "zero",
      "group": "overall",
      "minade20": 0.21808865666389465,
      "minfde20": 0.4004588723182678,
      "joint_oracle": 0.43200570344924927,
      "gt_family_covered": 0.9835092348284961,
      "speed_audit_p95": 1.2002362819622758,
      "acceleration_audit_mean": 0.10200386494398117,
      "acceleration_audit_p95": 0.7197398065206836,
      "jerk_audit_mean": 0.5610174536705017,
      "jerk_audit_p95": 3.3638025002919343
    },
    {
      "arm": "zero",
      "group": "stable",
      "minade20": 0.1747191697359085,
      "minfde20": 0.3164212107658386,
      "joint_oracle": 0.343559205532074,
      "gt_family_covered": 1,
      "speed_audit_p95": 1.2438838783361792,
      "acceleration_audit_mean": 0.10501181334257126,
      "acceleration_audit_p95": 0.7259011120572372,
      "jerk_audit_mean": 0.5775608420372009,
      "jerk_audit_p95": 3.540036991811175
    },
    {
      "arm": "zero",
      "group": "transition",
      "minade20": 0.27541038393974304,
      "minfde20": 0.5129474401473999,
      "joint_oracle": 0.5474008917808533,
      "gt_family_covered": 0.9823008849557522,
      "speed_audit_p95": 1.1272775372792478,
      "acceleration_audit_mean": 0.09401945769786835,
      "acceleration_audit_p95": 0.6814942607090174,
      "jerk_audit_mean": 0.5171035528182983,
      "jerk_audit_p95": 2.924391691751533
    },
    {
      "arm": "zero",
      "group": "change",
      "minade20": 0.36750805377960205,
      "minfde20": 0.688556432723999,
      "joint_oracle": 0.7382529377937317,
      "gt_family_covered": 0.905829596412556,
      "speed_audit_p95": 1.065333550874427,
      "acceleration_audit_mean": 0.09570334106683731,
      "acceleration_audit_p95": 0.7290195390136596,
      "jerk_audit_mean": 0.5263659954071045,
      "jerk_audit_p95": 2.9658860889874545
    },
    {
      "arm": "mean_std",
      "group": "overall",
      "minade20": 0.2179495394229889,
      "minfde20": 0.39989766478538513,
      "joint_oracle": 0.4315551519393921,
      "gt_family_covered": 0.9835092348284961,
      "speed_audit_p95": 1.2003782576908366,
      "acceleration_audit_mean": 0.10168565064668655,
      "acceleration_audit_p95": 0.7183088164838806,
      "jerk_audit_mean": 0.5592672228813171,
      "jerk_audit_p95": 3.3483861224252363
    },
    {
      "arm": "mean_std",
      "group": "stable",
      "minade20": 0.17466099560260773,
      "minfde20": 0.31548938155174255,
      "joint_oracle": 0.34302830696105957,
      "gt_family_covered": 1,
      "speed_audit_p95": 1.2441263688174815,
      "acceleration_audit_mean": 0.10461623221635818,
      "acceleration_audit_p95": 0.7230543042371612,
      "jerk_audit_mean": 0.5753850936889648,
      "jerk_audit_p95": 3.5234271827450883
    },
    {
      "arm": "mean_std",
      "group": "transition",
      "minade20": 0.27487677335739136,
      "minfde20": 0.5103973150253296,
      "joint_oracle": 0.5458890795707703,
      "gt_family_covered": 0.9823008849557522,
      "speed_audit_p95": 1.1267355671322794,
      "acceleration_audit_mean": 0.09379854798316956,
      "acceleration_audit_p95": 0.6837492127378069,
      "jerk_audit_mean": 0.515888512134552,
      "jerk_audit_p95": 2.902519931211229
    },
    {
      "arm": "mean_std",
      "group": "change",
      "minade20": 0.3673815131187439,
      "minfde20": 0.6917843818664551,
      "joint_oracle": 0.7392622828483582,
      "gt_family_covered": 0.905829596412556,
      "speed_audit_p95": 1.0656877352428715,
      "acceleration_audit_mean": 0.09565672278404236,
      "acceleration_audit_p95": 0.7306273591469385,
      "jerk_audit_mean": 0.5261095762252808,
      "jerk_audit_p95": 2.962722211452517
    }
  ],
  "ci": [
    {
      "policy": "constrained",
      "reference": "native",
      "group": "change",
      "metric": "minade20",
      "delta": -0.004793270124938278,
      "ci_low": -0.008491755224714232,
      "ci_high": -0.002182586659275008
    },
    {
      "policy": "constrained",
      "reference": "native",
      "group": "change",
      "metric": "minfde20",
      "delta": -0.004542003526325896,
      "ci_low": -0.012505876514726557,
      "ci_high": 0.00112844779920417
    },
    {
      "policy": "constrained",
      "reference": "native",
      "group": "change",
      "metric": "top3_ade",
      "delta": -0.0033308789635185703,
      "ci_low": -0.0061289660966717075,
      "ci_high": -0.0013319957338730145
    },
    {
      "policy": "constrained",
      "reference": "native",
      "group": "change",
      "metric": "joint_oracle",
      "delta": -0.007368998320826144,
      "ci_low": -0.014886097232456161,
      "ci_high": -0.0022096955839694898
    },
    {
      "policy": "constrained",
      "reference": "zero",
      "group": "change",
      "metric": "minade20",
      "delta": -0.00012652102094951054,
      "ci_low": -0.0031306229814363614,
      "ci_high": 0.002879976419528725
    },
    {
      "policy": "constrained",
      "reference": "zero",
      "group": "change",
      "metric": "minfde20",
      "delta": 0.0032279105308522737,
      "ci_low": -0.002125375654398058,
      "ci_high": 0.008959847988535736
    },
    {
      "policy": "constrained",
      "reference": "zero",
      "group": "change",
      "metric": "top3_ade",
      "delta": -0.00009198000679636215,
      "ci_low": -0.0023846497479408577,
      "ci_high": 0.002211658761840764
    },
    {
      "policy": "constrained",
      "reference": "zero",
      "group": "change",
      "metric": "joint_oracle",
      "delta": 0.001009324629566951,
      "ci_low": -0.004921183400190197,
      "ci_high": 0.007056207789222311
    }
  ]
}
```
