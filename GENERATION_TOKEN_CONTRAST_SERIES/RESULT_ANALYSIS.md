# TOKEN_CONTRAST audited result (cycle4 experiment2)
NormalSUCCESS2026-10-02T03:13:25.310809+08:00; start02:55:55.570976 =>17m30s, faster than40minplanning.18SHA/18sampleIDsfiniteK20/independentallstoredmetricsrecompute/frozen18/SUCCESS PASS. NativechangeADE/FDE .372175/.696326; temporalcontrast .367102/.689724 (gain1.36%/.95%), freshzero .367399/.688230.
NativeADEdelta-.005072 CI[-.007264,-.003007], FDE-.006602[-.013490,-.000966] exploratorynegative; freshzeroADE-.000297[-.001772,.001207], FDE+.001494[-.001865,.006069] crosszero,contrast increment notestablished. Boththree seeds nativeimprovementPASS but5%/nativeANDcontrolCI/motiondistributionFAIL. Protected/samepath/coverage/dup/hardmotion PASS. Overallaccel/jerkmean+2.24% vsnative,control+2.78%,notwithin1%. Changefamilycoverage.905830vsnative.910314. No major/baseline/intent/ranking claim; nofinalpolicy posthocpromotion.
Native gains may largely be sharedmean/value/newhead; cross-experiment controls differentarchitecture/init and repeatedETHdev cannot claimindependentreplication. Nexttest ordered centeredsequence versuszero,freshbotharms samebudget. Completed2/3,thirdcompletion mandatorycycle4review beforefurtherbigdirection. OldPID45988complete no restart.
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
      "minade20": 0.21803994476795197,
      "minfde20": 0.40034687519073486,
      "joint_oracle": 0.431938111782074,
      "gt_family_covered": 0.9835092348284961,
      "acceleration_audit_mean": 0.10223215073347092,
      "jerk_audit_mean": 0.5622729659080505
    },
    {
      "arm": "zero",
      "group": "stable",
      "minade20": 0.17470230162143707,
      "minfde20": 0.316394567489624,
      "joint_oracle": 0.3435874581336975,
      "gt_family_covered": 1,
      "acceleration_audit_mean": 0.10528416931629181,
      "jerk_audit_mean": 0.5790587067604065
    },
    {
      "arm": "zero",
      "group": "transition",
      "minade20": 0.2752705216407776,
      "minfde20": 0.5126436948776245,
      "joint_oracle": 0.5469912886619568,
      "gt_family_covered": 0.9823008849557522,
      "acceleration_audit_mean": 0.09417217969894409,
      "jerk_audit_mean": 0.5179435014724731
    },
    {
      "arm": "zero",
      "group": "change",
      "minade20": 0.3673994243144989,
      "minfde20": 0.6882303357124329,
      "joint_oracle": 0.7380731105804443,
      "gt_family_covered": 0.905829596412556,
      "acceleration_audit_mean": 0.09579741209745407,
      "jerk_audit_mean": 0.5268834233283997
    },
    {
      "arm": "mean_std",
      "group": "overall",
      "minade20": 0.21787555515766144,
      "minfde20": 0.40020081400871277,
      "joint_oracle": 0.43148279190063477,
      "gt_family_covered": 0.9835092348284961,
      "acceleration_audit_mean": 0.10169662535190582,
      "jerk_audit_mean": 0.5593276023864746
    },
    {
      "arm": "mean_std",
      "group": "stable",
      "minade20": 0.17448246479034424,
      "minfde20": 0.31578654050827026,
      "joint_oracle": 0.3430539667606354,
      "gt_family_covered": 1,
      "acceleration_audit_mean": 0.10461053252220154,
      "jerk_audit_mean": 0.575353741645813
    },
    {
      "arm": "mean_std",
      "group": "transition",
      "minade20": 0.27549877762794495,
      "minfde20": 0.5130602717399597,
      "joint_oracle": 0.5469155311584473,
      "gt_family_covered": 0.9823008849557522,
      "acceleration_audit_mean": 0.093898244202137,
      "jerk_audit_mean": 0.5164368748664856
    },
    {
      "arm": "mean_std",
      "group": "change",
      "minade20": 0.36710232496261597,
      "minfde20": 0.6897244453430176,
      "joint_oracle": 0.7376075983047485,
      "gt_family_covered": 0.905829596412556,
      "acceleration_audit_mean": 0.09565752744674683,
      "jerk_audit_mean": 0.5261139869689941
    }
  ],
  "change_ci": [
    {
      "policy": "constrained",
      "reference": "native",
      "group": "change",
      "metric": "minade20",
      "delta": -0.005072439604498702,
      "ci_low": -0.007263884747119673,
      "ci_high": -0.0030071929051235916
    },
    {
      "policy": "constrained",
      "reference": "native",
      "group": "change",
      "metric": "minfde20",
      "delta": -0.006601871585721214,
      "ci_low": -0.013490199576784902,
      "ci_high": -0.0009659990964816593
    },
    {
      "policy": "constrained",
      "reference": "native",
      "group": "change",
      "metric": "top3_ade",
      "delta": -0.0033479143262622247,
      "ci_low": -0.005260999361261926,
      "ci_high": -0.0016584560728906174
    },
    {
      "policy": "constrained",
      "reference": "native",
      "group": "change",
      "metric": "joint_oracle",
      "delta": -0.009023695204111849,
      "ci_low": -0.01408260404027898,
      "ci_high": -0.004915756429858808
    },
    {
      "policy": "constrained",
      "reference": "zero",
      "group": "change",
      "metric": "minade20",
      "delta": -0.00029711644144336205,
      "ci_low": -0.0017724197904992803,
      "ci_high": 0.0012065614669713202
    },
    {
      "policy": "constrained",
      "reference": "zero",
      "group": "change",
      "metric": "minfde20",
      "delta": 0.001494142222698494,
      "ci_low": -0.001864907181834292,
      "ci_high": 0.006068579295458011
    },
    {
      "policy": "constrained",
      "reference": "zero",
      "group": "change",
      "metric": "top3_ade",
      "delta": 0.000034123525491209844,
      "ci_low": -0.001143083659589558,
      "ci_high": 0.0013060937185948556
    },
    {
      "policy": "constrained",
      "reference": "zero",
      "group": "change",
      "metric": "joint_oracle",
      "delta": -0.000465535540662717,
      "ci_low": -0.003467258298143143,
      "ci_high": 0.0030316126102690162
    }
  ]
}
```
