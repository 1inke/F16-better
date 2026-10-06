# Control patch provenance

The first source-ordered launch ledger patch was rejected before file application because sorting used a substring that matched nested same-name JSON keys instead of the root field. Readback confirmed original ledger status and SHA c9df482b1a79e09caafff552bd0b248f974796a169cac152a5bc6c07be392f65 unchanged. The patch was sorted with strict newline/root anchors and the unchanged current_experiment no-op omitted; corrected patch applied successfully. Child/parent runtime and startup/report patch separately applied successfully. No scientific source/model/result was edited or threshold changed, and no worker/startup/progress query was repeated.

Failed patch preserved: C:/Users/linke/.codex/visualizations/2026/10/03/01a10100-026b-7573-a3c9-af7cae306591/time_launch_ledger_ordered.patch. Corrected source: time_launch_ledger_rootanchored.patch in the same staging folder. This is control bookkeeping provenance, not a production failure.
