"""Reuse SHA-locked parent identity/labels unchanged, no fit or model forward."""
import common as c,time,shutil
def prepare():
    parent=c.H.parent/"stage1_selector";old=c.read(parent/"source_manifest.json")["files"];c.verify(old);copied={}
    for name in ("split_registry.npz","labels_offline.npz","IDENTITY_AUDIT.json"):
        src=parent/name;dst=c.H/name;assert not dst.exists();shutil.copy2(src,dst);assert c.sha(src)==c.sha(dst);copied[name]=c.sha(src)
    c.verify(old)
    c.write("PREPARATION_REUSE.json",dict(status="PASS_PARENT_IDENTITY_LABEL_BYTE_IDENTICAL_REUSE",files=copied,inherited=len(old),no_new_split_or_labels_or_fits=True))
    c.stage("REUSED_PARENT_FIXED_IDS_LABELS",done=0)
if __name__=="__main__":prepare()
