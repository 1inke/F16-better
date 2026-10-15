"""Auditable AST adapter: original scientific training statements kept intact.

Only dataset input availability, preflight subset, epoch persistence/guards,
and completed-SHA checking are injected. This is NOT production preflight PASS.
"""
from __future__ import annotations
import argparse
import ast
import json
import sys
import types
from pathlib import Path
import torch
from epoch_recovery import EpochRecovery, InterruptedForFixture

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent.parent
SOURCE=WORKSPACE/'FRONT_ENCODER_SERIES/train_front_variant.py'

def injected(code):return ast.parse(code).body

class Adapter(ast.NodeTransformer):
    def __init__(self):self.counts={'resume':0,'commit':0,'guard_gradient':0,'subset':0,'completed_sha':0,'finish':0}
    def visit_FunctionDef(self,node):
        if node.name!='run':return node
        node.body=self.transform_body(node.body)
        node.body+=injected('_recovery.finish(locals())');self.counts['finish']+=1
        return node
    def transform_body(self,body):
        result=[]
        for node in body:
            # Before saving the original split, limit only actual preflight rows.
            if isinstance(node,ast.Expr) and ast.unparse(node).startswith('np.savez_compressed(run_dir /'):
                result+=injected("if _subset_rows is not None:\n    train_indices = train_indices[:_subset_rows]\n    val_indices = val_indices[:12]\n    split['preflight_subset_only'] = True\n    split['actual_train_samples'] = len(train_indices)\n    split['actual_validation_samples'] = len(val_indices)")
                self.counts['subset']+=1
            if isinstance(node,ast.If) and ast.unparse(node.test)=='metrics_path.is_file() and (not args.force)':
                node.body=injected('_recovery.skip_compatible(locals())')+node.body;self.counts['completed_sha']+=1
            if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='epoch':
                result+=injected('start_epoch, history, best_ade, best_checkpoint_score = _recovery.begin(locals())')
                node.iter=ast.parse('range(start_epoch, args.epochs)',mode='eval').body
                node.body=self.transform_body(node.body)+injected('_recovery.commit(locals())')
                self.counts['resume']+=1;self.counts['commit']+=1
            elif isinstance(node,(ast.For,ast.If,ast.With)):
                node.body=self.transform_body(node.body)
                if hasattr(node,'orelse'):node.orelse=self.transform_body(node.orelse)
            result.append(node)
            if isinstance(node,ast.Expr) and ast.unparse(node)=='loss.backward()':
                result+=injected('_recovery.observe(locals())');self.counts['guard_gradient']+=1
        return result

def load_adapter(branch,recovery,subset_rows=None):
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
    adapter=Adapter();tree=adapter.visit(tree);ast.fix_missing_locations(tree)
    if adapter.counts!={'resume':1,'commit':1,'guard_gradient':1,'subset':1,'completed_sha':1,'finish':1}:
        raise RuntimeError('STOP_AST_SOURCE_LAYOUT: '+str(adapter.counts))
    module=types.ModuleType('ped_motion_isolated_trainer');sys.modules[module.__name__]=module
    module.__dict__.update(HERE=SOURCE.parent,WORKSPACE=WORKSPACE,_recovery=recovery,_subset_rows=subset_rows)
    exec(compile(tree,str(SOURCE)+'::isolated-input-and-epoch-hooks','exec'),module.__dict__)
    original=module.TrajectoryDataset
    class InputDataset(original):
        def __getitem__(self,index):
            past,future,valid,scene,*empty=super().__getitem__(index)
            if branch=='last2':
                past=past.clone();past[:,:6]=past[:,6:7]
            elif branch=='noNbr':
                past=past.clone();past[1:]=past[:1];valid=torch.ones_like(valid)
            elif branch!='full':raise ValueError(branch)
            return past,future,valid,scene,*empty
    module.TrajectoryDataset=InputDataset
    return module,adapter.counts

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--input-branch',choices=('full','last2','noNbr'),required=True)
    p.add_argument('--source-binding',type=Path,required=True)
    p.add_argument('--resume-epoch',action='store_true');p.add_argument('--fixture-stop-after',type=int)
    p.add_argument('--fixture-subset-rows',type=int);p.add_argument('--production-gate',type=Path)
    extra,original=p.parse_known_args()
    if '--output-root' not in original:raise RuntimeError('Explicit isolated output root required')
    root=Path(original[original.index('--output-root')+1]).resolve()
    if not root.is_relative_to(HERE):raise RuntimeError('Refusing writes outside stage4 directory')
    if '--force' in original:raise RuntimeError('No force-overwrite supported')
    if extra.fixture_subset_rows is None:
        if extra.production_gate is None:raise RuntimeError('Production gates not verified; no launch')
        gate=json.loads(extra.production_gate.read_text(encoding='utf-8'))
        if gate.get('status')!='PASS_ALL_ACTUAL_PREFLIGHT_AND_FINAL_SOURCE':raise RuntimeError('Missing real production preflight')
    else:
        if not 21<=extra.fixture_subset_rows<=513:raise ValueError('Bounded preflight subset required')
        if not root.is_relative_to(HERE/'preflight_runs'):raise RuntimeError('Fixtures must use preflight_runs')
    recovery=EpochRecovery(HERE,extra.input_branch,extra.source_binding,extra.resume_epoch,extra.fixture_stop_after,extra.fixture_subset_rows is not None)
    module,counts=load_adapter(extra.input_branch,recovery,extra.fixture_subset_rows)
    sys.argv=[str(SOURCE),*original]
    try:module.run()
    except InterruptedForFixture as e:print(json.dumps({'intentional_fixture_interruption':str(e)}));return

if __name__=='__main__':main()
