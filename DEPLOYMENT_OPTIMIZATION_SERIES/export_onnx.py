"""Fixed-shape F16 deployment wrappers and ONNX export for EXP-180B."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from onnxconverter_common import float16
from torch import nn


SERIES = Path(__file__).resolve().parent
ROOT = SERIES.parent
EFFICIENCY = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "run_exp174a_final_efficiency.py"
SDD_HELPER = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175A_SCENE_UTILITY_AUDIT" / "run_exp175a_scene_utility_audit.py"
ETH_DATA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse")
SDD_DATA = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


class TrajectoryDeploymentWrapper(nn.Module):
    """Raw observed coordinates -> global candidates and native logits."""

    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, absolute_past: torch.Tensor, num_valid: torch.Tensor):
        lag = 3
        displacement = absolute_past[:, 0, -1] - absolute_past[:, 0, -1 - lag]
        speed = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        default = torch.zeros_like(displacement); default[:, 0] = 1.0
        heading = torch.where(speed > 0.03, displacement / torch.clamp(speed, min=1e-6), default)
        cosine, sine = heading[:, 0], heading[:, 1]
        matrix = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(-1, 1, 1, 2, 2)
        origin = absolute_past[:, :1, -1:]
        past = torch.matmul((absolute_past - origin).unsqueeze(-2), matrix).squeeze(-2)
        center = past.mean(dim=-2, keepdim=True)
        output = self.model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
        raw = output[2] if isinstance(output, (tuple, list)) else output
        local = (raw + center.unsqueeze(-3))[:, 0]
        c = cosine[:, None, None]; s = sine[:, None, None]
        x = local[..., 0] * c - local[..., 1] * s
        y = local[..., 0] * s + local[..., 1] * c
        prediction = torch.stack((x, y), dim=-1) + origin[:, 0, 0, None, None]
        return prediction, self.model.last_mode_logits[:, 0]


def build_profile(profile: str, device: torch.device):
    if profile == "eth_n9":
        helper = load_module("exp180b_efficiency", EFFICIENCY)
        model, cfg, trainer = helper.build_model("F16", 1, device, 9)
        dataset = trainer.TrajectoryDataset(ETH_DATA / "eth_data_test.npy", ETH_DATA / "eth_num_test.npy", cfg)
        agents = 9
    elif profile == "sdd_n16":
        helper = load_module("exp180b_sdd", SDD_HELPER)
        model, cfg, trainer = helper.build_model(1, device)
        dataset = trainer.TrajectoryDataset(
            SDD_DATA / "sdd_standard_test_data.npy", SDD_DATA / "sdd_standard_test_num.npy", cfg
        )
        agents = 16
    else:
        raise ValueError(profile)
    return TrajectoryDeploymentWrapper(model).to(device).eval(), dataset, agents


def sample_batch(dataset, device):
    item = dataset[0]; past, future, valid = item[0], item[1], item[2]
    if not torch.is_tensor(past): past = torch.as_tensor(past)
    if not torch.is_tensor(future): future = torch.as_tensor(future)
    if not torch.is_tensor(valid): valid = torch.as_tensor(valid)
    return past[None].float().to(device), future[None].float().to(device), valid.reshape(1).long().to(device)


def fold_fixed_eyelike(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace unsupported EyeLike nodes after fixed-shape export."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {}; element_types = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.HasField("dim_value") else None)
        shapes[value.name] = dims
        element_types[value.name] = int(tensor_type.elem_type)

    rewritten = []
    for node in model.graph.node:
        if node.op_type != "EyeLike":
            rewritten.append(node)
            continue
        shape = shapes.get(node.output[0]) or shapes.get(node.input[0])
        if not shape or len(shape) != 2 or any(d is None or d <= 0 for d in shape):
            raise RuntimeError(f"Cannot constant-fold {node.name}: inferred shape={shape}")
        attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        element_type = int(attrs.get("dtype", element_types.get(node.input[0], onnx.TensorProto.FLOAT)))
        dtype = onnx.helper.tensor_dtype_to_np_dtype(element_type)
        value = np.eye(shape[0], shape[1], k=int(attrs.get("k", 0)), dtype=dtype)
        rewritten.append(helper.make_node(
            "Constant", [], list(node.output), name=f"{node.name}_fixed_constant",
            value=numpy_helper.from_array(value),
        ))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return model


def stabilize_exported_atan2(model: onnx.ModelProto) -> onnx.ModelProto:
    """Make the legacy atan2 decomposition agree with torch.atan2(0, 0)=0."""
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    rewritten = []
    for node in model.graph.node:
        if node.op_type != "Div" or not any(x.op_type == "Atan" for x in consumers.get(node.output[0], [])):
            rewritten.append(node); continue
        denominator = node.input[1]
        stem = (node.name or node.output[0]).replace("/", "_")
        abs_name = f"{stem}_safe_abs"; threshold_name = f"{stem}_safe_threshold"
        small_name = f"{stem}_safe_small"; safe_name = f"{stem}_safe_denominator"
        rewritten.extend([
            helper.make_node("Abs", [denominator], [abs_name], name=f"{stem}_Abs"),
            helper.make_node(
                "Constant", [], [threshold_name], name=f"{stem}_Epsilon",
                value=numpy_helper.from_array(np.asarray(1.0e-7, dtype=np.float32)),
            ),
            helper.make_node("Less", [abs_name, threshold_name], [small_name], name=f"{stem}_NearZero"),
            helper.make_node("Where", [small_name, threshold_name, denominator], [safe_name], name=f"{stem}_SafeDenominator"),
        ])
        node.input[1] = safe_name
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return model


def reconcile_float16_casts(model: onnx.ModelProto) -> onnx.ModelProto:
    """Repair stale Cast-to-FP32 attributes left by the generic FP16 converter."""
    declared = {}
    for value in list(model.graph.value_info) + list(model.graph.output):
        if value.type.HasField("tensor_type"):
            declared[value.name] = int(value.type.tensor_type.elem_type)
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        target = declared.get(node.output[0])
        if target != onnx.TensorProto.FLOAT16:
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and int(attribute.i) == onnx.TensorProto.FLOAT:
                attribute.i = onnx.TensorProto.FLOAT16
    return model


def export_profile(profile: str, output_dir: Path, device: torch.device, force: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / f"{profile}_fp32.onnx"; fp16_path = output_dir / f"{profile}_fp16.onnx"
    wrapper, dataset, agents = build_profile(profile, device)
    past, _, valid = sample_batch(dataset, device)
    with torch.inference_mode():
        expected = wrapper(past, valid)
    if force or not fp32_path.is_file():
        torch.onnx.export(
            wrapper, (past, valid), fp32_path, input_names=["past", "num_valid"],
            output_names=["candidates", "logits"], opset_version=17,
            do_constant_folding=True, dynamo=False,
        )
        model = stabilize_exported_atan2(fold_fixed_eyelike(onnx.load(fp32_path)))
        onnx.checker.check_model(model); onnx.save(model, fp32_path)
    if force or not fp16_path.is_file():
        model = onnx.load(fp32_path)
        converted = float16.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=False,
            op_block_list=["TopK", "NonZero", "Range"],
        )
        converted = reconcile_float16_casts(converted)
        onnx.checker.check_model(converted); onnx.save(converted, fp16_path)
    return {
        "profile": profile, "agents": agents, "fp32": fp32_path, "fp16": fp16_path,
        "parameters": sum(p.numel() for p in wrapper.parameters()),
        "pytorch_probe": [x.detach().cpu().numpy() for x in expected],
    }
