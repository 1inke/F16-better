"""Small dependency-free config loader for the locked Q20 experiment."""

from __future__ import annotations

from pathlib import Path

import yaml


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def _convert(value):
    if isinstance(value, dict):
        return Config({key: _convert(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


def load_locked_q20(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as handle:
        cfg = _convert(yaml.safe_load(handle))
    structure = cfg.MODEL.STRUCTURE
    structure.VARIANT = "d1_state_no_motif_matched"
    structure.ENABLED = True
    structure.USE_RWPE = False
    structure.USE_NODE_ENCODING = False
    structure.USE_SELF_STATE = True
    structure.BUILD_SELF_STATE = True
    structure.USE_EDGE_DRWSE = False
    structure.BUILD_EDGE_DRWSE = False
    structure.EDGE_DRWSE_ZERO_CONTROL = False
    structure.USE_RBF = True
    structure.USE_KINEMATIC_BIAS = True
    structure.USE_STRUCTURE_BIAS = False
    structure.PER_HEAD_MOTIF = True
    cfg.MODEL.GRAPH.MODE = "adaptive_kinematic"
    decoder = cfg.MODEL.DECODER
    decoder.VARIANT = "query_endpoint_pca"
    decoder.NUM_MODES = 20
    decoder.PCA_COMPONENTS = 22
    decoder.QUERY_DIM = 96
    decoder.QUERY_HEADS = 4
    decoder.QUERY_FFN_DIM = 192
    decoder.QUERY_LAYERS = 2
    decoder.USE_MODE_SELF_ATTENTION = True
    decoder.USE_RESIDUAL_REFINER = True
    decoder.REFINER_HIDDEN_DIM = 64
    return cfg
