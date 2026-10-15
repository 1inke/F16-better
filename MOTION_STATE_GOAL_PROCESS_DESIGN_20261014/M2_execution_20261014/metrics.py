"""Unchanged registered quality and pooled motion; isolated M2 output context."""
import common as c
v=c.module("m2_readonly_M1_metrics",c.M1/"metrics.py")
masks=v.masks;errors=v.errors;motion=v.motion;sample=v.sample;clusters=v.clusters;evaluate=v.evaluate
