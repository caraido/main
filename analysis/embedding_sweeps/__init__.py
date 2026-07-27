# -*- coding: utf-8 -*-
"""Embedding model layer/variant sweeps.

  visual_layer_sweep
      Per-layer regression for DINOv2 (13 ViT layers) and SimCLR (5 stages).
      Tests whether intermediate visual representations predict neural HGA
      better than the final pooled layer.
"""
