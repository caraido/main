# -*- coding: utf-8 -*-
"""Tests probing whether phoneme decoding picks up genuine phonological
information or merely reflects semantic co-variance in the neural signal.

  Test 1: cross_category_generalization
      Does phoneme decoding generalize across semantic categories?

  Test 2: semantic_residual_regression
      Does phoneme decoding survive after removing semantic neural dimensions?

  Test 3: partial_rsa
      Partial RSA: what fraction of neural prediction geometry is phonemic
      after controlling for semantic confounds?

  Test 4: subspace_angle_analysis
      Do phonological and semantic PLS subspaces overlap in neural space?

  Test A: ensemble_retrieval
      Ensemble retrieval with a learned mixing weight alpha.

  Test B: banded_ridge_encoding
      Banded ridge encoding -- predict neural activity from sem+phon
      embeddings simultaneously.

  Test C: commonality_analysis
      Commonality analysis on retrieval-relevant variance.

  Test D: joint_embedding_pls
      Concatenated (joint) embedding target PLS.
"""
