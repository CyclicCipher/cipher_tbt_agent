# Macrocolumn Abstraction Experiments

## Overview

This folder contains experimental implementations of cortical macrocolumn abstractions as computational units, based on neuroscience research showing that macrocolumns implement sparse distributed codes with categorical structure.

## Motivation

**Goal:** Replace individual neurons with macrocolumns (10K neurons each) to match human neocortex scale (150K macrocolumns) on consumer GPUs.

**Budget:** 5K FP16 parameters per macrocolumn = 1.5 GB total for 150K macrocolumns

**Key Principles from Neuroscience:**
1. Sparse distributed codes (2-5% minicolumns active)
2. Minicolumns as winner-take-all units
3. Layer-specific computation (L2/3 codes, L5b sequences, L6 modulation)
4. Sparse connections between macrocolumns (not all-to-all)

## Categorical Structure

Each macrocolumn is a small category:
- **Objects:** Minicolumn activation patterns (sparse)
- **Morphisms:** Pattern transformations (sparse connections)
- **Composition:** Winner-take-all competition + lateral inhibition

This structure dramatically reduces connection count:
- Dense: 100×100 = 10K connections per macrocolumn
- Sparse categorical: ~3 active × 3 targets = 9 connections per pattern

## Limitations

**Applies only to columnar cortex:**
- Primary sensory areas (V1, A1, S1)
- Some association areas

**Does NOT apply to:**
- Prefrontal cortex (non-columnar, abstract processing)
- Olfactory cortex (distributed, no columns)
- Some limbic structures

**Therefore:** Macrocolumn abstraction is part of a hybrid architecture, not the entire network.

## Implementation Status

⚠️ **Experimental** - Do not modify main codebase. All experiments isolated here.

## References

- Sparse Distributed Coding Model: https://www.frontiersin.org/articles/10.3389/fnana.2010.00017/full
- Rapid Processing in Macrocolumns: https://pubmed.ncbi.nlm.nih.gov/15006090/
