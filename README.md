# Federated Generation of Local and Global Explanations via Decision Tree Merging

This repository contains the reference implementation of a framework that integrates **local and global explainability** within a **federated learning** environment.  
The method combines local rule-based explanations with a **global surrogate model** constructed through a **decision tree merging** process, allowing both instance-level and global interpretability while preserving data privacy.

---

## Getting Started

All code is implemented in **Python**.


## Overview

Each federated client trains:

* A private **black-box model** (e.g., a neural network).

* A **local surrogate decision tree** used for interpretability.

The central server merges the local trees into a single global surrogate (SuperTree) using a recursive merging algorithm.
This global model represents the collective decision logic learned by all clients while maintaining data confidentiality.

The framework supports mixed numerical and categorical tabular datasets and allows rule-based explanations that can be both factual and counterfactual.

## Implementation Notes

* All federated operations are managed by a client–server architecture based on the Flower framework.

* Only symbolic decision tree structures are exchanged — no raw data are shared.

* The merging process preserves privacy and enables transparent, auditable global explanations.

## Usage

This repository accompanies the manuscript submitted to IEEE CAI 2026.
In compliance with the double-blind review policy, all identifying information, datasets, and runnable notebooks are intentionally omitted.
Full code and experimental notebooks will be made available upon acceptance.

## Citation

If this work is published, please cite:

Federated Generation of Local and Global Explanations via Decision Tree Merging, IEEE CAI 2026 (under review).

