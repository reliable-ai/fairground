---
title: What is FairGround?
---

# What is FairGround?

## Overview

*FairGround* is a unified framework, data corpus, and Python package aimed at advancing reproducible research and critical data studies in fair ML classification. The initial release contains 44 tabular datasets, each annotated with rich fairness-relevant metadata. Our accompanying Python package standardizes dataset loading, preprocessing, transformation, and splitting, streamlining experimental workflows.

## Framework

The FairGround framework is organized into different components. It contains a **Corpus** of different datasets and scenarios (a scenario is a combination of a dataset with a selection of sensitive attributes), as well as multiple **Collections** of scenarios.

We also provide a **Website** (which you're currently on) to make the framework more accessible, as well as a **Python Package**, which operationalizes loading and (pre)processing of datasets. By utilizing the Python Package to load Collections, allows for the creation of reproducible **Evaluation Suites**.

![An illustration of the FairGround framework.](/framework.png)
