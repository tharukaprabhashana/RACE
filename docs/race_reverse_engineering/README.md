# RACE MARL Implementation: Reverse-Engineering & Technical Reference

This directory contains a complete, source-grounded reverse-engineering analysis of the **RACE** (**Representation Asymmetry and Collaborative Evolution**) multi-agent reinforcement learning framework.

The reference paper for this codebase is:
> **RACE: Improve Multi-Agent Reinforcement Learning with Representation Asymmetry and Collaborative Evolution**  
> Pengyi Li, Jianye Hao, Hongyao Tang, Yan Zheng, Xian Fu  
> *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, PMLR 202:19490–19503, 2023.  
> Official PDF: [`Race.pdf`](file:///home/tharuka/Documents/GitHub/RACE/Race.pdf).

---

## 1. Purpose of This Technical Guide

This guide is designed for an advanced undergraduate or master's-level artificial intelligence student (e.g., BSc Hons AI, final year) preparing to:
1. Explain the inner workings of RACE academically in an oral defense, literature review, or seminar.
2. Formulate, justify, and implement novel research extensions (such as adaptive diversity mechanisms, population schedules, or alternative representation estimators) without guessing how the underlying codebase operates.

This audit is **strictly read-only and source-verified**. Every technical claim, data path, loss term, and parameter role is tied directly to verified file paths and line numbers in the repository.

---

## 2. Learning Roadmap

To build a thorough understanding, read the documents in the following sequence:

| Step | Document | Key Focus |
|:---:|:---|:---|
| **0** | [**Glossary & Core Concepts**](glossary.md) | Terminological definitions (Dec-POMDP, CTDE, RATC, PeVFA, VFM, VMM, MINE, SSNE). |
| **1** | [**01 — Repository Structure & Execution Entry Points**](01_repository_and_execution.md) | Code organization, CLI invocation, configuration layering/precedence, Sacred setup. |
| **2** | [**02 — The RACE Algorithm Explained**](02_race_algorithm_explained.md) | Plain-language theory, architectural decomposition, and Paper vs. Implementation audit. |
| **3** | [**03 — Comprehensive Source Code Map**](03_source_code_map.md) | Component-by-component index of classes, responsibilities, callers, and callees. |
| **4** | [**04 — Training Iteration Step-by-Step**](04_training_iteration.md) | Execution timeline of one iteration: rollout, evolution, critic, V-Net, representation updates. |
| **5** | [**05 — Parameter Analysis & Deep Trace of `frac`**](05_parameters_and_frac_trace.md) | Full hyperparameter dictionary and exact trace of `frac` down to row-wise weight mutation. |
| **6** | [**06 — Data, Network, and Gradient Flow**](06_data_and_network_flow.md) | Observation/state shapes, network topologies, PeVFA embeddings, and backpropagation paths. |
| **7** | [**07 — Feasibility Run Analysis & Logging**](07_experiment_and_logging.md) | Dissection of the 10,000-step test, output artifacts, Sacred metrics, and boundary conditions. |
| **8** | [**08 — Uncertainties, Discrepancies & Research Checklist**](08_uncertainties_and_followup.md) | Catalog of code bugs, parameter quirks, unverified behaviors, and safe next experiments. |
| **9** | [**Mermaid Architecture & Sequence Diagrams**](diagrams.md) | Diagrams A through H rendering all dataflows, lifecycles, and network interconnections. |
| **10** | [**Source Code Index**](SOURCE_INDEX.md) | Direct reading checklist for navigating the source files efficiently. |

---

## 3. High-Level Verification Summary

### What Has Been Verified in Source
* **Representation Asymmetry Team Construction (RATC):**  
  Verified in [`src/controllers/EA_basic_controller.py:7-301`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L7-L301) and [`src/modules/agents/EA_qmix_agent.py:27-68`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py#L27-L68). Policies decompose into a nonlinear observation encoder (`QMIXRNNAgent_SR`, parameters $\phi$) shared across agents and teams, and linear policy heads (`QMIXRNNAgent_W`, parameters $W_j^i$).
* **Policy-Extended Value Function Approximator (PeVFA):**  
  Verified in [`src/modules/critics/facmac.py:71-126`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py#L71-L126) and [`src/modules/mixers/qmix.py:106-207`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py#L106-L207). Policy weights $W_j$ are embedded via a 3-layer MLP and injected into both decentralized critics and the centralized mixing network.
* **Value Function Maximization (VFM):**  
  Verified in [`src/learners/EA_facmac_learner_discrete.py:221-362, 404-425`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L221-L362). The shared encoder is optimized through the joint value gradients of the MARL team (critic $Q_\psi$) and a randomly sampled EA team (PeVFA $Q_\theta$).
* **Value-Aware Mutual Information Maximization (VMM):**  
  Verified in [`src/learners/EA_facmac_learner_discrete.py:110-129, 327-335, 366-402`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L110-L129). Mutual Information Neural Estimation (MINE) lower bound is computed between representations $Z$ and global state $s$, weighted by min-max normalized state values $V_\zeta(s)$ fitted to $\max(Q_\psi, Q_\theta)$.
* **Agent-Level Evolutionary Operators:**  
  Verified in [`src/ea/mod_neuro_evo.py:42-207`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L42-L207). Tournament selection (size 3), 1 elite team, whole-agent linear policy swaps for crossover, and row-wise Gaussian/reset perturbations for mutation.
* **MARL-to-EA Policy Injection:**  
  Verified in [`src/run.py:23-26, 426-435`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L23-L26). MARL linear policy parameters are copied directly into the worst-performing non-elite EA team.

### Key Discrepancies & Code Nuances Uncovered
1. **The 10k Feasibility Test Warm-Up Gate:**  
   In [`src/run.py:289, 315, 426`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L289), EA population rollout, evolution, multi-batch training (`repeat_num = 6`), and policy synchronization are gated by `runner.t_env > args.start_timesteps` (`start_timesteps = 10000`). Consequently, any run with `t_max = 10000` is purely a MARL warmup; evolutionary operators never fire during that initial budget.
2. **`Gen_BasicMAC.forward` Agent Representation Indexing:**  
   In [`src/controllers/EA_basic_controller.py:193`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L193), when `SAME=0` (independent agent policies), the forward loop calls `self.agent_W[0]` instead of `self.agent_W[i]`. While independent weights are constructed, evolved, and fed to PeVFA, action evaluation in rollout inadvertently shares agent 0's policy representation across all agents.
3. **`frac` Integer Truncation in Discrete Spaces:**  
   In [`src/ea/mod_neuro_evo.py:114`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L114), the number of mutated weights per action row is `int(W.shape[1] * self.frac)`. With `W.shape[1] = 64` and `frac = 0.005`, `int(64 * 0.005) = int(0.32) = 0`, causing mutation to alter zero weights per row unless `frac >= 1/64` ($\approx 0.0156$).
4. **Unused Configuration Parameters:**  
   `rl_to_ea_synch_period` is declared in `src/config/default.yaml:26` but never referenced in Python code; synchronization is governed exclusively by `EA_freq`. Furthermore, `src/run.py:222` hardcodes `range(5)` when instantiating EA population controllers, bypassing `args.pop_size`.

---

## 4. Verification Methodology & Safety Confirmation

In compliance with the project instructions:
* **No algorithm implementation files were modified.** Source code remains untouched.
* **No full training runs, SMAC battles, or GPU workloads were triggered.**
* **All findings are grounded in static inspection and read-only analysis of past feasibility outputs.**
* **Confidential environment variables (`.env`) are kept private and unexposed.**
