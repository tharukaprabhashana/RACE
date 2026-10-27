# 08 — Discrepancies, Uncertainties, and Follow-Up Inspection Checklist

This document catalogs code discrepancies, implementation bugs, unverified behaviors, and open questions identified during the source code audit of RACE. Each item is assigned a verification status (`verified`, `likely`, or `unresolved`) with supporting evidence and recommended next inspection steps.

---

## 1. Discrepancy and Uncertainty Register

| ID | Issue / Topic | Category | Status | Code Evidence | Impact & Analysis | Suggested Next Inspection Step |
|:---:|:---|:---|:---:|:---|:---|:---|
| **D-01** | **EA Warmup Boundary Blocks Evolution in Short Runs** | Algorithm Scheduling | **Verified** | [`src/run.py:289, 315, 426`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L289) | Population rollout, evolution, `repeat_num=6`, and RL-to-EA sync require `t_env > 10000`. In any test with `t_max <= 10000`, the EA never executes. | Test with `start_timesteps=1000` and `t_max=3000` to verify evolutionary operators fire. |
| **D-02** | **`Gen_BasicMAC.forward` Hardcodes `agent_W[0]`** | Implementation Bug | **Verified** | [`src/controllers/EA_basic_controller.py:193`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L193) | When `SAME=0`, `agent_W` creates independent policy heads for each agent, but the forward pass iterates calling `agent_W[0]` for all agents. EA agents unintentionally share agent 0's policy weights during rollout. | Inspect output of `agent_outs` in a quick print test comparing agent 0 and agent 1 logits. |
| **D-03** | **`frac` Truncates to Zero for Small Values in Discrete SMAC** | Parameter Implementation | **Verified** | [`src/ea/mod_neuro_evo.py:114`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L114) | `int(W.shape[1] * self.frac)` with `W.shape[1]=64` and `frac=0.005` evaluates to `int(0.32) = 0`. Causes zero weights to be mutated, running EA as crossover-only. | Test whether changing `int(...)` to `math.ceil(...)` alters parameter values after `mutate_inplace`. |
| **D-04** | **`pop_size` Hardcoded to 5 in Runner** | Configuration Discrepancy | **Verified** | [`src/run.py:222`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L222) | `for i in range(5): pop.append(...)` hardcodes 5 controllers. Changing `pop_size` in YAML config does not change the number of initialized EA controllers. | Verify if setting `pop_size=10` crashes in `evolver.epoch` due to size mismatch. |
| **D-05** | **`rl_to_ea_synch_period` Parameter is Unused** | Dead Configuration | **Verified** | [`src/config/default.yaml:26`](file:///home/tharuka/Documents/GitHub/RACE/src/config/default.yaml#L26), [`src/run.py:426`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L426) | Present in `default.yaml` but never read in Python source. Policy synchronization is triggered exclusively by `episode % args.EA_freq == 0`. | Remove or deprecate `rl_to_ea_synch_period` in future configs to avoid user confusion. |
| **D-06** | **PyTorch Adam Duplicate Parameter Warning** | Optimizer Setup | **Verified** | [`src/learners/EA_facmac_learner_discrete.py:165, 171`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L165), [`feasibility_results/20261027_165005/training.log:160`](file:///home/tharuka/Documents/GitHub/RACE/feasibility_results/20261027_165005/training.log#L160) | `PeVFA_params` includes `PeVFA_critic.parameters()` and `PeVFA_mixer.parameters()`. If parameter lists overlap, PyTorch warns of duplicate parameters. | Inspect `PeVFA_params` list uniqueness via `set(p.data_ptr() for p in self.PeVFA_params)`. |
| **D-07** | **Optional MuJoCo Import Warning in SMAC Runs** | Dependency Handling | **Verified** | [`src/envs/__init__.py:7-11`](file:///home/tharuka/Documents/GitHub/RACE/src/envs/__init__.py#L7-L11), [`feasibility_results/20261027_165005/training.log:2`](file:///home/tharuka/Documents/GitHub/RACE/feasibility_results/20261027_165005/training.log#L2) | Importing `envs` attempts to import `.mamujoco`. If missing `mujoco200`, it prints a missing-file notice but safely falls back to `MUJOCO_AVAILABLE = False`. | Harmless in SMAC experiments; document as expected non-fatal stdout. |
| **D-08** | **Serial Execution Only (No Parallelism)** | Performance Bottleneck | **Verified** | [`src/runners/episode_runner.py:16-17`](file:///home/tharuka/Documents/GitHub/RACE/src/runners/episode_runner.py#L16-L17) | `EpisodeRunner` asserts `batch_size_run == 1`. Neither environment rollouts nor EA team evaluations use Python `multiprocessing`. | Benchmark wall-clock rollout time per generation before scaling population size. |
| **D-09** | **WandB Disabled by Default in Code** | Telemetry Divergence | **Verified** | [`src/run.py:91-93`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L91-L93) | Hardcoded `os.environ["WANDB_MODE"] = "disabled"`. The README states WandB is used, but code disables it unless modified. | Re-enable WandB mode via environment variable or config flag if online dashboards are needed. |
| **D-10** | **Paper Algorithm 1 vs Code Ordering** | Algorithmic Structure | **Likely** | [`src/run.py:306-340`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L306-L340) | Paper lists critic update before EA evolution. Code executes EA evolution immediately following rollout, then trains critics and representations. | Verify whether performing evolution before or after gradient steps impacts convergence. |

---

## 2. Questions to Investigate Before Designing Any Research Extension

Before designing or implementing research modifications (e.g., adaptive diversity, dynamic mutation schedules, or novel credit assignment schemes), a student should investigate the following empirical and architectural questions:

### 1. Evolutionary Mutation Reality in SMAC
* **Question:** Since `frac = 0.005` leads to `int(64 * 0.005) = 0`, is the reported performance gain of RACE on SMAC (`3s_vs_3z`, `3s5z`) attributable solely to **agent-level crossover** and **representation sharing**, or did the authors use a continuous representation or larger `frac` during actual experiment runs?
* **Investigation:** Run an ablation comparing `frac = 0.0` (pure crossover), `frac = 0.005` (current code), and `frac = 0.05` (where `int(64 * 0.05) = 3` weights are mutated per action).

### 2. Impact of the `agent_W[0]` Indexing Discrepancy
* **Question:** In [`src/controllers/EA_basic_controller.py:193`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L193), does calling `self.agent_W[0]` degrade team coordination in heterogeneous or position-dependent scenarios?
* **Investigation:** Correct line 193 to `self.agent_W[i]` and compare test win rates on asymmetric or diverse unit maps.

### 3. Population Scaling & Computation Cost
* **Question:** Because execution is strictly serial (`batch_size_run = 1`, `cpu_num = 1`), how does wall-clock training time scale as population size $n$ increases from 5 to 10 or 20?
* **Investigation:** Profile the time spent in `EpisodeRunner.run()` across 5 EA teams versus time spent inside `learner.train()` across 6 batches.

### 4. Mutual Information Weighting Stability
* **Question:** How sensitive is the training stability to the min-max normalization of $V_\zeta(s)$ in [`src/learners/EA_facmac_learner_discrete.py:371`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L371)? Does batch-level min-max normalization cause gradient variance when sample batches contain narrow return ranges?
* **Investigation:** Monitor the running standard deviation of $V_\zeta$ weights during long training runs.

---

## 3. Safe Inspection Checklist

To inspect or test these items without altering the codebase:
- [x] Static audit of all controllers, modules, and learners completed.
- [x] Discrepancies cross-referenced against paper text and mathematical formulations.
- [x] Zero source files edited.
- [x] Workspace git status verified clean.
