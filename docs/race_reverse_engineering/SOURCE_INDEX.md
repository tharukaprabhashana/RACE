# Source Code Reading Index & Navigation Guide

This document provides a structured, chronological reading guide through the source code files of the RACE implementation. It is ordered logically so that a newcomer can trace the system from entry point to core algorithms.

---

## 1. Recommended Reading Order

### Stage 1: Entry Point and Orchestration
Begin here to understand how configuration files are merged and how the training loop is controlled:

1. [`src/main.py`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py)
   - `cpu_num` configuration & PyTorch thread limits: [`lines 16-23`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py#L16-L23)
   - Configuration extraction and hierarchical merging (`_get_config`, `recursive_dict_update`): [`lines 47-70`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py#L47-L70)
   - Sacred experiment observer registration and CLI execution: [`lines 82-107`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py#L82-L107)
   - Random seed assignment & entry call: [`lines 35-44`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py#L35-L44)

2. [`src/run.py`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py)
   - MARL-to-EA policy copy helper (`rl_to_evo`): [`lines 23-26`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L23-L26)
   - Scheme setup for discrete vs continuous environments: [`lines 106-213`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L106-L213)
   - Controller assembly & shared encoder injection (`agent_SR` passed into `pop`): [`lines 219-226`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L219-L226)
   - Learner selection (`EA_facmac_learner_discrete`): [`lines 231-236`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L231-L236)
   - Main episode loop & warmup gating (`start_timesteps = 10000`): [`lines 284-319`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L284-L319)
   - Multi-batch training loop (`repeat_num`): [`lines 322-343`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L322-L343)
   - Policy injection call: [`lines 426-435`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L426-L435)

---

### Stage 2: Controllers and Representation Asymmetry (RATC)
Read these files to understand how policies are split into shared observation representations and linear heads:

3. [`src/controllers/EA_basic_controller.py`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py)
   - `RL_BasicMAC`: Controller for the MARL team ([`lines 7-147`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L7-L147))
     * Instantiation of `agent_SR` and shared `agent_W`: [`lines 97-101`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L97-L101)
     * Forward pass evaluating $Z_\phi$ then $W$: [`lines 33-58`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L33-L58)
   - `Gen_BasicMAC`: Controller for EA candidate teams ([`lines 148-301`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L148-L301))
     * Injected with shared `agent_SR`: [`line 152`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L152)
     * Construction of independent linear heads when `SAME=0`: [`lines 249-257`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L249-L257)
     * Forward pass: [`lines 174-211`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L174-L211) (note `agent_W[0]` indexing quirk at line 193)

4. [`src/modules/agents/EA_qmix_agent.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py)
   - `QMIXRNNAgent_SR`: Shared nonlinear encoder (Linear 42 $\to$ 64, GRUCell 64 $\to$ 64): [`lines 27-48`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py#L27-L48)
   - `QMIXRNNAgent_W`: Linear policy head (Linear 64 $\to$ 9): [`lines 51-68`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py#L51-L68)

---

### Stage 3: Neuro-Evolutionary Engine
Read this file to understand genetic selection, tournament ranking, and low-level mutation:

5. [`src/ea/mod_neuro_evo.py`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py)
   - `SSNE.__init__`: Reading `frac`, `pop_size`, `elite_fraction`: [`lines 7-21`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L7-L21)
   - Tournament selection (`selection_tournament`, $k=3$): [`lines 22-32`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L22-L32)
   - Row-wise weight mutation (`mutate_inplace`, exact consumption of `frac`): [`lines 79-129`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L79-L129)
   - Parameter tensor cloning (`clone`): [`lines 130-135`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L130-L135)
   - Full evolutionary generation step (`epoch`): [`lines 143-207`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L143-L207)
     * Agent-level crossover branch: [`lines 185-189`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L185-L189)

---

### Stage 4: Representation Learning, Critics, and Loss Objectives
Read these files to understand the mathematical updates (VFM, VMM, PeVFA):

6. [`src/learners/EA_facmac_learner_discrete.py`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py)
   - MINE estimator network & Fenchel dual loss: [`lines 79-129`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L79-L129)
   - Optimizer instantiation (`agent_optimiser`, `critic_optimiser`, `PeVFA_optimiser`, `MINE_optimiser`, `V_Net_optimiser`): [`lines 158-204`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L158-L204)
   - PeVFA critic TD update (Step 1): [`lines 221-276`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L221-L276)
   - Conventional MARL critic TD update (Step 2): [`lines 280-322`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L280-L322)
   - State-value net $V_\zeta$ update against $\max(Q_{\psi'}, Q_{\theta'})$ (Step 3): [`lines 327-335`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L327-L335)
   - MARL policy gradient loss (Step 4): [`lines 343-362`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L343-L362)
   - Value-weighted MINE loss & PeVFA actor loss (Step 5): [`lines 366-420`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L366-L420)
   - Combined representation loss & joint gradient step (Step 6): [`lines 425-434`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L425-L434)
   - Target network update synchronization: [`lines 438-446`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L438-L446)

7. [`src/modules/critics/facmac.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py)
   - `FACMACDiscreteCritic`: Standard MARL critic: [`lines 38-68`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py#L38-L68)
   - `PeVFA_FACMACDiscreteCritic`: Policy-extended critic (MLP embedding of policy weights): [`lines 71-126`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py#L71-L126)

8. [`src/modules/mixers/qmix.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py)
   - `V_Net`: State-value network $V_\zeta(s)$: [`lines 6-28`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py#L6-L28)
   - `QMixer`: Standard monotonic mixer: [`lines 31-103`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py#L31-L103)
   - `PeVFA_QMixer`: Policy-extended mixer factoring joint value from state and policy embedding: [`lines 106-207`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py#L106-L207)

---

### Stage 5: Runners, Action Selectors, and Buffers
Read these files to understand environment stepping, data collection, and action distributions:

9. [`src/runners/episode_runner.py`](file:///home/tharuka/Documents/GitHub/RACE/src/runners/episode_runner.py)
   - Episodic rollout loop, action forwarding, and `t_env` tracking: [`lines 56-150`](file:///home/tharuka/Documents/GitHub/RACE/src/runners/episode_runner.py#L56-L150)
   - Action exploration flag handling (`explore=False` for EA, `explore=True` for MARL): [`lines 75-82`](file:///home/tharuka/Documents/GitHub/RACE/src/runners/episode_runner.py#L75-L82)

10. [`src/components/action_selectors.py`](file:///home/tharuka/Documents/GitHub/RACE/src/components/action_selectors.py)
    - `GumbelSoftmaxMultinomialActionSelector`: Action selection via Gumbel-Softmax reparameterization or categorical exploration: [`lines 63-101`](file:///home/tharuka/Documents/GitHub/RACE/src/components/action_selectors.py#L63-L101)

11. [`src/components/episode_buffer.py`](file:///home/tharuka/Documents/GitHub/RACE/src/components/episode_buffer.py)
    - `EpisodeBatch`: Multidimensional tensor container: [`lines 6-264`](file:///home/tharuka/Documents/GitHub/RACE/src/components/episode_buffer.py#L6-L264)
    - `ReplayBuffer`: Capacity management and uniform batch sampling: [`lines 472-517`](file:///home/tharuka/Documents/GitHub/RACE/src/components/episode_buffer.py#L472-L517)
