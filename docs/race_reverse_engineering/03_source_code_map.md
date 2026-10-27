# 03 — Comprehensive Source Code Map

This document maps all key architectural components of the RACE codebase, documenting classes, functions, files, execution responsibilities, call hierarchies, and supporting line numbers.

---

## 1. Master Component Table

| Component | Important Files | Key Classes / Functions | Primary Responsibility | Called By / Calls Into | Source Evidence |
|:---|:---|:---|:---|:---|:---|
| **Entry Point & Sacred Setup** | [`src/main.py`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py) | `my_main()`, `_get_config()`, `recursive_dict_update()` | Enforces single-thread limits, loads configuration hierarchy, registers Sacred FileStorageObserver, and boots framework. | **Called by:** CLI invocation.<br/>**Calls into:** `src/run.py:run()`. | [`src/main.py:16-107`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py#L16-L107) |
| **Execution Orchestrator** | [`src/run.py`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py) | `run()`, `run_sequential()`, `rl_to_evo()` | Coordinates experiment lifecycle: instantiates runner, buffer, controllers, and learner; executes episode loop, evolution, and periodic evaluation. | **Called by:** `src/main.py:my_main()`.<br/>**Calls into:** `EpisodeRunner`, `ReplayBuffer`, `RL_BasicMAC`, `Gen_BasicMAC`, `EA_FACMACDiscreteLearner`, `SSNE`. | [`src/run.py:28-459`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L28-L459) |
| **MARL Controller** | [`src/controllers/EA_basic_controller.py`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py) | `RL_BasicMAC` | Manages MARL team: owns shared representation encoder `agent_SR` and MARL linear policy heads `agent_W`; executes Gumbel action selection. | **Called by:** `src/run.py:run_sequential()`, `EpisodeRunner.run()`.<br/>**Calls into:** `QMIXRNNAgent_SR`, `QMIXRNNAgent_W`, `GumbelSoftmaxMultinomialActionSelector`. | [`src/controllers/EA_basic_controller.py:7-147`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L7-L147) |
| **EA Population Controller** | [`src/controllers/EA_basic_controller.py`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py) | `Gen_BasicMAC` | Represents one candidate team in the EA population; shares `agent_SR` with MARL team but maintains independent linear policy representations `agent_W`. | **Called by:** `src/run.py:run_sequential()`, `EpisodeRunner.run()`.<br/>**Calls into:** Injected `agent_SR`, team-specific `agent_W`. | [`src/controllers/EA_basic_controller.py:148-301`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L148-L301) |
| **Shared Representation Encoder** | [`src/modules/agents/EA_qmix_agent.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py) | `QMIXRNNAgent_SR` | Computes nonlinear observation embedding $Z_\phi(o^i)$ using a Linear layer (input to 64) and GRUCell (64 to 64). | **Called by:** `RL_BasicMAC.forward()`, `Gen_BasicMAC.forward()`. | [`src/modules/agents/EA_qmix_agent.py:27-48`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py#L27-L48) |
| **Linear Policy Representation** | [`src/modules/agents/EA_qmix_agent.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py) | `QMIXRNNAgent_W` | Maps 64-dimensional latent representation $Z$ linearly to action logits ($|\mathcal{U}|$). | **Called by:** `RL_BasicMAC.forward()`, `Gen_BasicMAC.forward()`. | [`src/modules/agents/EA_qmix_agent.py:51-68`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py#L51-L68) |
| **Conventional Critic** | [`src/modules/critics/facmac.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py) | `FACMACDiscreteCritic` | Computes agent-level utility values $Q_i(o^i, u^i)$ for the MARL team actions. | **Called by:** `EA_FACMACDiscreteLearner.train()`. | [`src/modules/critics/facmac.py:38-68`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py#L38-L68) |
| **PeVFA Critic** | [`src/modules/critics/facmac.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py) | `PeVFA_FACMACDiscreteCritic` | Evaluates EA candidate policies by embedding flat policy parameters $W_j$ through a 3-layer MLP and conditioning agent utility estimates. | **Called by:** `EA_FACMACDiscreteLearner.train()`. | [`src/modules/critics/facmac.py:71-126`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py#L71-L126) |
| **Conventional Mixer** | [`src/modules/mixers/qmix.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py) | `QMixer` | Monotonic hypernetwork factorizing joint action-value $Q_{tot}$ from local MARL utilities conditioned on global state $s$. | **Called by:** `EA_FACMACDiscreteLearner.train()`. | [`src/modules/mixers/qmix.py:31-103`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py#L31-L103) |
| **PeVFA Mixer** | [`src/modules/mixers/qmix.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py) | `PeVFA_QMixer` | Policy-extended mixer factoring $Q_{tot}$ for EA candidate teams, conditioned on both global state $s$ and policy representation embedding $e_{W_j}$. | **Called by:** `EA_FACMACDiscreteLearner.train()`. | [`src/modules/mixers/qmix.py:106-207`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py#L106-L207) |
| **State-Value Network** | [`src/modules/mixers/qmix.py`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py) | `V_Net` | Estimates maximum expected return $V_\zeta(s)$ across teams to generate state weights for Value-Aware Mutual Information (VMM). | **Called by:** `EA_FACMACDiscreteLearner.train()`. | [`src/modules/mixers/qmix.py:6-28`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py#L6-L28) |
| **Mutual Information Estimator** | [`src/learners/EA_facmac_learner_discrete.py`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py) | `MINE`, `fenchel_dual_loss()` | Computes Fenchel dual Jensen-Shannon lower bound on $I(Z; s)$ between latent observation representations and the global state. | **Called by:** `EA_FACMACDiscreteLearner.train()`. | [`src/learners/EA_facmac_learner_discrete.py:79-129`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L79-L129) |
| **RACE Discrete Learner** | [`src/learners/EA_facmac_learner_discrete.py`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py) | `EA_FACMACDiscreteLearner` | Orchestrates gradient updates: trains PeVFA critic/mixer, MARL critic/mixer, $V_\zeta$ net, and computes combined VFM + VMM loss for $Z_\phi$ and MARL $W$. | **Called by:** `src/run.py:run_sequential()`.<br/>**Calls into:** All critics, mixers, `V_Net`, and `MINE`. | [`src/learners/EA_facmac_learner_discrete.py:132-456`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L132-L456) |
| **Evolutionary Algorithm Engine** | [`src/ea/mod_neuro_evo.py`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py) | `SSNE` | Implements Steady-State Neuro-Evolution: ranking, tournament selection ($k=3$), agent-level crossover, and row-wise parameter mutation. | **Called by:** `src/run.py:run_sequential()`.<br/>**Calls into:** `clone()`, `mutate_inplace()`, `selection_tournament()`. | [`src/ea/mod_neuro_evo.py:7-208`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L7-L208) |
| **Environment Runner** | [`src/runners/episode_runner.py`](file:///home/tharuka/Documents/GitHub/RACE/src/runners/episode_runner.py) | `EpisodeRunner` | Drives sequential episode rollout with SMAC environment; tracks episodic statistics, win rates, and environment step counter `t_env`. | **Called by:** `src/run.py:run_sequential()`.<br/>**Calls into:** `smac.env.StarCraft2Env`, `BasicMAC.select_actions()`. | [`src/runners/episode_runner.py:11-160`](file:///home/tharuka/Documents/GitHub/RACE/src/runners/episode_runner.py#L11-L160) |
| **Experience Replay Buffer** | [`src/components/episode_buffer.py`](file:///home/tharuka/Documents/GitHub/RACE/src/components/episode_buffer.py) | `ReplayBuffer`, `EpisodeBatch` | Stores full episodic transitions collected by both MARL and EA teams; performs uniform batch sampling for off-policy training. | **Called by:** `src/run.py:run_sequential()`. | [`src/components/episode_buffer.py:6-264, 472-517`](file:///home/tharuka/Documents/GitHub/RACE/src/components/episode_buffer.py#L6-L264) |
| **Action Selector** | [`src/components/action_selectors.py`](file:///home/tharuka/Documents/GitHub/RACE/src/components/action_selectors.py) | `GumbelSoftmaxMultinomialActionSelector` | Evaluates action probabilities using Gumbel-Softmax reparameterization or epsilon-greedy categorical exploration. | **Called by:** `BasicMAC.select_actions()`. | [`src/components/action_selectors.py:63-101`](file:///home/tharuka/Documents/GitHub/RACE/src/components/action_selectors.py#L63-L101) |
| **Logging & Telemetry** | [`src/utils/logging.py`](file:///home/tharuka/Documents/GitHub/RACE/src/utils/logging.py) | `Logger` | Aggregates time-series training statistics, manages windowed metric averaging, and synchronizes data with Sacred observers. | **Called by:** `src/run.py`, `EpisodeRunner`, `EA_FACMACDiscreteLearner`. | [`src/utils/logging.py:6-65`](file:///home/tharuka/Documents/GitHub/RACE/src/utils/logging.py#L6-L65) |

---

## 2. Key Call Hierarchies & Inter-Module Connections

### 2.1. Initialization & Assembly Path
```text
src/main.py:__main__
 ├── _get_config("--env-config", "envs")        --> loads src/config/envs/sc2.yaml
 ├── _get_config("--config", "algs")            --> loads src/config/algs/facmac_smac.yaml
 ├── recursive_dict_update()                    --> hierarchical merge over default.yaml
 ├── ex.observers.append(FileStorageObserver)   --> attaches results/sacred observer
 └── ex.run_commandline(params)
      └── my_main()                             --> sets random seeds
           └── src/run.py:run()
                └── run_sequential()
                     ├── r_REGISTRY["episode"]()                 --> instantiates EpisodeRunner
                     ├── ReplayBuffer()                          --> creates episodic tensor storage
                     ├── utils_ne.SSNE(args)                     --> creates EA evolution engine
                     ├── mac_REGISTRY["RL_basic_mac"]()          --> instantiates RL_BasicMAC (creates agent_SR & agent_W)
                     ├── mac_REGISTRY["EA_basic_mac"]() (x5)     --> instantiates Gen_BasicMAC (injected with mac.agent_SR)
                     └── le_REGISTRY["facmac_learner_discrete_EA"]()
                          ├── FACMACDiscreteCritic & QMixer       --> MARL critic and mixer
                          ├── PeVFA_FACMACDiscreteCritic & PeVFA_QMixer --> EA policy-extended critic and mixer
                          ├── V_Net                               --> state-value network for VMM weights
                          └── MINE                                --> mutual information neural estimator
```

### 2.2. Training Iteration Runtime Flow
```text
src/run.py:run_sequential() (while runner.t_env <= args.t_max)
 ├── Branch A: EA Active (runner.t_env > 10000 and episode % EA_freq == 0)
 │    ├── for ea_mac in pop: runner.run(ea_mac, EA=True)  --> collect EA episodes & fitness
 │    ├── runner.run(mac, EA=False)                       --> collect MARL episode
 │    ├── buffer.insert_episode_batch()                   --> push all episodes to shared buffer
 │    └── for agent_idx in range(n_agents):
 │         evolver.epoch(pop, fitness, agent_idx)         --> tournament selection, agent-level crossover, mutation
 │
 ├── Branch B: Warmup / Non-EA (runner.t_env <= 10000)
 │    ├── runner.run(mac, EA=False)                       --> collect MARL episode only
 │    └── buffer.insert_episode_batch()                   --> push MARL episode to buffer
 │
 ├── Learner Update (repeat_num = 6 if EA Active, else 1)
 │    └── for i in range(repeat_num):
 │         batch = buffer.sample(batch_size=32)
 │         learner.train(batch, all_teams, t_env, episode)
 │              ├── Step 1: PeVFA critic & PeVFA mixer TD(lambda) update
 │              ├── Step 2: Conventional critic & mixer TD(lambda) update
 │              ├── Step 3: V_Net MSE update towards max(target_marl, target_ea)
 │              ├── Step 4: MARL policy gradient (pg_loss)
 │              ├── Step 5: Sample team -> VMM MINE loss + EA policy gradient (ea_pg_loss)
 │              ├── Step 6: Backward total_loss -> step agent_optimiser & MINE_optimiser
 │              └── Step 7: Target network sync (every 200 critic steps)
 │
 ├── Policy Injection (runner.t_env > 10000 and episode % EA_freq == 0)
 │    └── rl_to_evo(mac, pop[worst_non_elite], agent_idx) --> copy MARL policy to worst EA team
 │
 └── Telemetry & Logging
      ├── runner._log() (runner_log_interval=2000)
      └── logger.print_recent_stats() (log_interval=2000)
```
