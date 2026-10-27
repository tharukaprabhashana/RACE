# System Diagrams: Architecture, Dataflows, and Execution Sequences

This document contains eight Mermaid diagrams illustrating the structure, execution sequences, representation mechanics, and parameter traces of the RACE framework as verified directly in the source code.

---

## Diagram A — System Architecture

This diagram shows the complete system architecture: the environment, MARL team, EA population, shared observation representation encoders, replay buffer, critics, PeVFA, and MINE.

```mermaid
flowchart TD
    subgraph Environment ["Dec-POMDP Environment (StarCraft II / SMAC)"]
        Env["SC2 Engine (3s_vs_3z)<br/>States s in R^48, Observations o^i in R^30"]
    end

    subgraph Policies ["Representation-Asymmetry Team Construction (RATC)"]
        SR["Shared Representation Encoder Z_phi<br/>(QMIXRNNAgent_SR: Linear + GRU, dim=64)<br/>SHARED ACROSS ALL AGENTS & ALL TEAMS"]
        
        subgraph MARLTeam ["MARL Team (RL_BasicMAC)"]
            W_marl["Linear Policy Head W_marl<br/>(QMIXRNNAgent_W: Linear(64, 9))"]
        end
        
        subgraph EAPopulation ["EA Population P (Gen_BasicMAC x5)"]
            W_ea1["Team W_1: {W_1^1, W_1^2, W_1^3}"]
            W_ea2["Team W_2: {W_2^1, W_2^2, W_2^3}"]
            W_eaN["Teams W_3, W_4, W_5"]
        end
    end

    subgraph Buffer ["Replay Storage (src/components/episode_buffer.py)"]
        D[("ReplayBuffer D<br/>(Capacity: 5000 episodes)")]
    end

    subgraph CriticsAndValues ["Value Functions & Estimators (src/learners/EA_facmac_learner_discrete.py)"]
        Critic_MARL["Conventional Critic Q_psi & QMixer<br/>Evaluates MARL Team Actions"]
        PeVFA["PeVFA Critic Q_theta & PeVFA_QMixer<br/>Conditioned on embedded policy weights e_W"]
        V_Net["State-Value Net V_zeta(s)<br/>Target: max(Q_psi', Q_theta')"]
        MINE["MINE Estimator T_omega(Z, s)<br/>Fenchel Dual Jensen-Shannon Lower Bound"]
    end

    subgraph Evolver ["Neuro-Evolution Engine (src/ea/mod_neuro_evo.py)"]
        SSNE["SSNE: Tournament(k=3), Elitism, Agent Crossover, Mutation"]
    end

    %% Interactions
    Env -->|"Local obs o^i"| SR
    SR -->|"Latent Z in R^64"| W_marl
    SR -->|"Latent Z in R^64"| W_ea1
    SR -->|"Latent Z in R^64"| W_ea2
    SR -->|"Latent Z in R^64"| W_eaN

    W_marl -->|"Actions u_marl"| Env
    W_ea1 -->|"Actions u_ea"| Env
    
    Env -->|"Episodes (o, s, u, r, s')"| D
    Env -->|"Episodic Return Sum r_t"| SSNE

    %% Training Flows
    D -->|"Sampled Batch (size 32)"| Critic_MARL
    D -->|"Sampled Batch (size 32)"| PeVFA
    D -->|"Sampled Batch (size 32)"| V_Net
    D -->|"Sampled Batch (size 32)"| MINE

    Critic_MARL -->|"MARL Policy Gradient (pg_loss)"| W_marl
    Critic_MARL -->|"Value Gradient via pg_loss"| SR
    PeVFA -->|"Value Gradient via ea_pg_loss"| SR
    V_Net -->|"State Value Weights V(s)"| MINE
    MINE -->|"Value-Aware MI Gradient"| SR

    SSNE -.->|"Evolved Policy Parameters W"| EAPopulation
    W_marl -.->|"Policy Injection (rl_to_evo)"| EAPopulation
```

**Legend:**
* **Solid Arrows ($\longrightarrow$):** Verified runtime data flow (observations, actions, transition batches, gradients).
* **Dashed Arrows ($-\cdot\to$):** Policy parameter transfer and evolutionary manipulation (cloning, injection, mutation).
* **Supporting Code References:** [`src/controllers/EA_basic_controller.py:7-301`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L7-L301), [`src/learners/EA_facmac_learner_discrete.py:209-456`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L209-L456), [`src/ea/mod_neuro_evo.py:7-208`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L7-L208).

---

## Diagram B — Startup and Module Call Graph

This diagram shows how execution flows from CLI invocation through Sacred configuration loading to the training loop.

```mermaid
flowchart TD
    CLI["Command Line Invocation<br/>python3 src/main.py --config=facmac_smac --env-config=sc2 with ..."] --> Main["src/main.py:__main__"]
    
    subgraph ConfigPipeline ["Configuration Layering (src/main.py)"]
        Def["1. Load src/config/default.yaml"]
        EnvConf["2. Intercept --env-config -> Load src/config/envs/sc2.yaml"]
        AlgConf["3. Intercept --config -> Load src/config/algs/facmac_smac.yaml"]
        Merge["4. recursive_dict_update() merges configs"]
        SacredAdd["5. ex.add_config(config_dict)"]
        Observer["6. ex.observers.append(FileStorageObserver('results/sacred'))"]
        RunCmd["7. ex.run_commandline(params) parses 'with' overrides"]
        
        Def --> EnvConf --> AlgConf --> Merge --> SacredAdd --> Observer --> RunCmd
    end

    Main --> ConfigPipeline
    RunCmd --> MyMain["src/main.py:my_main(_run, _config, _log)"]
    
    subgraph Setup ["Initialization Pipeline (src/run.py)"]
        Seed["Set seeds: np.random, th.manual_seed"]
        RunFn["src/run.py:run()"]
        Sanity["args_sanity_check()"]
        RunSeq["src/run.py:run_sequential(args, logger)"]
        
        MyMain --> Seed --> RunFn --> Sanity --> RunSeq
    end

    subgraph Assembly ["Framework Assembly (src/run.py:run_sequential)"]
        InitRunner["runner = EpisodeRunner(args, logger)<br/>Launches StarCraft II engine"]
        InitBuf["buffer = ReplayBuffer(...)"]
        InitEvo["evolver = SSNE(args)"]
        InitMARL["mac = RL_BasicMAC(...)<br/>Creates agent_SR & agent_W"]
        InitEA["pop = [Gen_BasicMAC(..., mac.agent_SR) for _ in range(5)]"]
        InitLearner["learner = EA_FACMACDiscreteLearner(mac, ...)<br/>Builds Critic, PeVFA, V_Net, MINE"]
        
        RunSeq --> InitRunner
        RunSeq --> InitBuf
        RunSeq --> InitEvo
        RunSeq --> InitMARL
        RunSeq --> InitEA
        RunSeq --> InitLearner
    end

    Assembly --> LoopHead["Main Training Loop<br/>while runner.t_env <= args.t_max:"]
```

**Supporting Code References:** [`src/main.py:16-107`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py#L16-L107), [`src/run.py:28-284`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L28-L284).

---

## Diagram C — One Training Iteration Sequence

This flowchart documents the exact execution sequence observed during a single episodic training iteration.

```mermaid
flowchart TD
    Start(["Start Iteration: while runner.t_env <= args.t_max"]) --> Gate{"runner.t_env > 10000<br/>and episode % EA_freq == 0?"}

    subgraph RolloutEA ["EA Active: Full Population Rollout & Evolution"]
        RollPop["For each ea_mac in pop:<br/>runner.run(ea_mac, EA=True, explore=False)<br/>Collect return f(W_j); advance t_env; store in buffer"]
        RollMARL_1["runner.run(mac, EA=False, explore=True)<br/>Collect return; advance t_env; store in buffer"]
        EvoEpoch["evolver.epoch(pop, fitness, agent_idx, agent_level=True)<br/>1. Rank by fitness & preserve 1 elite team<br/>2. Tournament selection (size 3) for offspring<br/>3. Agent-level crossover (swap agent W matrices)<br/>4. Agent-level mutation (perturb weights with frac)"]
        Set6["repeat_num = 6"]
        
        RollPop --> RollMARL_1 --> EvoEpoch --> Set6
    end

    subgraph RolloutWarmup ["Warmup Phase (t_env <= 10,000)"]
        RollMARL_2["runner.run(mac, EA=False, explore=True)<br/>Advance t_env; store in buffer"]
        Set1["repeat_num = 1"]
        
        RollMARL_2 --> Set1
    end

    Gate -- "Yes" --> RolloutEA
    Gate -- "No" --> RolloutWarmup

    Set6 --> LearnerLoop
    Set1 --> LearnerLoop

    subgraph LearnerLoop ["Learner Training (for i in range(repeat_num))"]
        SampleBatch["buffer.sample(batch_size=32)"]
        TrainPeVFA["Step 1: Train PeVFA Critic & Mixer (TD-lambda loss)"]
        TrainCritic["Step 2: Train MARL Critic & Mixer (TD-lambda loss)"]
        TrainVNet["Step 3: Train V_Net on max(Q_psi', Q_theta') (MSE loss)"]
        EvalMARL["Step 4: Compute MARL policy gradient (pg_loss)"]
        EvalVMM["Step 5: Sample team -> Compute Value-weighted MINE loss<br/>+ PeVFA policy loss (ea_pg_loss)"]
        BackTotal["Step 6: Backward total_loss = Org_alpha*pg_loss + ea_pg_loss<br/>Step agent_optimiser & MINE_optimiser"]
        UpdateTargets["Step 7: Update Target Networks (every 200 critic steps)"]

        SampleBatch --> TrainPeVFA --> TrainCritic --> TrainVNet --> EvalMARL --> EvalVMM --> BackTotal --> UpdateTargets
    end

    LearnerLoop --> CheckTest{"(t_env - last_test_T) >= 10000?"}
    CheckTest -- "Yes" --> ExecTest["Execute 32 Greedy Test Episodes<br/>Log win rates to WandB"]
    CheckTest -- "No" --> CheckInjection

    ExecTest --> CheckInjection{"runner.t_env > 10000<br/>and episode % EA_freq == 0?"}
    CheckInjection -- "Yes" --> Inject["rl_to_evo(): Copy mac.agent_W -> pop[worst_non_elite].agent_W"]
    CheckInjection -- "No" --> IncEpisode

    Inject --> IncEpisode["episode += 1"]
    IncEpisode --> CheckLogging{"(t_env - last_log_T) >= 2000?"}
    CheckLogging -- "Yes" --> PrintLog["Print Recent Stats & Sync Sacred info.json"]
    CheckLogging -- "No" --> NextLoop(["Next Iteration"])
    PrintLog --> NextLoop
```

**Supporting Code References:** [`src/run.py:284-456`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L284-L456), [`src/learners/EA_facmac_learner_discrete.py:209-456`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L209-L456).

---

## Diagram D — Representation-Asymmetry Team Construction (RATC)

This diagram details the decomposition of agent policies into a shared nonlinear representation encoder and individual linear policy heads.

```mermaid
flowchart TD
    subgraph AgentObs ["Agent Local Observation (Agent i)"]
        O_i["o_t^i (dim: 30) + last_action (dim: 9) + agent_id (dim: 3)<br/>Concatenated Input in R^42"]
    end

    subgraph SharedEncoder ["Shared Observation Representation Encoder Z_phi (SHARED ACROSS ALL TEAMS)"]
        FC1["fc1: Linear(42, 64) + ReLU"]
        GRU["rnn: GRUCell(64, 64)"]
        Z["Latent Representation Z_i in R^64"]
        FC1 --> GRU --> Z
    end

    AgentObs --> SharedEncoder

    subgraph LinearHeads ["Linear Policy Representations W"]
        subgraph MARLHead ["MARL Team Head (RL_BasicMAC)"]
            W_marl["agent_W_marl<br/>Linear(64, 9)<br/>Updated via Policy Gradient"]
        end

        subgraph EAHeads ["EA Population Candidate Heads (Gen_BasicMAC)"]
            W_ea1["Team 1: agent_W_1<br/>Linear(64, 9)<br/>Updated via EA"]
            W_ea2["Team 2: agent_W_2<br/>Linear(64, 9)<br/>Updated via EA"]
            W_ea5["Team 5: agent_W_5<br/>Linear(64, 9)<br/>Updated via EA"]
        end
    end

    Z --> W_marl
    Z --> W_ea1
    Z --> W_ea2
    Z --> W_ea5

    subgraph ActionOutputs ["Decentralized Action Probabilities"]
        Act_marl["pi_marl(o^i) in R^9"]
        Act_ea1["pi_1(o^i) in R^9"]
        Act_ea2["pi_2(o^i) in R^9"]
        Act_ea5["pi_5(o^i) in R^9"]
    end

    W_marl --> Act_marl
    W_ea1 --> Act_ea1
    W_ea2 --> Act_ea2
    W_ea5 --> Act_ea5
```

**Supporting Code References:** [`src/controllers/EA_basic_controller.py:33-58, 174-211`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L33-L58), [`src/modules/agents/EA_qmix_agent.py:27-68`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/agents/EA_qmix_agent.py#L27-L68).

---

## Diagram E — Evolutionary Operators

This diagram shows parent selection, elitism, agent-level crossover, and mutation inside `SSNE.epoch`.

```mermaid
flowchart TD
    subgraph PopRanking ["1. Fitness Ranking"]
        Fit["Population Fitness: {f(W_1), f(W_2), f(W_3), f(W_4), f(W_5)}"]
        Sort["Sort Descending: index_rank"]
        Elite["Elitist Selection:<br/>num_elitists = int(0.2 * 5) = 1<br/>Elite Team = index_rank[0]"]
        Fit --> Sort --> Elite
    end

    subgraph SelectionTournament ["2. Tournament Parent Selection"]
        Tourn["Tournament Selection (k=3):<br/>Sample 3 random teams, pick best fitness.<br/>Produce offspring candidates."]
        Sort --> Tourn
    end

    subgraph Preservation ["3. Elite Preservation"]
        CloneElite["self.clone(master=pop[elite], replacee=pop[unselect_1], agent_index)"]
        Elite --> CloneElite
    end

    subgraph Crossover ["4. Agent-Level Crossover (Team Composition Search)"]
        PairOff["Pair Unselected Candidates: (i, j) with parents (elite, offspring)"]
        SwapChoice{"random.random() < 0.5?"}
        CloneA["clone(pop[i] -> pop[j], agent_index)"]
        CloneB["clone(pop[j] -> pop[i], agent_index)"]
        
        Tourn --> PairOff --> SwapChoice
        SwapChoice -- "True" --> CloneA
        SwapChoice -- "False" --> CloneB
    end

    subgraph Mutation ["5. Agent-Level Mutation (Individual Policy Search)"]
        MutProb{"random.random() < mutation_prob (0.9)?"}
        MutInPlace["mutate_inplace(pop[k], agent_index, agent_level=True)<br/>Sample frac incoming weights per action row<br/>Apply Super-Mutation (0.05), Reset (0.05), or Gaussian Noise (0.85)"]
        
        CloneA --> MutProb
        CloneB --> MutProb
        MutProb -- "Yes" --> MutInPlace
        MutProb -- "No" --> Retain["Retain Weights Unchanged"]
    end
```

**Supporting Code References:** [`src/ea/mod_neuro_evo.py:143-207`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L143-L207).

---

## Diagram F — Shared Representation Learning

This diagram shows how Value Function Maximization (VFM) and Value-Aware Mutual Information Maximization (VMM) integrate to train the shared encoder $Z_\phi$.

```mermaid
flowchart TD
    subgraph Inputs ["Training Batch Inputs"]
        Obs["Observations o_t^i"]
        Actions["Actions u_t"]
        State["Global State s_t"]
    end

    subgraph VFM ["Value Function Maximization (VFM)"]
        SR_VFM["Encoder Z_phi(o_t^i)"]
        W_marl["MARL Head W_marl"]
        W_ea["Sampled EA Head W_j"]
        
        Critic_MARL["Critic Q_psi & QMixer"]
        PeVFA["PeVFA Q_theta & PeVFA_QMixer<br/>(conditioned on e_W)"]
        
        PG_MARL["pg_loss = - Q_tot_marl"]
        PG_EA["ea_qvals = - Q_tot_ea"]

        Obs --> SR_VFM
        SR_VFM --> W_marl --> Critic_MARL --> PG_MARL
        SR_VFM --> W_ea --> PeVFA --> PG_EA
    end

    subgraph VMM ["Value-Aware Mutual Information Maximization (VMM)"]
        VNet["State-Value Net V_zeta(s_t)"]
        MinMax["Min-Max Batch Normalization -> weight in [0, 1]"]
        MINE["MINE Network T_omega(Z, s_t)<br/>Fenchel Dual JSD Loss -> mi_loss"]
        VMM_Loss["MINE_loss = (weight * mi_loss).mean()"]

        State --> VNet --> MinMax
        SR_VFM --> MINE
        State --> MINE
        MinMax --> VMM_Loss
        MINE --> VMM_Loss
    end

    subgraph TotalObjective ["Combined Representation Loss (Eq. 6)"]
        Total["total_loss = Org_alpha * pg_loss - EA_alpha * ea_qvals + state_alpha * MINE_loss"]
        PG_MARL --> Total
        PG_EA --> Total
        VMM_Loss --> Total
    end

    Total -->|"total_loss.backward()"| Backprop["Gradients update:<br/>1. Shared Encoder Z_phi (via agent_optimiser)<br/>2. MARL Head W_marl (via agent_optimiser)<br/>3. MINE Parameters T_omega (via MINE_optimiser)"]
```

**Supporting Code References:** [`src/learners/EA_facmac_learner_discrete.py:327-434`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L327-L434).

---

## Diagram G — Parameter Trace for `frac`

This diagram illustrates the path of `frac` from the command-line argument down to row-wise weight sampling.

```mermaid
flowchart TD
    CLI["CLI Argument:<br/>frac=0.005<br/>(run_race_feasibility.sh:188)"] --> Sacred["Sacred Experiment Commandline Parser<br/>(src/main.py:107)"]
    Sacred --> Config["Unified Config Dict: config['frac'] = 0.005<br/>(src/main.py:38)"]
    Config --> RunPy["SimpleNamespace: args.frac = 0.005<br/>(src/run.py:33)"]
    RunPy --> SSNEInit["SSNE.__init__():<br/>self.frac = args.frac (0.005)<br/>(src/ea/mod_neuro_evo.py:14)"]
    SSNEInit --> MutateInPlace["SSNE.mutate_inplace(gene, agent_index, agent_level=True)<br/>(src/ea/mod_neuro_evo.py:79)"]

    subgraph WeightMutation ["Low-Level Weight Mutation (Lines 108-128)"]
        GetW["W = gene.agent_W[agent_index].state_dict()[key]<br/>Shape: (n_actions=9, rnn_hidden_dim=64)"]
        LoopRows["for index in range(W.shape[0]): (9 rows)"]
        CalcSample["index_list = random.sample(range(64), int(64 * 0.005))<br/>CRITICAL: int(64 * 0.005) = int(0.32) = 0!"]
        SampleEmpty["index_list = [] (Empty List)"]
        NoPerturb["for ind in index_list: (Executes 0 times!)<br/>No weights are modified!"]

        GetW --> LoopRows --> CalcSample --> SampleEmpty --> NoPerturb
    end

    MutateInPlace --> WeightMutation
```

**Supporting Code References:** [`src/ea/mod_neuro_evo.py:14, 114`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L14).

---

## Diagram H — Experiment and Telemetry Logging Flow

This diagram shows how metrics flow from episode rollouts through loggers to Sacred files and terminal displays.

```mermaid
flowchart TD
    subgraph EnvStep ["Environment Execution (src/runners/episode_runner.py)"]
        Step["self.env.step(actions)"]
        Stats["Accumulate stats: return, battle_won, dead_allies, dead_enemies"]
        AdvanceT["self.t_env += episode_length (t)"]
        Step --> Stats --> AdvanceT
    end

    subgraph Telemetry ["Telemetry Routing (src/utils/logging.py)"]
        LogStat["Logger.log_stat(key, value, t)"]
        AdvanceT --> LogStat
    end

    subgraph Outputs ["Persistent Outputs & Console Display"]
        SacredInfo["Sacred info.json (results/sacred/6/info.json)<br/>Stores time-series arrays: return_mean_T, critic_loss_T, etc."]
        Console["Console Logger / training.log<br/>Prints 'Recent Stats' table every log_interval=2000"]
        WandB["Weights & Biases (Disabled by default: WANDB_MODE='disabled')"]
    end

    LogStat --> SacredInfo
    LogStat --> Console
    LogStat -.-> WandB
```

**Supporting Code References:** [`src/runners/episode_runner.py:129-160`](file:///home/tharuka/Documents/GitHub/RACE/src/runners/episode_runner.py#L129-L160), [`src/utils/logging.py:27-53`](file:///home/tharuka/Documents/GitHub/RACE/src/utils/logging.py#L27-L53).
