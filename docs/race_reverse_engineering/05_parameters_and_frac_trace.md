# 05 — Parameter Analysis and Deep Trace of `frac`

This document provides a comprehensive audit of all key hyperparameters in RACE and details a step-by-step trace of the mutation parameter `frac` from the command-line interface down to low-level tensor operations.

---

## 1. Master Parameter Dictionary

| Parameter Name | Default Value (`default.yaml`) | Feasibility Test Value | Consumed in Code (File & Line) | Functional Role & Affected Subsystem | Verified Effect & Code Nuances |
|:---|:---:|:---:|:---|:---|:---|
| **`frac`** | `1.0` | `0.005` | [`src/ea/mod_neuro_evo.py:14, 114`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L14) | Evolutionary Mutation | Controls fraction of representation dimensions mutated per action row in linear policy head. In discrete SMAC with 64 units, `int(64 * 0.005) = 0`, causing zero mutated weights. |
| **`state_alpha`** | `0.1` | `0.001` | [`src/learners/EA_facmac_learner_discrete.py:419`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L419) | Representation Learning (VMM) | Hyperparameter $\beta$ from Paper Eq. 6. Weights the Value-Aware MINE loss in the shared observation encoder update. |
| **`EA`** | `0` | `1` | [`src/run.py:219, 232, 289, 315, 334, 426`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L219), [`src/learners/EA_facmac_learner_discrete.py:163, 221, 366`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L163) | Algorithm Mode Switch | Master flag enabling RACE. When `1`, constructs EA population controllers, PeVFA critic/mixer, $V_\zeta$ net, and triggers neuro-evolution. |
| **`EA_freq`** | `1` | `1` | [`src/run.py:289, 315, 426`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L289) | Evolutionary Scheduling | Frequency (in episodes) at which EA population rollout, evolution, `repeat_num=6` learner training, and MARL-to-EA policy injection occur. |
| **`EA_alpha`** | `1.0` | `1.0` | [`src/learners/EA_facmac_learner_discrete.py:419`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L419) | Representation Learning (VFM) | Coefficient weighting the EA team PeVFA policy gradient in Value Function Maximization (Paper Eq. 2). |
| **`Org_alpha`** | `1.0` | `1.0` | [`src/learners/EA_facmac_learner_discrete.py:425`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L425) | Representation Learning (VFM) | Coefficient weighting the MARL team conventional policy gradient in Value Function Maximization (Paper Eq. 2). |
| **`SAME`** | `0` | `0` | [`src/controllers/EA_basic_controller.py:183, 250`](file:///home/tharuka/Documents/GitHub/RACE/src/controllers/EA_basic_controller.py#L183), [`src/run.py:303`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L303) | Team Policy Construction | When `0`, instantiates independent linear policy representations $W_j^i$ for each agent in an EA team. When `1`, shares one $W$ across all agents. |
| **`pop_size`** | `5` | `5` | [`src/ea/mod_neuro_evo.py:15`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L15), [`src/run.py:336, 430`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L336) | Population Dimension | Size of the evolutionary team pool $n$. **Nuance:** [`src/run.py:222`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L222) hardcodes `range(5)` when instantiating controllers. |
| **`elite_fraction`** | `0.2` | `0.2` | [`src/ea/mod_neuro_evo.py:16-17`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L16-L17) | Evolutionary Selection | Fraction of population designated as elites. For `pop_size=5`, `int(0.2 * 5) = 1` elite team. Clamped to minimum of 1. |
| **`rl_to_ea_synch_period`** | `1` | `1` | [`src/config/default.yaml:26`](file:///home/tharuka/Documents/GitHub/RACE/src/config/default.yaml#L26) | Policy Synchronization | **Unused in Code.** Defined in YAML but never accessed in Python. Policy synchronization is governed exclusively by `EA_freq`. |
| **`mutation_prob`** | `0.9` | `0.9` | [`src/ea/mod_neuro_evo.py:204`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L204) | Evolutionary Mutation | Probability that any non-elite team undergoes mutation during an evolutionary epoch. |
| **`t_max`** | `10000` (`default.yaml`)<br/>`2000000` (`facmac_smac.yaml`) | `10000` | [`src/run.py:284`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L284) | Training Termination | Total environment timestep budget. Governs termination: `while runner.t_env <= args.t_max:`. |
| **`batch_size_run`** | `1` | `1` | [`src/runners/episode_runner.py:16-17`](file:///home/tharuka/Documents/GitHub/RACE/src/runners/episode_runner.py#L16-L17), [`src/run.py:449`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L449) | Environment Parallelism | Number of parallel environments. `EpisodeRunner` asserts `batch_size_run == 1` (serial execution only). |
| **`use_cuda`** | `True` | `False` | [`src/run.py:34, 464`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L34), [`src/learners/EA_facmac_learner_discrete.py:139`](file:///home/tharuka/Documents/GitHub/RACE/src/learners/EA_facmac_learner_discrete.py#L139) | Hardware Dispatch | When `False`, moves tensors, modules, and buffers to CPU device (`"cpu"`). Automatically disabled if CUDA is unavailable. |
| **`start_timesteps`** | `10000` | `10000` | [`src/run.py:289, 315, 426`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L289) | EA Warmup Boundary | Timestep threshold before EA population rollout, evolution, repeat training, and RL-to-EA sync can execute. |
| **`pr_dim`** | `64` | `64` | [`src/modules/critics/facmac.py:81`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/critics/facmac.py#L81), [`src/modules/mixers/qmix.py:114`](file:///home/tharuka/Documents/GitHub/RACE/src/modules/mixers/qmix.py#L114) | Architecture Dimension | Dimensionality of policy representation embedding $e_{W_j}$ inside PeVFA critic and PeVFA mixer. |

---

## 2. Deep Dive: End-to-End Parameter Trace of `frac`

The parameter `frac` represents the evolutionary mutation intensity. In the research paper (Section 4.1, Table 4), it is designated as hyperparameter $\alpha \in \{0.005, 0.01, 0.05, 0.1\}$.

### Step 1: Command-Line Injection & Sacred Parsing
1. In [`run_race_feasibility.sh:188`](file:///home/tharuka/Documents/GitHub/RACE/run_race_feasibility.sh#L188), `frac=0.005` is passed as a command-line argument.
2. In [`src/main.py:107`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py#L107), `ex.run_commandline(params)` parses `frac=0.005` and overrides `frac: 1.0` from `default.yaml:31`.
3. In [`src/main.py:36-44`](file:///home/tharuka/Documents/GitHub/RACE/src/main.py#L36-L44), `my_main(_run, _config, _log)` receives `_config["frac"] = 0.005`.
4. In [`src/run.py:33`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L33), config is packaged into a SimpleNamespace: `args.frac = 0.005`.

### Step 2: Ingestion by the Neuro-Evolution Engine
1. In [`src/run.py:217`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L217), `evolver = utils_ne.SSNE(args)` is called.
2. In [`src/ea/mod_neuro_evo.py:14`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L14), `SSNE.__init__` assigns:
   ```python
   self.frac = args.frac # self.frac becomes 0.005
   ```

### Step 3: Invocation During Evolution
1. When `runner.t_env > args.start_timesteps` and `episode % args.EA_freq == 0`, [`src/run.py:307`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L307) invokes:
   ```python
   elite_index = evolver.epoch(pop, fitness, agent_index, agent_level=True)
   ```
2. Inside `SSNE.epoch` ([`src/ea/mod_neuro_evo.py:204-205`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L204-L205)):
   ```python
   for i in range(self.population_size):
       if i not in new_elitists:
           if random.random() < self.args.mutation_prob:
               self.mutate_inplace(pop[i], agent_index=agent_index, agent_level=agent_level)
   ```

### Step 4: Low-Level Tensor Mutation Operation
Inside `SSNE.mutate_inplace` ([`src/ea/mod_neuro_evo.py:79-129`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L79-L129)):
```python
model_params = gene.agent_W[agent_index].state_dict()

for i, key in enumerate(model_params):
    if is_lnorm_key(key):
        continue
    W = model_params[key]
    if len(W.shape) == 2: # Weight matrix (out_features, in_features)
        if agent_level:
            ssne_prob = 1.0
            action_prob = 1.0
        
        if random.random() < ssne_prob:
            num_variables = W.shape[0] # Number of actions (rows)
            for index in range(num_variables):
                random_num_num = random.random()
                if random_num_num <= action_prob:
                    # EXACT CONSUMPTION POINT:
                    index_list = random.sample(range(W.shape[1]), int(W.shape[1] * self.frac))
                    random_num = random.random()
                    if random_num < super_mut_prob: # 0.05
                        for ind in index_list:
                            W[index, ind] += random.gauss(0, super_mut_strength * W[index, ind])
                    elif random_num < reset_prob:   # 0.10
                        for ind in index_list:
                            W[index, ind] = random.gauss(0, 1)
                    else:                           # 0.85
                        for ind in index_list:
                            W[index, ind] += random.gauss(0, mut_strength * W[index, ind])
                    W[index, :] = np.clip(W[index, :], a_min=-1000000, a_max=1000000)
```

---

## 3. Critical Discovery: The Integer Truncation Quirk

Notice line 114:
$$\text{num\_mutated\_weights} = \text{int}\left( W.\text{shape}[1] \cdot \text{self.frac} \right)$$

In PyTorch, a linear layer `nn.Linear(in_features, out_features)` stores its weight matrix in the shape `(out_features, in_features)`.
* In discrete SMAC with `QMIXRNNAgent_W`, `in_features = rnn_hidden_dim = 64` and `out_features = n_actions`.
* Therefore, `W.shape[1] = 64`.
* When `frac = 0.005`:
  $$\text{int}(64 \times 0.005) = \text{int}(0.32) = 0$$
* `random.sample(range(64), 0)` returns an empty list: `[]`.
* **Result:** The inner loop `for ind in index_list:` executes zero times! No weights are perturbed at all!

### Minimum Effective `frac` Threshold
For any weights to be mutated in discrete SMAC, `int(64 * frac)` must be at least 1:
$$\text{frac} \ge \frac{1}{64} = 0.015625$$
In the paper (Table 4), the authors report values of $\alpha$ such as `0.005` for `3s_vs_3z` and `3s5z`. In this implementation, setting `frac = 0.005` completely disables weight mutation in discrete SMAC tasks, effectively running the EA as **crossover-only**!

### Comparison with Ablation Files
In [`src/ea/mod_neuro_evo_abl_mutation.py:111`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo_abl_mutation.py#L111), the formula uses ceiling:
```python
num_mutations = fastrand.pcg32bounded(int(math.ceil(self.frac * num_weights)))
```
Using `math.ceil` prevents truncation to zero when `frac > 0`. However, the active production engine in [`src/ea/mod_neuro_evo.py:114`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py#L114) uses Python's standard `int(...)`, causing silent truncation to 0 for small fractions.
