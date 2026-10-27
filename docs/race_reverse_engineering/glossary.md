# Glossary of Concepts and Abbreviations

This glossary defines the foundational concepts, mathematical terms, and abbreviations used throughout the RACE codebase and research paper.

---

### Core MARL Concepts

* **Dec-POMDP (Decentralized Partially Observable Markov Decision Process):**  
  The mathematical formalization for cooperative multi-agent tasks under partial observability, defined as $\langle \mathcal{N}, \mathcal{S}, \mathcal{U}, \mathcal{O}, \mathcal{T}, \mathcal{R}, \gamma \rangle$. Each agent $i \in \mathcal{N}$ observes an individual observation $o^i \sim \mathcal{O}(s, i)$, selects an action $u^i \sim \pi^i(\cdot \mid o^i)$, and the team transitions according to $\mathcal{T}(s, \mathbf{u})$ receiving a shared team reward $r = \mathcal{R}(s, \mathbf{u})$.
* **CTDE (Centralized Training with Decentralized Execution):**  
  A prominent MARL paradigm where global information (true state $s$, other agents' actions and parameters) is permitted during centralized offline training (e.g., within critics and mixers), while execution requires each agent $i$ to select actions solely based on its local observation history $o^i$ without inter-agent communication.
* **FACMAC (Factored Multi-Agent Centralised Policy Gradients):**  
  The baseline MARL algorithm upon which RACE is constructed in this codebase. FACMAC maintains deep decentralized deterministic or stochastic actors and a centralized critic factorized using non-linear monotonic mixing networks (QMIX style) that compute policy gradients directly through the mixer.
* **QMIX:**  
  A value factorization method ensuring monotonicity between local utilities $Q_i(o^i, u^i)$ and joint action-value $Q_{tot}(s, \mathbf{u})$ via $\frac{\partial Q_{tot}}{\partial Q_i} \ge 0$, enforced using state-dependent absolute weights produced by hypernetworks.

---

### Evolutionary & Hybrid ERL Concepts

* **EA (Evolutionary Algorithm):**  
  A class of population-based, gradient-free optimization heuristics that iteratively perform environment evaluation, fitness ranking, selection, crossover, and mutation over a population of candidate solutions.
* **ERL (Evolutionary Reinforcement Learning):**  
  A hybrid learning paradigm combining EA and RL. Typically, EA maintains population diversity and robust exploratory policy trajectories, populating an off-policy replay buffer, while gradient-based RL optimizes a specialized policy that periodically injects high-performing genes into the EA population.
* **SSNE (Steady-State Neuro-Evolution):**  
  The specific genetic algorithm variant implemented in [`src/ea/mod_neuro_evo.py`](file:///home/tharuka/Documents/GitHub/RACE/src/ea/mod_neuro_evo.py). Unlike generational GA that replaces an entire population at once, SSNE preserves elitists, conducts tournament selection, and performs crossover and mutation in-place.
* **Fitness ($f(W_j)$):**  
  The performance metric used by the EA to evaluate team $j$. In RACE, fitness is defined as the undiscounted cumulative episodic return $f(W_j) = \sum_{t=0}^T r_t$ over an evaluation rollout ([`src/run.py:293`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L293)).
* **Elite Team:**  
  The candidate team in the population achieving the highest fitness in the current evaluation generation ($\arg\max_j f(W_j)$). Elites are shielded from mutation and directly preserved as parents for the next generation.
* **Policy Injection / Synchronization:**  
  The mechanism by which knowledge acquired via gradient ascent in MARL is transferred into the evolutionary population. In RACE, the MARL policy representation is copied into the worst-performing non-elite team ([`src/run.py:426-435`](file:///home/tharuka/Documents/GitHub/RACE/src/run.py#L426-L435)).

---

### RACE-Specific Architectural Concepts

* **RATC (Representation Asymmetry Team Construction):**  
  The core structural decomposition introduced by RACE (Paper Section 3.1). Instead of each agent possessing an independent end-to-end deep policy network, any policy controlling agent $i$ across any team $j$ is split into:
  1. A shared non-linear observation representation encoder $Z_{\phi_i}(o^i)$ producing latent feature vector $z^i \in \mathbb{R}^d$.
  2. An individual linear policy representation $W_j^i \in \mathbb{R}^{(d+1) \times |\mathcal{U}_i|}$.
  This creates a representation asymmetry: deep shared representations capture general task and coordination features, while linear representations form a compact, favorable search space for evolution.
* **MARL Team ($W_{marl}$):**  
  The single designated team whose linear policy parameters $W_{marl}$ are updated using gradient ascent on the conventional MARL centralized critic $Q_\psi$.
* **EA Population ($\mathcal{P} = \{W_1, \dots, W_n\}$):**  
  The pool of $n$ candidate teams ($n = 5$ by default) whose linear policy representations are optimized via selection, crossover, and mutation.
* **PeVFA (Policy-Extended Value Function Approximator):**  
  A centralized critic $Q_\theta(s, \mathbf{u}, W_j)$ that takes the candidate team policy representation $W_j$ as an explicit auxiliary input alongside state and actions (Paper Section 2.1 & 3.2). PeVFA enables value generalization across distinct candidate policies in the population.
* **VFM (Value Function Maximization):**  
  The objective function (Paper Eq. 2) used to optimize the shared encoder $Z_\phi$ towards regions of the policy space that maximize expected return for both the MARL team ($Q_\psi$) and EA teams ($Q_\theta$).
* **VMM (Value-Aware Mutual Information Maximization):**  
  The auxiliary representation objective (Paper Eq. 5) that maximizes mutual information $I(Z_\phi(o^i); s)$ between local agent representations and the global state $s$, weighted by normalized state values $V_\zeta(s)$ so that representations focus on superior coordination states rather than poor ones.
* **MINE (Mutual Information Neural Estimation):**  
  A neural estimator ($T_\omega$) that provides a lower bound on mutual information between continuous representations using the Fenchel dual formulation of the Jensen-Shannon or KL divergence.
* **$V_\zeta$ (State-Value Network):**  
  A neural network $V_\zeta(s)$ trained to estimate the maximum expected value attainable from state $s$ across both the MARL team and EA teams ($F(s) = r + \max(Q_{\psi'}, Q_{\theta'})$).
