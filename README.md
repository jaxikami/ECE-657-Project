# project
# Safe Projection Reinforcement Learning (SPRL) for Photo-Production

This repository implements a **Safe Projection Reinforcement Learning (SPRL)** framework designed to maximize biomass production in a bioreactor while strictly adhering to physical safety constraints. The system utilizes a **Residual Safeguard** to project risky agent "intents" onto a mathematically safe action manifold.

---

## 🏗️ Project Architecture

The core of this project is the comparison between a standard PPO agent and an SPRL-enabled agent:

* **NonResNet (Baseline)**: A standard Actor-Critic PPO agent that learns safety constraints indirectly through environment penalties.
* **SPRL (Proposed)**: An agent that learns a latent intent $z$, which is then passed through a pre-trained **ActionProjectionNetwork** to ensure the final physical action is safe.



### The Residual Safeguard Logic
The safeguard does not predict actions directly. Instead, it predicts a **reduction** ($\Delta$) to be subtracted from the agent's intent:
$$u_{safe} = z_{intent} - \text{ReLU}(\text{network}(s, z_{intent}))$$
This ensures the safeguard can only decrease production-related actions (throttling) to maintain safety, never increase them beyond the agent's intent.

---

## 📂 File Registry

| File | Primary Function |
| :--- | :--- |
| `pretrain.py` | Trains the `ActionProjectionNetwork` using an **Asymmetric Loss** (20x penalty for safety violations). |
| `res_net_agent.py` | Implements the **SPRL_Agent** with budget-aware projection and mapping penalties. |
| `data_gen.py` | Generates biased datasets ($450,000$ samples) focusing on "boundary conditions" near physical limits. |
| `main.py` | Orchestrates the $50,000$ episode training loop and subsequent evaluation. |
| `mismatch.py` | Stress-tests the safeguard against edge-case Light and Nitrogen violations. |
| `lag_agent.py` | Provides the baseline PPO `NonResNet_Agent`. |
| `utils.py` | Handles logging and generates trajectory/convergence plots. |

---

## ⚙️ Physical Constraints & State Space

The environment is governed by two critical physical limits:
1.  **Light Constraint**: Prevents photo-inhibition. $I_{safe} \leq \frac{I_{CRIT}}{\text{shading} + 1e-6} \cdot 0.95$.
2.  **Nitrogen Budget**: Prevents nitrate exhaustion. $F_{N} \leq (N_{LIMIT} \cdot 0.95) - c_N$.

### Budget-Aware Features
To improve safety, the safeguard uses a **4D state space**:
* $c_x$: Biomass concentration.
* $c_N$: Nitrate concentration.
* $c_q$: Product concentration.
* **Nitrogen Budget Distance**: A normalized feature representing how close the current nitrate level is to the safety "wall".

---

## 🚀 Execution Guide

### 1. Manifold Training
Train the safety safeguard first. The training will automatically exit once the `SmoothL1Loss` converges below $3.0 \times 10^{-4}$.
```bash
python pretrain.py