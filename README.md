# Safe Policy Reinforcement Learning (SPRL) for Bioreactor Control

This repository implements a **Safe Reinforcement Learning** framework for optimizing Phycocyanin production in a photobioreactor. The project compares a standard Lagrangian PPO approach with **SPRL (Safe Policy Reinforcement Learning)**, which utilizes a pre-trained **Action Projection Network** (Safety Filter) to ensure physical constraints are never violated during the learning process.

---

## 🚀 Overview

The bioreactor system is defined by non-linear kinetics (modeled in `env.py`) involving:
* **States:** Biomass ($c_x$), Nitrate ($c_N$), Phycocyanin ($c_q$), and Normalized Time.
* **Actions:** Light Intensity ($I$) and Nitrate Feed Rate ($F_N$).
* **Goal:** Maximize $c_q$ production while respecting safety boundaries.

### Safety Constraints
* **$g_1$ (Path):** Nitrate concentration must stay below **800 mg/L**.
* **$g_2$ (Ratio):** Maintains a specific bioproduct-to-biomass ratio.
* **$g_3$ (Terminal):** Nitrate levels must be below **150 mg/L** at the end of the 240-hour batch.

---

## 🏗️ Architecture

### Safe Policy Reinforcement Learning (SPRL)
The SPRL agent (`res_net_agent.py`) consists of two primary components:
1.  **Actor-Critic (PPO):** Generates a nominal control "intent."
2.  **Safety Filter:** A pre-trained Residual Network that intercepts the intent and projects it onto a safe manifold if a violation is predicted. It also includes analytical overrides for terminal safety.

### Standard RL (Baseline)
The baseline agent (`lag_agent.py`) uses a standard reward-shaping approach, where safety is encouraged via negative penalties (Lagrangian signals) rather than hard architectural constraints.

---

## 📂 Project Structure

| File | Function |
| :--- | :--- |
| `main.py` | Primary execution script for training and evaluation loops. |
| `env.py` | Bioreactor environment using Numba-accelerated ODE integration. |
| `pretrain.py` | Scripts for training the Safeguard/Safety Filter manifold. |
| `res_net_agent.py` | Implementation of the SPRL agent with the integrated filter. |
| `lag_agent.py` | Baseline PPO agent using reward-based penalties. |
| `data_gen.py` | Synthetic data generation for safety filter pre-training. |
| `validation.py` | Stress-tests for verifying safeguard identity mapping and safety. |
| `utils.py` | Data logging and research-quality visualization tools. |

---
