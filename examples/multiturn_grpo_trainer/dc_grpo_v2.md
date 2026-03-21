# DC-GRPO: Decomposed Credit GRPO

---

## Notation

| Symbol | Meaning |
|--------|---------|
| G | Group size (number of trajectories per prompt) |
| T | Number of turns |
| i | Trajectory index, i ∈ {1, ..., G} |
| t | Turn index, t ∈ {1, ..., T} |
| r_{i,t} | Reward of trajectory i at turn t |
| R_{i,t} | Discounted return of trajectory i **from turn t onward**: R_{i,t} = Σ_{k=t}^{T} γ^{k-t} · r_{i,k} |
| μ^r_t | Group mean of **immediate rewards** at turn t: μ^r_t = (1/G) Σ_j r_{j,t} |
| σ^r_t | Group std of **immediate rewards** at turn t: σ^r_t = std_j(r_{j,t}) |
| μ^R_t | Group mean of **discounted returns from turn t**: μ^R_t = (1/G) Σ_j R_{j,t} |
| σ^R_t | Group std of **discounted returns from turn t**: σ^R_t = std_j(R_{j,t}) |

The superscript distinguishes what is being averaged:
- **r** superscript → statistics over single-turn rewards
- **R** superscript → statistics over multi-turn discounted returns

---

## Exact Decomposition

The discounted return satisfies:

```
R_{i,t} = r_{i,t} + γ · R_{i,t+1}
```

The group mean of returns satisfies the same recursion:

```
μ^R_t = (1/G) Σ_j R_{j,t}
      = (1/G) Σ_j [r_{j,t} + γ · R_{j,t+1}]
      = μ^r_t + γ · μ^R_{t+1}
```

Therefore the centered return decomposes exactly:

```
R_{i,t} - μ^R_t = (r_{i,t} - μ^r_t) + γ · (R_{i,t+1} - μ^R_{t+1})
                   ───────────────────   ─────────────────────────────
                   immediate credit       future credit
```

This is not an approximation. It is algebraically exact.

---

## Three Methods

### Method A: Normalize → Discount (current)

```
A_{i,t} = Σ_{k=t}^{T} γ^{k-t} · (r_{i,k} - μ^r_k) / σ^r_k
```

Each term is divided by a different σ^r_k before summation. Bellman structure is broken.

### Method B: Discount → Normalize (MT-GRPO)

```
A_{i,t} = (R_{i,t} - μ^R_t) / σ^R_t
```

Divides the **combined** (immediate + future) deviation by σ^R_t. When the future return dominates the variance of R_{i,t}, the immediate reward signal is suppressed.

Concretely: if rewards are in {-1, 0, 1} and T=5, then:
- Immediate deviation |r_{i,t} - μ^r_t| is O(1)
- Future return R_{i,t+1} spans O(T) in range
- σ^R_t is dominated by variance in future returns
- So (r_{i,t} - μ^r_t) / σ^R_t becomes small

### Method C: DC-GRPO (proposed)

```
A_{i,t} = (r_{i,t} - μ^r_t) / (σ^r_t + ε)  +  γ · (R_{i,t+1} - μ^R_{t+1}) / (σ^R_{t+1} + ε)
          ─────────────────────────────────      ───────────────────────────────────────────────
          I_{i,t}  (immediate term)               F_{i,t}  (future term)
```

For the last turn (t = T):

```
A_{i,T} = (r_{i,T} - μ^r_T) / (σ^r_T + ε)
```

Each component is normalized by **its own scale**:
- Immediate credit is measured against σ^r_t (reward-scale)
- Future credit is measured against σ^R_{t+1} (return-scale)

Neither can drown out the other.

---

## Why DC-GRPO Solves the Refusal Problem

For a refusal trajectory with rewards [-1, 1, 1, 1, 1]:

**MT-GRPO (Method B)**: R_{i,1} = -1 + γ·(~3.94) ≈ 2.90. This is compared against μ^R_1 and divided by σ^R_1. The -1 is buried inside a return of 2.90.

**DC-GRPO (Method C)**: The immediate term I_{i,1} = (-1 - μ^r_1) / σ^r_1, which is a large negative number evaluated purely at reward scale. The future term F_{i,1} is positive but added separately. The immediate penalty survives.

---

## Relationship to PPO

PPO's TD error:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

DC-GRPO's advantage (unnormalized form):

```
A_{i,t} ∝ (r_{i,t} - μ^r_t) + γ · (R_{i,t+1} - μ^R_{t+1})
```

Correspondence:
- μ^r_t plays the role of E[r_t] (part of V(s_t))
- μ^R_{t+1} plays the role of V(s_{t+1})
- No learned value function required

---

## Bias Analysis

Both baselines μ^r_t and μ^R_{t+1} include trajectory i in the average. This introduces O(1/G) dependence on a_{i,t}. This is the same bias as standard GRPO. At G=8, the bias is ~12.5% and vanishes as G → ∞.

The per-component normalization (dividing by σ^r_t and σ^R_{t+1}) rescales the gradient magnitude but not its direction. This does not introduce additional bias.

**DC-GRPO is an asymptotically unbiased policy gradient estimator with O(1/G) bias, identical to standard GRPO.**

---

## Single-Turn Equivalence

When T = 1:

```
A_{i,1} = (r_{i,1} - μ^r_1) / (σ^r_1 + ε) + 0
        = standard GRPO advantage
```

DC-GRPO reduces to GRPO exactly. ∎

---

## Implementation

```python
def compute_dc_grpo_advantages(rewards, gamma=0.99, eps=1e-8):
    """
    Args:
        rewards: Tensor [G, T]
    Returns:
        advantages: Tensor [G, T]
    """
    G, T = rewards.shape
    advantages = torch.zeros_like(rewards)

    # Compute discounted returns R_{i,t} for all i, t
    # R[:, t] = r[:, t] + gamma * R[:, t+1]
    R = torch.zeros(G, T + 1)
    for t in range(T - 1, -1, -1):
        R[:, t] = rewards[:, t] + gamma * R[:, t + 1]

    for t in range(T):
        # Immediate term: I_{i,t} = (r_{i,t} - μ^r_t) / σ^r_t
        r_t = rewards[:, t]
        mu_r = r_t.mean()
        sigma_r = r_t.std()
        I_t = (r_t - mu_r) / (sigma_r + eps)

        if t < T - 1:
            # Future term: F_{i,t} = (R_{i,t+1} - μ^R_{t+1}) / σ^R_{t+1}
            R_future = R[:, t + 1]
            mu_R = R_future.mean()
            sigma_R = R_future.std()
            F_t = (R_future - mu_R) / (sigma_R + eps)

            advantages[:, t] = I_t + gamma * F_t
        else:
            advantages[:, t] = I_t

    return advantages
```

---

## Summary

| Property | A: Norm→Disc | B: Disc→Norm | C: DC-GRPO |
|----------|-------------|-------------|-----------|
| Bellman consistent | ✗ | ✓ | ✓ |
| Immediate credit preserved | ✗ | ✗ (diluted by σ^R_t) | **✓** (own scale σ^r_t) |
| Future credit preserved | mixed | ✓ | **✓** (own scale σ^R_{t+1}) |
| Unbiased | ✗ | ✓ O(1/G) | ✓ O(1/G) |
| Value function needed | ✗ | ✗ | ✗ |
| Single-turn = GRPO | ✓ | ✓ | ✓ |
