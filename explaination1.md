# Radial-Kernel-IPS: Detailed Explanation

This document explains the full algorithmic flow we discussed for **radial-kernel-ips** in a 2D box setting, including policy parameterization, sampling, old/new policy probabilities, Jacobian correction, and gradient flow.

---

## 1) High-level idea

Radial-kernel-ips combines:

- **trajectory sampling** from a stochastic policy in continuous action space
- **reward scaling** using a kernel-based inverse-probability estimate
- **group-normalized advantages**
- **PPO-style clipped ratio optimization** using `pi_new/pi_old`

The key difference from a plain Cartesian action policy is that action is modeled in **radial form**:

- angle-like latent: \(u_\theta \in (0,1)\)
- radius-like latent: \(u_r \in (0,1)\)

These are transformed into a 2D action \(a=(a_x,a_y)\).

---

## 2) Policy output structure (for \(K=2\))

Policy outputs \(1 + 6K\) numbers per state.  
For \(K=2\), that is \(1 + 12 = 13\) values:

1. one **stop logit** \(z_{\text{stop}}\)
2. theta mixture branch:
   - logits: \(z^\theta_1, z^\theta_2\)
   - shape raws: \(\tilde\alpha^\theta_1, \tilde\alpha^\theta_2, \tilde\beta^\theta_1, \tilde\beta^\theta_2\)
3. radius mixture branch:
   - logits: \(z^r_1, z^r_2\)
   - shape raws: \(\tilde\alpha^r_1, \tilde\alpha^r_2, \tilde\beta^r_1, \tilde\beta^r_2\)

So a single state determines:
- whether to stop vs continue
- if continue: distribution over angle and radius latent variables

---

## 3) Stop logit: logit vs probability

The stop output is a **logit**, not a normalized probability.

\[
P(\text{stop}\mid s)=\sigma(z_{\text{stop}}), \quad
P(\text{continue}\mid s)=1-\sigma(z_{\text{stop}})
\]

Example:
- \(z_{\text{stop}}=0.7 \Rightarrow \sigma(0.7)=0.668\)
- about 66.8% chance of stop sample on that step

Final terminal decision also depends on boundary/start rules:
- cannot stop at exact start \(s_0\)
- forced-exit near boundary can force terminal behavior

---

## 4) Theta/radius logits: what they mean

The theta and radius logits are also **raw logits**, converted to mixture weights with softmax.

For \(K=2\):
\[
w^\theta_k = \text{softmax}(z^\theta)_k,\quad
w^r_k = \text{softmax}(z^r)_k
\]

Example:
- \(z^\theta=[1.2,-0.3]\Rightarrow w^\theta\approx[0.818,0.182]\)
- \(z^r=[-0.1,0.8]\Rightarrow w^r\approx[0.289,0.711]\)

These are component-selection probabilities for mixture-of-Beta distributions.

---

## 5) Role of \(\alpha,\beta\): how they create \(u\)

Policy does not directly output \(u\).  
It outputs shape parameters that define a Beta distribution, then \(u\) is sampled from that distribution.

Shapes are bounded positive:
\[
\alpha = \beta_{\max}\sigma(\tilde\alpha)+\beta_{\min},\quad
\beta = \beta_{\max}\sigma(\tilde\beta)+\beta_{\min}
\]

Then:
\[
u \sim \text{Beta}(\alpha,\beta),\quad u\in(0,1)
\]

So \(\alpha,\beta\) control **where probability density mass lies on \((0,1)\)**:
- \(\alpha>\beta\): more mass near 1
- \(\alpha<\beta\): more mass near 0
- \(\alpha=\beta=1\): uniform
- \(\alpha,\beta>1\): center-peaked
- \(\alpha,\beta<1\): U-shaped (mass near both ends)

Examples:
- Beta(2,5): mean \(2/7\approx0.286\), favors small values
- Beta(5,2): mean \(5/7\approx0.714\), favors large values
- Beta(0.5,0.5): U-shaped
- Beta(3,3): centered near 0.5

---

## 6) Is mixture density really a weighted sum?

Yes, for both theta and radius:

\[
p_\theta(u_\theta\mid s)=\sum_{k=1}^{K} w^\theta_k\,\text{BetaPDF}(u_\theta;\alpha^\theta_k,\beta^\theta_k)
\]
\[
p_r(u_r\mid s)=\sum_{k=1}^{K} w^r_k\,\text{BetaPDF}(u_r;\alpha^r_k,\beta^r_k)
\]

Important distinction:
- **Density / log_prob** is this weighted sum in probability space (then log taken).
- **Sampling** is not weighted-average of samples:
  1) sample component \(k\) from categorical weights
  2) sample \(u\) from that component’s Beta

---

## 7) End-to-end action sampling (numeric walk-through, \(K=2\))

Assume:
- state \(s=[0.2,0.3]\)
- \(\delta=0.2\)
- not start state, not forced-exit

Suppose policy implies:
- stop probability \(0.668\), sampled continue
- theta mixture sampled \(u_\theta=0.62\)
- radius mixture sampled \(u_r=0.35\)

Transform:
\[
\theta = u_\theta\cdot\frac{\pi}{2}=0.973\ \text{rad}
\]
\[
r_{\max}=\min(\delta,\text{boundary limit})=\min(0.2,0.848)=0.2
\]
\[
r = u_r\cdot r_{\max}=0.07
\]
\[
a_x=r\cos\theta,\quad a_y=r\sin\theta
\]
\[
a\approx[0.0394,0.0578]
\]

If terminal branch is selected, action is replaced by terminal token action.

---

## 8) Old vs new policy probabilities

During rollout, trajectories are sampled using a snapshot policy (**old**).  
During optimization, same recorded state-action pairs are rescored under current policy (**new**).

Trajectory-level:
\[
\log P_{\text{old}}=\sum_t \log\pi_{\text{old}}(a_t\mid s_t),\quad
\log P_{\text{new}}=\sum_t \log\pi_{\text{new}}(a_t\mid s_t)
\]
\[
r=\frac{P_{\text{new}}}{P_{\text{old}}}
=\exp(\log P_{\text{new}}-\log P_{\text{old}})
\]

This ratio goes into PPO clipped objective with advantage.

---

## 9) Continuous action density formula and derivation

For non-terminal action:
\[
\pi(a\mid s)=P(\text{continue}\mid s)\cdot p_\theta(u_\theta\mid s)\cdot p_r(u_r\mid s)\cdot \frac{2}{\pi}\cdot\frac{1}{r_{\max}}\cdot\frac{1}{r}
\]

where:
\[
\theta=\operatorname{atan2}(a_y,a_x),\quad
r=\sqrt{a_x^2+a_y^2},\quad
u_\theta=\frac{2\theta}{\pi},\quad
u_r=\frac{r}{r_{\max}(s,\theta)}
\]

### Why Jacobian factors are needed

Density must transform correctly under variable changes:

1. \(u_\theta \to \theta=\frac{\pi}{2}u_\theta\):
\[
\left|\frac{du_\theta}{d\theta}\right|=\frac{2}{\pi}
\]

2. \(u_r \to r=r_{\max}u_r\):
\[
\left|\frac{du_r}{dr}\right|=\frac{1}{r_{\max}}
\]

3. Polar \((\theta,r)\to\) Cartesian \((a_x,a_y)\):
\[
\left|\frac{\partial(a_x,a_y)}{\partial(\theta,r)}\right|=r
\Rightarrow \text{inverse contributes } \frac{1}{r}
\]

Multiply all three:
\[
\frac{2}{\pi}\cdot\frac{1}{r_{\max}}\cdot\frac{1}{r}
\]

Without this, likelihood values in Cartesian action space are incorrect, and policy gradients become biased.

---

## 10) Log-density form

For non-terminal action:
\[
\log\pi(a\mid s)=
\log P(\text{continue}\mid s)
+\log p_\theta(u_\theta\mid s)
+\log p_r(u_r\mid s)
-\log(\pi/2)-\log r_{\max}-\log r
\]

For terminal action:
\[
\log\pi(a_{\dagger}\mid s)=\log P(\text{stop}\mid s)
\]

---

## 11) Gradient flow: which outputs receive signal

For **non-terminal** action, gradients flow through:
- stop logit (continue term)
- theta logits (mixture weights)
- theta raw alpha/beta outputs (through bounded shape transform and Beta density)
- radius logits (mixture weights)
- radius raw alpha/beta outputs (same)

For **terminal** action, primary signal is through stop logit.

Only \( \pi_{\text{new}} \) is differentiated; \( \pi_{\text{old}} \) is fixed for ratio denominator.

---

## 12) Concrete old/new ratio example (numbers)

Example trajectory with two steps:
- step 1: continue action \(a_1=[0.0394,0.0578]\)
- step 2: terminal action

Suppose:
\[
\log\pi_{\text{old}}(a_1|s_1)=2.925,\quad
\log\pi_{\text{old}}(a_2|s_2)=-0.288
\]
\[
\log P_{\text{old}}=2.637,\quad P_{\text{old}}=e^{2.637}\approx13.98
\]

\[
\log\pi_{\text{new}}(a_1|s_1)=2.412,\quad
\log\pi_{\text{new}}(a_2|s_2)=-0.371
\]
\[
\log P_{\text{new}}=2.041,\quad P_{\text{new}}=e^{2.041}\approx7.70
\]

\[
r=\exp(2.041-2.637)=\exp(-0.596)\approx0.551
\]

So this trajectory became less likely under new policy.

---

## 13) Training signal context

After generating trajectories:
1. compute rewards from final states
2. apply kernel inverse-probability scaling to rewards
3. normalize advantages within groups
4. optimize clipped ratio objective with entropy regularization

This determines whether each trajectory’s likelihood should increase or decrease.

---

## 14) Final takeaways

- The 13 outputs (\(K=2\)) define a **hybrid stop + continuous mixture model**.
- Theta/radius continuous part is a **mixture of Betas on \((0,1)\)**.
- Action density in Cartesian coordinates requires Jacobian correction.
- PPO-style update uses old/new likelihood ratio on whole trajectories.
- Backprop to logits/shape outputs is fully differentiable through the log-prob computation for \( \pi_{\text{new}} \).
