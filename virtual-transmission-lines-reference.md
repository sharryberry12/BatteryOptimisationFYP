# Robust Virtual Transmission Lines via Negative Imaginary Systems Theory: Technical Reference Guide

This reference document compiles the complete mathematical foundations, system definitions, control laws, stability theorems, and simulation parameters from the paper *"Negative Imaginary Power Grids: Robust Virtual Transmission Line Technology"* by Elizabeth L. Ratnam, Kanghong Shi, Yijun Chen, and Ian R. Petersen. It is structured for direct programmatic or theoretical use in developer tools like Claude Code.

---

## 1. System Dynamics & The Classical Equal Area Criterion

### 1.1 Single-Machine Infinite-Bus (SMIB) Swing Dynamics
An undamped generator connected to an infinite bus via a lossless transmission line is modeled by the swing equation:

$$M\ddot{\delta} = P_M - P_E(\delta)$$

Where:
* $\delta \in \mathbb{R}$: Rotor/power angle difference between the generator and the infinite bus.
* $M > 0$: Generator inertia coefficient.
* $P_M$: Mechanical power injection from the turbine (treated as a constant under fault conditions).
* $P_E(\delta) \approx P_{\max} \sin\delta$: Electrical power exported to the infinite bus, where $P_{\max}$ is the maximum power transfer limit.

### 1.2 Post-Fault Equilibrium
Let $\bar{\delta}$ represent the post-fault stable equilibrium angle, satisfying:

$$P_M = P_{\max} \sin\bar{\delta}, \quad 0 < \bar{\delta} < \frac{\pi}{2}$$

The system has an unstable equilibrium point at $\pi - \bar{\delta}$. To maintain transient stability, the maximum angle swing $\delta_m$ must satisfy $\delta_m < \pi - \bar{\delta}$ to avoid generator pole-slip.

### 1.3 Critical Energy Boundary ($E_{cr}$)
Using the classical Equal Area Criterion (EAC), the accelerating kinetic energy area $A_1$ (where $P_M > P_E$) is matched against the decelerating potential energy area $A_2$ (where $P_E > P_M$). The critical stability energy limit $E_{cr}$ where the EAC fails (i.e., $\delta_m = \pi - \bar{\delta}$) is:

$$E_{cr} = \int_{\bar{\delta}}^{\pi - \bar{\delta}} (P_{\max}\sin\delta - P_M) d\delta = P_{\max} \left( 2\cos\bar{\delta} + (2\bar{\delta} - \pi)\sin\bar{\delta} \right)$$

---

## 2. Nonlinear Negative Imaginary (NI) Systems Theory

### 2.1 General State-Space Formulation
Consider a nonlinear multiple-input multiple-output (MIMO) system:

$$\begin{aligned}
\dot{x} &= f(x, u) \\
y &= h(x) + g(u)
\end{aligned}$$

Where $x \in \mathbb{R}^n$ is the state, $u \in \mathbb{R}^m$ is the input, $y \in \mathbb{R}^m$ is the output. $f$ is locally Lipschitz, $h$ is continuously differentiable, and $g$ is continuous.

#### Definition 1: Negative Imaginary (NI)
The system is nonlinear NI if there exists a continuously differentiable positive semidefinite storage function $V(x) \ge 0$ ($V(0) = 0$) such that:

$$\dot{V}(x) \le u^T \dot{h}(x), \quad \forall t \ge 0$$

#### Definition 2: Output Strictly Negative Imaginary (OSNI)
The system is nonlinear OSNI if there exists a storage function $V(x) \ge 0$ and a scalar $\epsilon > 0$ such that:

$$\dot{V}(x) \le u^T \dot{h}(x) - \epsilon \|\dot{h}(x)\|^2, \quad \forall t \ge 0$$

---

### 2.2 Closed-Loop Positive-Feedback Stability (Theorem 1)
Consider the interconnection of an NI system $H_1$ and an OSNI system $H_2$:

$$\begin{aligned}
H_1 &: \dot{x}_1 = f_1(x_1, u_1), \quad y_1 = h_1(x_1) \\
H_2 &: \dot{x}_2 = f_2(x_2, u_2), \quad y_2 = h_2(x_2) + g_2(u_2)
\end{aligned}$$

Under the positive feedback interconnection:

$$u_1 = y_2, \quad u_2 = y_1$$

The Lur'e-Postnikov-type closed-loop Lyapunov candidate $W(x_1, x_2)$ is defined as:

$$W(x_1, x_2) = V_1(x_1) + V_2(x_2) - h_1(x_1)^T h_2(x_2) - \sum_{k=1}^m \int_0^{h_1^k(x_1)} g_2^k(v^k) dv^k$$

#### System Assumptions for Theorem 1:
1. **Assumption 1**: $W(x_1, x_2)$ is locally positive definite in a neighborhood of the origin.
2. **Assumption 2**: If the state-dependent output component $h(x)$ is constant, then the state $x$ is constant.
3. **Assumption 3**: If the system state $x$ is constant, then the input $u$ must be constant.
4. **Assumption 4**: The closed-loop positive-feedback interconnection has no nonzero steady state in the neighborhood of the equilibrium (steady-state loop-gain condition).

#### Theorem 1 Statement:
If Assumptions 1–4 are satisfied, the equilibrium of the closed-loop system at the origin is **locally asymptotically stable**.

---

## 3. Single-Loop NI Control with Battery Storage

### 3.1 Deviation Coordinates & Plant-Feedback Separation
By introducing angle deviations $\tilde{\delta} := \delta - \bar{\delta}$ and damping $D \ge 0$, the post-fault swing dynamics are separated into:

$$\underbrace{M\ddot{\tilde{\delta}} + D\dot{\tilde{\delta}}}_{H_1 \text{ (Linear Plant)}} = \underbrace{P_{\max}\left(\sin\bar{\delta} - \sin(\tilde{\delta} + \bar{\delta})\right)}_{H_2 \text{ (Nonlinear Feedback)}}$$

* **Linear Subsystem $H_1$**: From active power input $u_1$ to angle deviation $y_1 = \tilde{\delta}$.
  
  $$\dot{x}_1 = Ax_1 + Bu_1, \quad y_1 = Cx_1$$
  
  With $x_1 = \begin{bmatrix} \tilde{\delta} & \dot{\tilde{\delta}} \end{bmatrix}^T$, and:
  
  $$A = \begin{bmatrix} 0 & 1 \\ 0 & -D/M \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 1/M \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}$$
  
  This plant is NI with storage function $V_1(x_1) = \frac{1}{2} M \dot{\tilde{\delta}}^2$, yielding $\dot{V}_1 \le u_1 \dot{y}_1$.
* **Static Feedback $H_2$**: Output $y_2 = P_{\max}\left(\sin\bar{\delta} - \sin(u_2 + \bar{\delta})\right)$ with $u_2 = y_1$. This is NI with $V_2 \equiv 0$.

### 3.2 Uncontrolled Lyapunov Function
The Lyapunov function for the uncontrolled SMIB system is:

$$W(\tilde{\delta}, \dot{\tilde{\delta}}) = \frac{1}{2}M\dot{\tilde{\delta}}^2 + P_{\max}\left(\cos\bar{\delta} - \tilde{\delta}\sin\bar{\delta} - \cos(\tilde{\delta} + \bar{\delta})\right)$$

Its time derivative is:

$$\dot{W} = -D\dot{\tilde{\delta}}^2 \le 0$$

If physical damping $D > 0$, the system converges asymptotically to the origin within the positive invariance region:

$$\Lambda = \left\{ (\tilde{\delta}, \dot{\tilde{\delta}}) \in \mathbb{R}^2 : W(\tilde{\delta}, \dot{\tilde{\delta}}) < W_{cr} \text{ and } |\tilde{\delta} + 2\bar{\delta}| \le \pi \right\}$$

Where the critical Lyapunov level matches the EAC limit:
$$W_{cr} = P_{\max}\left(2\cos\bar{\delta} + (2\bar{\delta} - \pi)\sin\bar{\delta}\right) = E_{cr}$$

### 3.3 Dynamic Battery Storage Control Law
For large disturbances where physical damping $D$ is low or absent ($D=0$), a battery storage system is added to inject/absorb active power $P_{ST}$. The swing equation becomes:

$$M\ddot{\tilde{\delta}} + D\dot{\tilde{\delta}} = P_{\max}\left(\sin\bar{\delta} - \sin(\tilde{\delta} + \bar{\delta})\right) + P_{ST}$$

The **battery storage control law** is defined by:

$$\begin{aligned}
\dot{x}_2 &= -\frac{1}{\tau}x_2 + \frac{\kappa}{\tau}\tilde{\delta} \\
P_{ST} &= x_2 - \sigma\tilde{\delta}
\end{aligned}$$

Where $\tau > 0$ is the time constant, and parameters satisfy $0 < \kappa < \sigma$. 

The corresponding closed-loop Lyapunov function is:

$$W(\tilde{\delta}, \dot{\tilde{\delta}}, x_2) = \frac{1}{2}M\dot{\tilde{\delta}}^2 + P_{\max}\left(\cos\bar{\delta} - \tilde{\delta}\sin\bar{\delta} - \cos(\tilde{\delta} + \bar{\delta})\right) + \frac{1}{2\kappa}(x_2 - \kappa\tilde{\delta})^2 + \frac{\sigma - \kappa}{2}\tilde{\delta}^2$$

Differentiating yields:

$$\dot{W} = -D\dot{\tilde{\delta}}^2 - \frac{1}{\kappa\tau}(x_2 - \kappa\tilde{\delta})^2 \le 0$$

*Key Result:* The battery dynamic control loop provides strict negative energy dissipation even if physical machine damping is completely absent ($D = 0$).

---

## 4. Multi-Machine Network Consensus & Robust VTLs

### 4.1 Graph-Based Multi-Machine Modeling
Let the transmission network be represented by an undirected connected graph $G = (V, E)$ with $N$ generator nodes ($V$) and $L$ edges ($E$). 
* $Q \in \mathbb{R}^{N \times L}$ is the incidence matrix defining edge orientations.
* For each generator node $i \in V$, the nominal swing dynamics are:

  $$M_i \ddot{\delta}_i + D_i \dot{\delta}_i = P_{M i} - P_{L i} - \sum_{j \in \mathcal{N}_i} P_{\max, ij} \sin(\delta_i - \delta_j)$$

* In deviation coordinates $\tilde{\delta}_i = \delta_i - \bar{\delta}_i$, the networked system is:

  $$M_i \ddot{\tilde{\delta}}_i + D_i \dot{\tilde{\delta}}_i = \sum_{j \in \mathcal{N}_i} P_{\max, ij} \left( \sin\psi_{ij} - \sin(\psi_{ij} + \tilde{\delta}_i - \tilde{\delta}_j) \right)$$

  Where $\psi_{ij} = \bar{\delta}_i - \bar{\delta}_j$ is the steady-state inter-node operating angle.

### 4.2 Networked Consenus Interconnection Architecture
The aggregated node dynamics $H_p$ and aggregated edge controllers $H_c$ are coupled via:

$$U_c = (Q^T \otimes I_m) Y_p, \quad U_p = (Q \otimes I_m) Y_c$$

Where:
* Node plants $H_{pi}$ are **output strictly NI** due to synchronous physical generator damping $D_i > 0$.
* Edge controllers $H_{cl}$ representing lines are **NI**.
* Using the Lur'e-Postnikov Lyapunov candidate $\hat{W}$, the system converges to the invariant set where $\dot{\hat{W}} \le 0$. Under a connected topology, this guarantees **local output consensus**:

  $$\lim_{t \to \infty} \|y_{pi}(t) - y_{pj}(t)\| = 0, \quad \forall i, j \in V$$

---

### 4.3 Robust Virtual Transmission Line (VTL) Control Law
Select edges $E_B \subseteq E$ are retrofitted with VTLs (collocated battery storage devices at endpoint nodes).
* **Uncontrolled Edges ($e_l \in E \setminus E_B$)**: Static physical line power-flow equation:
  
  $$y_{cl} = \varphi_l(u_{cl}) = P_{\max, l}\left(\sin\psi_l - \sin(\psi_l + u_{cl})\right)$$

* **VTL Controlled Edges ($e_l \in E_B$)**: Dynamic feedback augmented with battery storage:
  
  $$\begin{aligned}
  \dot{x}_{cl} &= -\frac{1}{\tau_l} x_{cl} + \frac{\kappa_l}{\tau_l} u_{cl} \\
  y_{cl} &= \varphi_l(u_{cl}) + x_{cl} - \sigma_l u_{cl}
  \end{aligned}$$
  
  With parameters satisfying $\tau_l > 0$ and $\sigma_l > \kappa_l > 0$. The battery active power injection at generator node $i$ is:
  
  $$P_{ST, i} = \sum_{l \in E(i) \cap E_B} q_{il} (x_{cl} - \sigma_l u_{cl})$$

#### System-Wide VTL Lyapunov Function:
$$\hat{W}_{ps} = \sum_{i \in V} \frac{1}{2} M_i \dot{\tilde{\delta}}_i^2 + \sum_{l \in E_B} \left( \frac{(x_{cl} - \kappa_l u_{cl})^2}{2\kappa_l} + \frac{\sigma_l - \kappa_l}{2} u_{cl}^2 \right) + \sum_{l \in E} P_{\max, l} \left[ \cos\psi_l - u_{cl} \sin\psi_l - \cos(\psi_l + u_{cl}) \right]$$

Differentiating yields:

$$\dot{\hat{W}}_{ps} = -\sum_{i \in V} D_i \dot{\tilde{\delta}}_i^2 - \sum_{l \in E_B} \frac{\tau_l}{\kappa_l} \dot{x}_{cl}^2 \le 0$$

This guarantees frequency synchronisation ($\dot{\tilde{\delta}}_i \to 0$) and local output consensus ($u_{cl} \to 0$) across the grid.

---

## 5. Benchmark System: Modified Kundur Two-Area Four-Machine System

This benchmark is modeled based on the standard Kundur system implemented in **OPAL-RT HYPERSIM** at a grid frequency of $60\text{ Hz}$.

### 5.1 Topology & Adjacency
* **Nodes ($V$)**: $\{G_1, G_2, G_3, G_4\}$ co-located with battery units.
* **Edges ($E$)**: $\{(G_1, G_2), (G_3, G_4), (G_2, G_4)\}$.
* **Areas**: Area 1 $\{G_1, G_2\}$, Area 2 $\{G_3, G_4\}$. Inter-area tie-line transfers $413\text{ MW}$ under steady state.

### 5.2 Parameters and Numerical Values

| Parameter | Symbol / Formula | Nominal Value / Expression |
| :--- | :--- | :--- |
| **Grid Frequency** | $f_0$ | $60\text{ Hz}$ |
| **Area 1 Load Resistance** | $R_1$ | $54.705\ \Omega$ |
| **Area 2 Load Resistance** | $R_2$ | $29.937\ \Omega$ (reassigned to $23\ \Omega$ under high-load stress) |
| **Area 1 Load Capacitance** | $C_1$ | $19.406\ \mu\text{F}$ |
| **Area 2 Load Capacitance** | $C_2$ | $26.926\ \mu\text{F}$ |
| **Load Inductances** | $L_1, L_2$ | $1.403\text{ H}$ |
| **VTL Edge Controllers ($l \in \{1,2,3\}$)** | $\dot{x}_{cl}$ | $-3x_{cl} + 8u_{cl}$ |
| **VTL Control Output ($l \in \{1,2,3\}$)** | $\hat{y}_{cl}$ | $x_{cl} - 5u_{cl}$ |
| **Implied VTL Parameters** | $\tau_l, \kappa_l, \sigma_l$ | $\tau_l = 0.333\text{ s}, \quad \kappa_l = 2.667, \quad \sigma_l = 5.0$ |
| **Actuator Power Saturation** | $P_{ST, \max}$ | $\pm 150\text{ MW}$ |

### 5.3 Simulated Disturbances
1. **Three-Phase Fault**: A three-phase-to-ground fault is simulated at the midpoint of one of the $220\text{ km}$ inter-area tie-lines.
2. **Timeline of Event Sequence**:
   * **$t = 5.0\text{ s}$**: Fault occurs.
   * **$t = 5.1\text{ s}$**: Tie-line circuit breakers disconnect the faulted line.
   * **$t = 5.2\text{ s}$**: Fault clears.
   * **$t = 30.0\text{ s}$**: Disconnected line is reconnected.
3. **High Load Stress**: Shunt resistance in Area 2 is reduced to $R_2 = 23\ \Omega$, shifting inter-area power transfer to $603\text{ MW}$. Under this stress, the uncontrolled and standard IEEE PSS1A/PSS4B stabilizers fail (resulting in unstable oscillations), whereas the NI-VTL controller preserves frequency synchronisation and dampens the oscillations.
