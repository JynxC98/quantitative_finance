# Broadie–Kaya Exact Simulation of the Heston Model — Theory

This document derives every formula implemented in this repository, in the
same notation used by the code (`HestonParams`: `kappa, theta, sigma, v_u,
v_t, dt, v0, rho`). It is meant to be read side-by-side with
`helpers/char_function.hpp`, `helpers/solvers.hpp`,
`helpers/integrated_variance.hpp` and `src/models.cpp`.

Primary reference: M. Broadie & Ö. Kaya, *"Exact Simulation of Stochastic
Volatility and Other Affine Jump Diffusion Processes"*, Operations Research,
54(2), 2006.

---

## 1. The Heston model

Under the risk-neutral measure, the asset price $S_t$ and its instantaneous
variance $V_t$ solve

$$
\begin{aligned}
dS_t &= r\,S_t\,dt + \sqrt{V_t}\,S_t\,dW_t^{S} \\
dV_t &= \kappa(\theta - V_t)\,dt + \sigma\sqrt{V_t}\,dW_t^{V}
\end{aligned}
$$

with $\operatorname{Corr}(dW_t^S, dW_t^V) = \rho\,dt$, mean-reversion speed
$\kappa$, long-run variance $\theta$, vol-of-vol $\sigma$, and initial
variance $V_0 = v_0$. $V_t$ is a Cox–Ingersoll–Ross (CIR) process; the Feller
condition $2\kappa\theta \ge \sigma^2$ keeps it strictly positive, though the
scheme below remains valid even when it is violated.

The Broadie–Kaya scheme simulates $(S_t, V_t)$ over a step $[u,t]$,
$\Delta = t-u$ (`p.dt`), **exactly** — no discretization bias — in three
stages:

1. Draw $V_t \mid V_u$ exactly (§2).
2. Draw the integrated variance $\displaystyle Y=\int_u^t V_s\,ds$
   conditional on $(V_u, V_t)$, by inverting its CDF (§3–§5).
3. Draw $\log S_t \mid (V_u, V_t, Y)$ exactly, since conditional on the
   variance path it is Gaussian (§6).

---

## 2. Exact transition of the CIR process

### 2.1 Non-central chi-squared law

Conditional on $V_u$, $V_t$ has a scaled non-central chi-squared
distribution (Cox, Ingersoll & Ross, 1985):

$$
V_t \;=\; \frac{\sigma^2\bigl(1-e^{-\kappa\Delta}\bigr)}{4\kappa}\;
\chi'^2_d(\lambda),
\qquad
d = \frac{4\kappa\theta}{\sigma^2},
\qquad
\lambda = \frac{4\kappa e^{-\kappa\Delta}}{\sigma^2\bigl(1-e^{-\kappa\Delta}\bigr)}\,V_u .
$$

$d$ = degrees of freedom, $\lambda$ = non-centrality parameter. This is
`sampleVt()` in `src/integrated_variance.cpp`:

```
scaling_factor = sigma^2 (1 - e^{-kappa dt}) / (4 kappa)
dof            = 4 kappa theta / sigma^2
lambda_        = 4 kappa e^{-kappa dt} v_u / (sigma^2 (1 - e^{-kappa dt}))
V_t            = scaling_factor * ChiSquared'(dof, lambda_)
```

### 2.2 PDF (Wikipedia normalization, used by `NonCentralChi2PDF`)

$$
f_{\chi'^2_d(\lambda)}(z) = \tfrac12 e^{-(z+\lambda)/2}
\left(\frac{z}{\lambda}\right)^{d/4-1/2} I_{d/2-1}\!\bigl(\sqrt{\lambda z}\bigr),
\qquad z>0,
$$

where $I_\alpha$ is the modified Bessel function of the first kind, order
$\alpha = d/2-1$.

### 2.3 Sampling via Poisson–Gamma mixture

Directly inverting the non-central chi-squared CDF is numerically fragile.
Instead we use the well-known mixture representation:

$$
N \sim \mathrm{Poisson}\!\left(\frac{\lambda}{2}\right), \qquad
\chi'^2_d(\lambda) \mid N \;\sim\; \chi^2_{d+2N},
\qquad
\chi^2_{k} \;\stackrel{d}{=}\; \mathrm{Gamma}\!\left(\frac{k}{2},\,2\right).
$$

i.e. draw $N\sim\mathrm{Poisson}(\lambda/2)$, then draw a central
$\mathrm{Gamma}(d/2+N,\,2)$ variate. This is exactly what
`SampleNonCentralChi2()` does in `src/non_central_chi_sqd.cpp`, and it avoids
ever evaluating $I_{d/2-1}$ for sampling (only the PDF path needs the Bessel
evaluation).

---

## 3. Characteristic function of the integrated variance

Let $Y = \int_u^t V_s\,ds$. Broadie & Kaya (2006, Eq. 7–9) give the
conditional characteristic function

$$
\Phi(a) \;=\; \mathbb{E}\!\left[e^{iaY}\;\middle|\;V_u, V_t\right]
$$

in closed form. Define

$$
\gamma(a) \;=\; \sqrt{\kappa^2 - 2\sigma^2 i a}.
$$

Then

$$
\Phi(a) =
\underbrace{\frac{\gamma(a)\,e^{-\frac12(\gamma(a)-\kappa)\Delta}\bigl(1-e^{-\kappa\Delta}\bigr)}
{\kappa\bigl(1-e^{-\gamma(a)\Delta}\bigr)}}_{\text{term 1}}
\;\cdot\;
\underbrace{\exp\!\left[\frac{V_u+V_t}{\sigma^2}
\left(\frac{\kappa(1+e^{-\kappa\Delta})}{1-e^{-\kappa\Delta}}
-\frac{\gamma(a)(1+e^{-\gamma(a)\Delta})}{1-e^{-\gamma(a)\Delta}}\right)\right]}_{\text{term 2}}
\;\cdot\;
\underbrace{\frac{I_{\,0.5d-1}\!\bigl(\sqrt{V_uV_t}\,\tfrac{4\gamma(a)e^{-0.5\gamma(a)\Delta}}{\sigma^2(1-e^{-\gamma(a)\Delta})}\bigr)}
{I_{\,0.5d-1}\!\bigl(\sqrt{V_uV_t}\,\tfrac{4\kappa e^{-0.5\kappa\Delta}}{\sigma^2(1-e^{-\kappa\Delta})}\bigr)}}_{\text{term 3}} ,
$$

with $d=4\kappa\theta/\sigma^2$ as before, $\alpha = 0.5d-1$ the Bessel
order. When $a=0$: $\gamma(0)=\kappa$, term 1 $\to 1$, term 2 $\to 1$, term 3
$\to 1$, so $\Phi(0)=1$ as required for any characteristic function — this is
the `std::abs(u) < 1e-12` guard at the top of `CharFunction`.

This is implemented verbatim in `src/char_function.cpp`: `const_gamma`
$=\gamma(a)$, `first_term`, `second_term`, `third_term` (evaluated as
`exp(log I_num - log I_den)` to avoid overflow — both Bessel arguments blow
up like $e^{|z|}$ for real-ish $z$, so the ratio is computed in log-space via
`ModifiedBessel(..., log_space=true)`, §8).

**Sanity properties** (both verified against the implementation):
- $|\Phi(a)| \le 1$ for real $a$ (Φ is a characteristic function).
- $\Phi(-a) = \overline{\Phi(a)}$ (Hermitian symmetry, since $Y$ is real).

---

## 4. Recovering the CDF: Gil-Pelaez inversion

Given $\Phi$, the Gil-Pelaez (1951) inversion formula recovers the CDF of
the (a.s. non-negative, absolutely continuous) random variable $Y$:

$$
F(x) \;=\; \frac12 - \frac1\pi \int_0^\infty
\frac{\operatorname{Im}\!\bigl[e^{-iux}\Phi(u)\bigr]}{u}\,du .
$$

Equivalently, using $\Phi(-u)=\overline{\Phi(u)}$ to symmetrize,

$$
F(x) \;=\; \frac{2}{\pi}\int_0^\infty \frac{\sin(ux)}{u}\,\operatorname{Re}\bigl[\Phi(u)\bigr]\,du .
$$

This is `calculateCDF` / `CDFIntegrand` in `helpers/solvers.hpp`. The
integrand has a removable singularity at $u=0$
($\sin(ux)/u \to x$, and $\operatorname{Re}\Phi(0)=1$), handled by the
explicit small-$u$ branch returning $x/\pi$ before the $2/\pi$ prefactor.

A small imaginary shift `damp` is folded into the CF argument
($\Phi(u+i\!\cdot\!\text{damp})$) together with an explicit
$e^{-\text{damp}\cdot u}$ multiplier, to further tame the tail of the
oscillatory integral before it is truncated at a finite upper limit — a
standard damped-Fourier-inversion trick (in the spirit of Carr–Madan /
Lord–Kahl). It introduces a $O(\text{damp})$ bias, which is why `damp` is
kept very small ($5\times10^{-5}$).

**Truncation.** The integral is cut off at a `critical_freq` where $|\Phi(u)|$
first drops below a tolerance (`findCriticalfreq`), then evaluated as a sum
of 32-point Gauss–Legendre panels over $[0,\text{critical\_freq}]$
(`legendreIntegrate`, `helpers/quadrature.hpp`, `helpers/legendre_nodes.hpp`).

> **Correctness note.** `critical_freq` depends on the *specific* Heston
> parameters $(\kappa,\theta,\sigma,V_u,V_t,\Delta)$ passed to
> `calculateIntegral`, and must be recomputed for every distinct call — it
> cannot be cached across calls with different `p`. (An earlier version of
> this file cached it in a `static` local, which silently reused the
> integration range computed from whichever `(V_u,V_t)` happened to be
> evaluated first — see the Appendix.)

The corresponding PDF and its derivative — needed for Halley's method below
— follow by differentiating under the integral sign:

$$
f(x) = \frac{2}{\pi}\int_0^\infty \cos(ux)\,\operatorname{Re}[\Phi(u)]\,du,
\qquad
f'(x) = -\frac{2}{\pi}\int_0^\infty u\,\sin(ux)\,\operatorname{Re}[\Phi(u)]\,du .
$$

These are `PDFIntegrand` / `d_PDFIntegrand`.

---

## 5. Quantile inversion: Halley's method

To sample $Y$, draw $U\sim\mathrm{Uniform}(0,1)$ and solve $F(x)=U$ for $x$.
Define $g(x) = F(x) - U$. Halley's method (second-order Newton) uses both
$g'=f$ and $g''=f'$ for cubic convergence:

$$
x_{n+1} \;=\; x_n \;-\; \frac{2\,g(x_n)\,g'(x_n)}{2\,g'(x_n)^2 - g(x_n)\,g''(x_n)} .
$$

This is `runNewtonSolver` in `helpers/solvers.hpp`. The initial guess uses
the trapezoidal estimate of the mean,

$$
x_0 = \frac{V_u+V_t}{2}\,\Delta,
$$

which is exact in the $\Delta\to0$ limit (see the `test_first_moment_sanity`
first-moment check).

**Tail handling.** Halley's method is a local, high-order method — it can
misbehave when $F$ is very flat (deep in a tail, $F'(x)\approx0$). For
$U<0.05$ or $U>0.90$ the solver instead brackets the root
($[10^{-7}, \text{hi}]$, doubling `hi` until $F(\text{hi})>U$) and runs plain
bisection to `tolerance`, trading convergence order for guaranteed
robustness where Halley's method is least reliable.

**CDF table caching.** Since Halley/bisection each require several
Gauss–Legendre CDF/PDF evaluations, and a full Monte Carlo run needs one
quantile inversion *per timestep per path*, `helpers/cdf_table.hpp`
precomputes $F(x)$ on a grid of $x\in[0.001,\,u_\epsilon]$
(with $u_\epsilon = \mu_1 + 10\sigma_1$, a generous Gaussian
moment-matched upper bound — see `calculateUEpsilon`/`buildCDFTable`) for a
grid of $(V_u,V_t)$ pairs, and inverts by linear interpolation
(`sampleFromTable`) instead of solving Halley's method online. The grid is
serialized to `cdf_cache.bin` (`saveCDFTableGrid`/`loadCDFTableGrid`); it is
**only valid for the exact $(\theta,\kappa,\sigma,\Delta, n_v,\text{n\_points})$
combination used to build it** — delete the cache file if any of these
change.

---

## 6. Exact log-price update

### 6.1 The variance Brownian integral, in closed form

Integrating the CIR SDE over $[u,t]$:

$$
V_t - V_u = \kappa\theta\Delta - \kappa Y + \sigma\!\int_u^t\!\sqrt{V_s}\,dW_s^V
\quad\Longrightarrow\quad
\int_u^t \sqrt{V_s}\,dW_s^V
= \frac{1}{\sigma}\Bigl(V_t - V_u - \kappa\theta\Delta + \kappa Y\Bigr).
$$

Crucially, the right-hand side is a function only of quantities we have
already sampled ($V_u, V_t, Y$) — **no further stochastic simulation is
needed** to know this integral exactly. This is `VarianceBrownianIntegral`.

### 6.2 Decomposing the correlated Brownian motion

Write $dW_t^S = \rho\,dW_t^V + \sqrt{1-\rho^2}\,dW_t^{\perp}$ with
$W^\perp \perp W^V$. Then

$$
\log S_t - \log S_u
= r\Delta - \tfrac12 Y + \rho\!\int_u^t\!\sqrt{V_s}\,dW_s^V
+ \sqrt{1-\rho^2}\int_u^t\!\sqrt{V_s}\,dW_s^{\perp}.
$$

Conditional on the entire variance path (equivalently, conditional on
$V_u,V_t,Y$, since $W^\perp$ is independent of the variance process), the
last term is Gaussian with mean 0 and variance $(1-\rho^2)Y$
(Itô isometry). Hence

$$
\log S_t \;\Big|\; (V_u,V_t,Y) \;\sim\;
\mathcal N\!\Bigl(\log S_u + r\Delta - \tfrac12 Y + \rho\cdot\text{(§6.1)},\;\;(1-\rho^2)\,Y\Bigr).
$$

This is exactly `priceStep` in `src/integrated_variance.cpp`: `mean` as
above, `std_dev = sqrt((1-rho^2) Y)`, then
`price = exp(mean + std_dev * Z)`, $Z\sim N(0,1)$. This step is exact — no
discretization error — which is the entire point of the Broadie–Kaya scheme
versus Euler discretization.

---

## 7. Algorithm summary

For each path, for each of $N$ steps of size $\Delta = T/N$:

```
1.  V_t   ~ NonCentralChiSquared(V_u; kappa, theta, sigma, dt)     (§2)
2.  U     ~ Uniform(0,1)
3.  Y     = F^{-1}(U | V_u, V_t)   via Halley's method / table lookup   (§3–5)
4.  S_t   = exp( log S_u + r*dt - 0.5*Y + rho*IntegralV(V_u,V_t,Y)
                 + sqrt((1-rho^2)*Y) * Z ),   Z ~ N(0,1)              (§6)
5.  V_u  <- V_t;  S_u <- S_t
```

For vanilla European payoffs (the only case exercised here) $N=1$ suffices —
since the scheme is exact, a single step from $0$ to $T$ gives an unbiased
draw of $(S_T,V_T)$ with no discretization bias, unlike Euler. Multiple
steps only matter for path-dependent payoffs.

---

## 8. Supporting numerics

- **Modified Bessel function $I_\alpha(z)$** (`helpers/bessel.hpp`,
  `src/bessel.cpp`): power series for $|z|\le\text{threshold}$,

$$
I_\alpha(z) = \left(\frac z2\right)^{\!\alpha}\sum_{k=0}^\infty
\frac{(z^2/4)^k}{k!\,\Gamma(\alpha+k+1)},
$$

  and the minimum-term-truncated asymptotic expansion for $|z|>\text{threshold}$,

$$
I_\alpha(z) \sim \frac{e^{z}}{\sqrt{2\pi z}}\sum_{k=0}^\infty (-1)^k\frac{a_k(\alpha)}{z^k},
\qquad a_k(\alpha)=\prod_{j=1}^{k}\frac{4\alpha^2-(2j-1)^2}{8k} \ \ (\text{as a term ratio}),
$$

  both evaluated and returned in **log-space** ($\log I_\alpha(z)$) to avoid
  overflow, since $I_\alpha$ grows like $e^{|z|}$ and the CF's Bessel ratio
  (§3, term 3) needs only the *difference* of two such logs.

- **Gamma function** (`helpers/gamma.hpp`): Lanczos approximation with a
  reflection formula for $\operatorname{Re}(z)<\tfrac12$, extended to
  complex arguments (used by the power-series Bessel evaluation).

- **Gauss–Legendre quadrature** (`helpers/quadrature.hpp`,
  `helpers/legendre_nodes.hpp`): a cached 32-point node/weight table built
  once via Newton's method on the Legendre polynomial roots, reused for
  every integral evaluation.

---

## Appendix: implementation gotchas fixed in this pass

- `calculateIntegral`'s `critical_freq` **must not** be memoized in a
  `static` local — it depends on the Heston parameters of the specific call
  (in particular $V_u,V_t$), and different steps/paths/table cells pass
  wildly different parameters. Caching it froze the integration domain to
  whatever was evaluated first, which silently produced non-monotonic,
  badly wrong CDF/PDF values (and hence unreliable Halley/bisection
  quantile inversion) for every other $(V_u,V_t)$ pair — this was the root
  cause of the Halley-solver convergence failures. Fixed by recomputing it
  on every call.
- `simulateBroadieKayaHeston` must loop over all $N$ timesteps
  (§7) and carry `params.v_u` forward step to step; a version that samples
  only a single step regardless of `N` silently ignores the timestep
  parameter for any $N>1$.
- Any cached `cdf_cache.bin` built while the above bug was present (or under
  different `n_v`/`v_min`/grid parameters) is invalid and must be deleted so
  it gets rebuilt — the cache performs no parameter validation on load.
