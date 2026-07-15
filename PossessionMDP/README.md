# Possession-Clock Markov Decision Processes for Football

*A team- and lineup-specific nonstationary MDP framework for simulating football
possessions, adapted from the shot-clock MDP methodology used in basketball
play modeling, and applied to Ecuador's Liga Pro event data.*

## Abstract

We model football possessions as episodes from team-specific nonstationary
Markov decision processes (MDPs) with transition probabilities that depend on
the *possession clock* — the time elapsed since a team won the ball. Bayesian
hierarchical models parametrize these transition probabilities, borrowing
strength across lineups (the specific eleven players on the pitch), across
opponent match-ups, and through the season. To keep inference and simulation
tractable, we combine lineup-specific MDPs into a single team-average MDP
using a transition-weighting scheme: the team-average process is constructed
so that its expected transition count for any state pair equals the
minutes-weighted sum of the expected counts from the separate lineup-specific
MDPs. We use the resulting nonstationary MDPs to drive a possession simulator
in which parameter uncertainty is propagated forward via posterior draws of
the hierarchical model. After fitting the model to Ecuadorian Liga Pro
event data, we simulate matches both on-policy (as observed) and under
altered tactical policies, and measure the resulting change in possession
efficiency (expected goals, shot volume, turnover rate). We close with a
discussion of the game-theoretic complications that arise when a team
credibly commits to an altered policy — namely, that the transition
probabilities we fit are themselves a function of what opponents expect a
team to do, so counterfactual simulation of a fixed policy against a fixed
defense is a first-order approximation, not an equilibrium.

## 1. Introduction

Possession-based sports share a common structural feature: within a
possession, a team makes a sequence of decisions (which action to attempt)
that produce a sequence of on-ball states (where the ball is, how much time
remains under some clock), until the possession ends in a score, a loss of
the ball, or a stoppage. The basketball literature has formalized this as a
team-specific, shot-clock-dependent MDP: the state captures ball location and
shot-clock value, the action captures the decision made with the ball, and
the transition captures both the outcome of that decision and the resulting
next state. Because basketball's 24-second shot clock imposes a hard,
common, and exogenous deadline, it is a natural nonstationarity driver — teams
behave differently with 20 seconds left on the shot clock than with 3.

Football has no shot clock, but it has a close analogue: the **possession
clock**, i.e. the number of seconds a team has controlled the ball since the
turnover, restart, or recovery that started the current possession. Empirically,
the character of a possession changes with this clock in ways that echo the
basketball shot clock — early in a possession, teams probe and circulate the
ball with a low turnover rate and a low shot rate; as the clock runs on
without progress, passing lanes close, defensive shape compresses, and both
shot attempts and turnovers become more likely, because the defense has had
time to get organized around the ball rather than being caught in transition.
Unlike the shot clock, the possession clock is unbounded and endogenous (it
resets on every turnover rather than counting down to a forced action), but it
plays the same statistical role: it is the variable that makes the transition
probabilities *nonstationary within an episode*, and it is the variable a
hierarchical model must condition on to avoid pooling together possessions
that are behaviorally very different.

This document adapts the basketball possession-MDP framework to football
possessions, using the same three structural ideas:

1. **Team- and lineup-specific nonstationary MDPs**, indexed by the
   possession clock, in place of the shot clock.
2. **Bayesian hierarchical parametrization** of the transition probabilities,
   pooling across lineups, across the roster's turnover through substitutions,
   and through the season.
3. **A transition-weighting scheme** that combines lineup-specific MDPs into
   a single team-average MDP whose expected transition counts equal the
   exposure-weighted sum of the lineup-specific expected counts, making
   season-long, multi-lineup simulation computationally tractable without
   discarding the lineup structure in the data.

We then build a possession simulator from the fitted MDPs, propagate
posterior uncertainty into simulated outcomes, and use the simulator to
compare on-policy and altered-policy season simulations. An accompanying
implementation (`mdp_model.py`, `run_analysis.py` in this directory) fits
this model to Ecuadorian Liga Pro event data (Opta/Stats Perform F24-style
feeds, already used elsewhere in this repository for xT, pi-ratings, and
header-clearance modeling) and produces the comparison described in
Section 6.

## 2. Football Possessions as Episodes of a Nonstationary MDP

### 2.1 Episodes

An **episode** is a single team's possession: it begins the moment that team
gains controlled use of the ball (a recovery, an interception, a restart) and
ends when the ball is lost, a shot is taken, or play is stopped in a way that
hands the opponent the restart. This mirrors the basketball paper's treatment
of a possession as an episode that terminates in a score, a turnover, or the
clock/period expiring.

### 2.2 State space

The state at any point within an episode is

$$s = (z, c)$$

where:

- $z \in \{1, \dots, Z\}$ is the **pitch zone** the ball occupies, taken from a
  coarse grid over the attacking half-space of the pitch (we use a
  length-by-width grid, e.g. 6 × 4 = 24 zones, with $x=100$ always meaning
  "closest to the opponent's goal" in the possessing team's attacking
  direction — the same normalization convention already used by the xT model
  elsewhere in this repository);
- $c \in \{1, \dots, C\}$ is the **possession-clock bucket**, a discretization
  of seconds elapsed since the possession began (e.g. 0–5s, 5–10s, 10–15s,
  15–25s, 25s+), playing the role the shot-clock bucket plays in basketball.

Two absorbing outcome states close every episode: $\text{GOAL}$ and
$\text{TURNOVER}$ (a non-scoring shot is a special, low-probability-of-goal
member of the terminal shot outcome, and is tracked separately for
efficiency reporting even though, from the MDP's point of view, both a saved
shot and a lost ball end the possession).

### 2.3 Action space

At each non-terminal state the team on the ball chooses an action

$$a \in \{\text{advance}, \text{sideways}, \text{back}, \text{cross}, \text{shot}\}$$

capturing the *attempted* decision — the direction and type of the pass,
carry, or shot the player on the ball attempts — independent of whether it
succeeds. This mirrors the basketball MDP's separation of a decision (e.g.
"attempt a three-pointer") from its outcome (make, miss, or foul): here,
"advance" is the decision to attempt to progress the ball into the next zone
toward goal, and its *outcome* — successful progression, an intercepted or
misplaced pass, a foul won — is captured by the transition, not the action
label.

### 2.4 Transitions and nonstationarity

Given a state-action pair $(s, a)$ with $s=(z,c)$, the transition distribution

$$P\big(s' \mid s, a\big), \qquad s' \in \{1,\dots,Z\}\times\{1,\dots,C+\} \cup \{\text{GOAL}, \text{TURNOVER}\}$$

gives the probability of each destination zone (with the clock advanced to
the next bucket), or of the possession ending in a goal or a turnover. The
distribution is explicitly allowed to depend on $c$: a pass attempted from
the same zone $z$ behaves differently at $c=$"0–5s" (freshly won ball, defense
not yet set) than at $c=$"25s+" (defense compressed, passing lanes tighter,
elevated turnover risk). This is the direct football analogue of shot-clock
dependent shot-attempt and turnover probabilities in the basketball model,
and it is the reason a single, clock-collapsed transition matrix per zone
would understate both how dangerous early-possession transition moments are
and how turnover-prone a stalled possession becomes.

## 3. Bayesian Hierarchical Parametrization

Football produces far fewer possessions per team per season than basketball
does (roughly 700–900 events per match rather than ~100 possessions per team
per game translating into thousands of shot-clock states), and the
possession-clock × zone × action state space is large relative to the data
available for any one lineup. Naively estimating $P(s'\mid s,a)$ separately
for each of the eleven-player lineups a team fields over a season would leave
most cells with a handful of observations or none. We therefore borrow
strength using a three-level Bayesian hierarchy, with a Dirichlet–Multinomial
conjugate structure at each level (each $(s,a)$ cell's outcome distribution is
a categorical distribution over destination states, so a Dirichlet prior on
that distribution is updated by observed transition counts into a Dirichlet
posterior in closed form):

$$
\begin{aligned}
\alpha^{\text{league}}(s,a) &= \kappa_{\text{league}} \cdot \bar p^{\text{league}}(s,a) \\
\alpha^{\text{team}}_T(s,a) &= \kappa_{\text{team}} \cdot \bar p^{\text{league}}(s,a) \;+\; n^{\text{team}}_T(s,a) \\
\alpha^{\text{lineup}}_{T,L}(s,a) &= \kappa_{\text{lineup}} \cdot \bar p^{\text{team}}_T(s,a) \;+\; n^{\text{lineup}}_{T,L}(s,a)
\end{aligned}
$$

where $\bar p^{\text{league}}(s,a)$ and $\bar p^{\text{team}}_T(s,a)$ are the
posterior-mean outcome distributions at the level above, $\kappa_{\text{league}},
\kappa_{\text{team}}, \kappa_{\text{lineup}}$ are concentration
hyperparameters controlling how strongly each level is pulled toward its
parent (equivalently, how many "pseudo-possessions" of prior belief each
level contributes), and $n(s,a)$ is the vector of observed transition counts
out of $(s,a)$ at that level. A sparsely visited $(s,a)$ cell for a given
lineup — say, a back-up striker's partnership in the final ten minutes of a
handful of matches — is therefore shrunk heavily toward that team's overall
tendency in that zone/clock state, while a frequently visited cell for a
settled first-choice lineup is dominated by its own data. This directly
mirrors the basketball paper's use of hierarchical models to borrow strength
across players; here the pooling axis is lineup composition rather than
individual shot-takers, because football's transition probabilities are a
property of how a group of players combine (a winger's crossing tendency
interacts with who is making the run into the box), not of a single player in
isolation.

**Borrowing through time.** Within a season, team- and lineup-level counts
are accumulated with an exponential recency weight, $w_k = \rho^{\,(K-k)}$ for
matchday $k$ of $K$ played so far ($\rho$ slightly below 1), so that the
current transition estimate for a team reflects its most recent tactical
identity more than its August form, while still using the full season's data
rather than an arbitrarily truncated recent window.

## 4. From Lineup MDPs to a Team-Average MDP

A team fields many distinct lineups over a season (starting XI changes,
substitutions, injuries, suspensions, rotation), and each lineup induces its
own MDP with its own transition probabilities $P_L(s'\mid s,a)$. Simulating a
season faithfully would require carrying every lineup's MDP forward
separately and switching between them at every substitution boundary — exact,
but computationally unwieldy once posterior uncertainty and multi-episode
season simulation are layered on top. We instead define a single
**team-average MDP** per team, constructed so that it reproduces, in
expectation, the transition behavior actually generated by the mix of
lineups the team used.

Let $w_L$ be the team's **lineup exposure weight** — the share of the team's
total possession-time (or possession count) played by lineup $L$ over the
window being modeled — and let $n_L(s,a)$ be the (posterior-expected) number
of times lineup $L$ took action $a$ in state $s$. Define the team-average
transition distribution as the exposure- and volume-weighted mixture of the
lineup-specific posterior means:

$$
P_T\big(s' \mid s,a\big) \;=\; \frac{\sum_L w_L \, n_L(s,a) \, P_L\big(s' \mid s,a\big)}{\sum_L w_L \, n_L(s,a)}.
$$

**Property (expected-count consistency).** By construction, the expected
number of $(s,a)\to s'$ transitions the team-average MDP generates when it is
"used" for $N$ total visits to $(s,a)$ equals the exposure-weighted sum of the
expected counts the individual lineup MDPs would generate for those same $N$
visits:

$$
N \cdot P_T(s'\mid s,a) \;=\; \sum_L w_L \, n_L(s,a)\, P_L(s'\mid s,a) \cdot \frac{N}{\sum_L w_L\,n_L(s,a)},
$$

so that, taking the weights $w_L\,n_L(s,a)$ as the allocation of the $N$
visits across lineups, the team-average process's expected transition count
for the state pair $(s,a,s')$ is exactly the weighted sum of the
expected counts of the separate lineup-specific MDPs — the same defining
property used to build team-average MDPs from lineup MDPs in the basketball
model. This is stronger than simply pooling raw transition counts across
lineups before normalizing (which would silently overweight whichever lineup
happened to visit $(s,a)$ most often, independent of how much of the team's
overall play that lineup represents); weighting by $w_L$ first ties the
team-average process to how the manager actually used the squad, which is
what a "team policy" should mean when the team fielded several different
lineups.

In practice we compute $P_T$ from the *posterior* lineup distributions
$P_L$ (i.e. using the hierarchical posterior means from Section 3, not raw
empirical proportions), so that thin lineup samples are already shrunk
before being folded into the team average, and the team-average MDP inherits
well-calibrated uncertainty rather than the noise of the least-used lineups.

## 5. The Possession Simulator

The simulator draws whole possessions forward from the fitted MDPs while
propagating parameter uncertainty:

1. **Posterior draw.** For each state-action pair, draw a transition vector
   $\tilde P(\cdot \mid s,a) \sim \text{Dirichlet}\big(\alpha_{T}(s,a)\big)$
   (team-average posterior parameters) or, for lineup-level analysis,
   $\text{Dirichlet}\big(\alpha_{T,L}(s,a)\big)$. This single draw defines one
   internally-consistent "possible world" for how the team's possessions
   behave.
2. **Episode rollout.** Starting from an empirical distribution over
   possession-start zones (where teams actually win the ball), repeatedly:
   sample an action from the policy $\pi(a\mid s)$ (see below), sample the
   next state from the drawn $\tilde P(\cdot\mid s,a)$, and advance the
   possession clock, until the episode reaches $\text{GOAL}$,
   $\text{TURNOVER}$/no-goal shot, or a maximum-length censor.
3. **Repeat** across many posterior draws and many rollouts per draw. Because
   step 1 is repeated per draw, the spread across draws (not just across
   rollouts within a draw) reflects genuine estimation uncertainty in the
   fitted MDP, giving credible intervals on any simulated season-level
   statistic rather than only Monte Carlo sampling noise.

**Policy vs. dynamics.** We deliberately separate the *policy*
$\pi(a\mid s)$ — how often a team chooses to attempt to advance, go sideways,
go back, cross, or shoot from a given zone/clock state — from the
*dynamics* $P(s'\mid s,a)$ fit in Sections 3–4, which describe how the world
responds once a decision is attempted (whether the pass finds a teammate,
whether the defense wins the ball back, whether the shot goes in). This split
is what makes altered-policy simulation meaningful: we can ask "what happens
if this team attempts more crosses from the wide final third and fewer
sideways passes" by reweighting $\pi$ directly, while leaving the fitted
$P(s'\mid s,a)$ (execution quality, defensive response) exactly as observed.

## 6. On-Policy and Altered-Policy Simulation

`run_analysis.py` fits the model above per team from the available Liga Pro
matches, then runs the simulator under two policies:

- **On-policy**: $\pi$ estimated empirically from the team's own action
  choices per (zone, clock-bucket) state, hierarchically pooled the same way
  as the transition probabilities.
- **Altered policy ("directness")**: a mechanical reweighting of $\pi$ in the
  middle and attacking thirds that shifts probability mass from
  `sideways`/`back` toward `advance`/`cross`, renormalized to remain a valid
  distribution, holding the fitted dynamics fixed. This is a stand-in for the
  kind of tactical question the basketball paper's altered-policy simulations
  address (e.g. "shoot more/fewer threes") — here, "build more directly."

For each policy we report, per simulated possession and with posterior
credible intervals: probability of ending in a goal, probability of ending in
a shot, mean turnover rate, and mean possession-clock at termination. The
comparison isolates the *simulated* efficiency consequence of a policy change
under the team's own fitted execution quality — it does not, by itself, tell
us whether the policy change is a good idea once the opponent adapts (Section
7).

## 7. Game-Theoretic Considerations

The transition probabilities $P(s'\mid s,a)$ fit in Sections 3–4 are not
policy-invariant. They were generated by defenses that were reacting to the
team's *actual*, observed tendencies: if a team rarely crosses from deep wide
positions, opposing full-backs do not need to respect that threat and can
squeeze infield, which is part of why the fitted turnover/completion rates
for crossing look the way they do in the data. Simulating an altered policy
("attempt far more crosses") while holding $P(s'\mid s,a)$ fixed therefore
implicitly assumes the opponent's defensive shape does not change in
response — a reasonable first-order approximation for a small, one-off
deviation, but increasingly wrong the larger and more sustained the policy
shift, and wrong in a specific, predictable direction: sustained changes
should make the altered policy look *less* favorable in reality than the
fixed-dynamics simulation suggests, because a defense that has adapted will
suppress exactly the state transitions the new policy tries to exploit.

This is the same caveat that applies to counterfactual policy simulation in
the basketball setting, and it has the same two practical implications here.
First, altered-policy simulation is best read as a bound on the *immediate,
surprise-value* upside of a tactical change, not a steady-state prediction —
which argues for validating any promising altered policy in-season, in short
bursts, before committing to it as a full-match identity. Second, because
both teams in a match are simultaneously choosing $\pi$ against dynamics that
depend on the opponent's fitted tendencies, the "correct" policy is properly
a best response in a game between two teams' policies rather than a
single-agent optimization — an equilibrium concept, not a maximization one.
Extending the simulator to iterate simulated policy adjustments against a
simulated opponent response (a simple fictitious-play loop over the fitted
MDPs) is the natural next step for turning this from a one-shot
counterfactual tool into an equilibrium-aware one, and is left for future
work.

## 8. Data and Scope

The implementation in this directory fits the model to the Opta/Stats
Perform F24-style event feeds already used elsewhere in this repository
(`Event/*.json`, Ecuadorian Liga Pro, 2026 season), using the same
`typeId`/`qualifierId` schema as the existing xT, pi-ratings, and
header-clearance models. Possession episodes, lineups (via starting-XI and
substitution events), and zone/clock states are reconstructed directly from
the raw event stream — see `mdp_model.py` for the exact possession-chain
segmentation rules and their simplifying assumptions (documented inline,
since a few whistle-level edge cases — e.g. exactly how a loose ball after a
blocked clearance is attributed — are collapsed for tractability rather than
modeled with full set-piece detail).

## 9. Limitations

- The zone/clock/action discretization is coarse by design; finer grids
  would need materially more matches of data per lineup before hierarchical
  shrinkage stops dominating the lineup-level estimates.
- Set pieces (corners, free kicks, throw-ins) are folded into the general
  turnover/restart handling rather than modeled as their own sub-MDPs with
  set-piece-specific routines, unlike the bespoke corner-routine and
  clearance-landing models already present elsewhere in this repository,
  which could be composed with this framework in future work.
- The game-theoretic discussion in Section 7 is qualitative; no
  opponent-adaptation term is currently fit into the model.
