---
title: Structural Estimation of Life Cycle Models with Wealth in the Utility
subject: Economics
# subtitle: Evolve your markdown documents into structured data
short_title: Structural Estimation using HARK
authors:
  - name: Alan Lujan
    affiliations:
      - Ohio State University
      - Econ-ARK
    email: alanlujan91@gmail.com
license: CC-BY-4.0
keywords: structural estimation, life cycle, wealth in the utility
exports:
  - format: tex+pdf
    template: arxiv_nips
    output: structural_estimation.pdf
    show_date: true
---

+++ {"part": "abstract"}

Heterogeneous Agent Models (HAMs) are a powerful tool for understanding the effects of monetary and fiscal policy on the economy. However, state of the art frameworks such as HANK have been unable to replicate the observed hoarding of wealth at the very top of the distribution and generally lack important life cycle properties such as time-varying mortality and income risk. On the one hand, the inability to pin down wealth at the tail of the distribution has been a problem for HANK models precisely because it has implications for the transmission of monetary and fiscal policy. On the other hand, agents in HANK are generally conceived as perpetual youths with infinite horizons without age-specific profiles of mortality and income risk. This is problematic as it ignores the effects of these policies on potentially more affected communities, such as young families with children or the low-wealth elderly. In this paper, I investigate the effects of both life cycle considerations as well as wealth in the utility on the structural estimation of HAMs. Structural estimation is the first step in evaluating the effect of monetary and fiscal policies in a HANK framework, and my hope is that this paper will lead to better models of the economy that can be used to inform policy..

+++

+++ {"part": "acknowledgements"}

I would like to thank my advisor, Chris Carroll, for his guidance and support throughout this project, as well as the members of the Econ-ARK team for their support and for providing a great community to work in.

+++

# Introduction

# Life Cycle Models

\begin{align}
  {\vFunc}_{t}({m}_{t}) & = \max_{{c}_{t}}~~~ \uFunc({c}_{t})+\beth\Alive_{t+1}\hat{\DiscFac}_{t+1}
  \Ex_{t}[(\PermShk_{t+1}\PermGroFac_{t+1})^{1-\CRRA}{\vFunc}_{t+1}({m}_{t+1})]                                 \\
                        & \text{s.t.}                                                               & \nonumber \\
  {a}_{t}               & = {m}_{t}-{c}_{t} \nonumber
  \\  {m}_{t+1}  & = {a}_{t}\underbrace{\left(\frac{\Rfree}{\PermShk_{t+1}\PermGroFac_{t+1}}\right)}_{\equiv \RNrm_{t+1}}+ ~\TranShkEmp_{t+1}
\end{align}

\begin{align}
  \Alive _{t}^{t+n} & : & \text{probability to }\Alive\text{ive until age $t+n$ given alive at age $t$}
  \\  \hat{\DiscFac}_{t}^{t+n} &:&\text{age-varying discount factor between ages $t$ and $t+n$}
  \\     \Psi_{t} &:&\text{mean-one shock to permanent income}
  \\     \beth &:&\text{time-invariant `pure' discount factor}
\end{align}

\begin{align}
  \Xi_{s}           & =
  \begin{cases}
      0\phantom{/\pZero}     & \text{with probability $\pZero>0$}                                                                                                            \\
      \TranShkEmp_{s}/\pZero & \text{with probability $(1-\pZero)$, where $\log \TranShkEmp_{s}\thicksim \mathcal{N}(-\sigma_{\TranShkEmp}^{2}/2,\sigma_{\TranShkEmp}^{2})$} \\
  \end{cases} \\
  \log \PermShk_{s} & \thicksim \mathcal{N}(-\sigma_{\PermShk}^{2}/2,\sigma_{\PermShk}^{2})
\end{align}

## Wealth in the Utility Function

### Separable Utility

### Non-separable Utility

### Generalized Endogenous Grid Method

# Calibration and Estimation

# Conclusion

# References

[](doi:10.3386/w7826)
[](doi:10.3386/w6549)
[](doi:10.1162/rest_a_00893)
[](doi:10.3982/ECTA17434)
[](doi:10.3386/w26941)
[](doi:10.1257/aer.20160042)
[](doi:10.3386/w26647)