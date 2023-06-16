---
title: Structural Estimation of Life Cycle Models with Wealth in the Utility Function
subject: Economics
# subtitle: Evolve your markdown documents into structured data
short_title: Structural Estimation
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

Heterogeneous Agent Models (HAM) are a powerful tool for understanding the effects of monetary and fiscal policy on the economy. However, state of the art frameworks such as Heterogeneous Agent New Keynsian (HANK) models have been unable to replicate the observed hoarding of wealth at the very top of the distribution and generally lack important life cycle properties such as time-varying mortality and income risk. On the one hand, the inability to pin down wealth at the tail of the distribution has been a problem for HANK models precisely because it has implications for the transmission of monetary and fiscal policy. On the other hand, agents in HANK are generally conceived as perpetual youth with infinite horizons and without age-specific profiles of mortality and income risk. This is problematic as it ignores the effects of these policies on potentially more affected communities, such as young families with children or the low-wealth elderly. In this paper, I investigate the effects of both life cycle considerations as well as wealth in the utility on the structural estimation of HAMs. Structural estimation is the first step in evaluating the effect of monetary and fiscal policies in a HANK framework, and my hope is that this paper will lead to better models of the economy that can be used to inform policy..

+++

+++ {"part": "acknowledgements"}

I would like to thank my advisor, Chris Carroll, for his guidance and support throughout this project, as well as the members of the Econ-ARK team for providing a great collaborative community to work in.

+++

# Introduction

# Life Cycle Models

## The Baseline Model

The agent's objective is to maximize present discounted utility from consumption over a last cycle with a terminal period of $T$: 

\begin{equation}\label{eq:MaxProb}
  \max ~ \uFunc({\cLvl}_{t}) + \Ex_{t}\left[ \sum_{n=1}^{T-t} {\beth}^{n} \Alive_{t}^{t+n}\hat{\DiscFac}_{t}^{t+n} \uFunc({\cLvl}_{t+n})\right]
\end{equation}

where 

\begin{align}
       \beth &:&\text{time-invariant `pure' discount factor}
       \\ \Alive _{t}^{t+n} & : & \text{probability to }\Alive\text{ive until age $t+n$ given alive at age $t$}
  \\  \hat{\DiscFac}_{t}^{t+n} &:&\text{age-varying discount factor between ages $t$ and $t+n$}
\end{align}



\begin{align}
  {\vFunc}_{t}({m}_{t}) & = \max_{\cNrm_{t}} ~  \uFunc(\cNrm_{t})+\beth\Alive_{t+1}\hat{\DiscFac}_{t+1}
  \Ex_{t}[(\PermShk_{t+1}\PermGroFac_{t+1})^{1-\CRRA}{\vFunc}_{t+1}({m}_{t+1})]                                 \\
                        & \text{s.t.}                                                               & \nonumber \\
  \aNrm_{t}               & = {m}_{t}-\cNrm_{t} \nonumber
  \\  {m}_{t+1}  & = \aNrm_{t}\underbrace{\left(\frac{\Rfree}{\PermShk_{t+1}\PermGroFac_{t+1}}\right)}_{\equiv \RNrm_{t+1}}+ ~\TranShkEmp_{t+1}
\end{align}

\begin{align}
      \Psi_{t} &:&\text{mean-one shock to permanent income}
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

\begin{align}
  {\vFunc}_{t}({m}_{t}) & = \max_{\cNrm_{t}} ~  \uFunc(\cNrm_{t}, \aNrm_{t})+\beth\Alive_{t+1}\hat{\DiscFac}_{t+1}
  \Ex_{t}[(\PermShk_{t+1}\PermGroFac_{t+1})^{1-\CRRA}{\vFunc}_{t+1}({m}_{t+1})]                                 \\
                        & \text{s.t.}                                                               & \nonumber \\
  \aNrm_{t}               & = {m}_{t}-\cNrm_{t} \nonumber
  \\  {m}_{t+1}  & = \aNrm_{t}\RNrm_{t+1}+ ~\TranShkEmp_{t+1}
\end{align}

### Separable Utility

\begin{equation}
  \uFunc(\cNrm_{t}, \aNrm_{t}) = \frac{\cNrm_{t}^{1-\CRRA}}{1-\CRRA} + \kapShare_{t} \frac{(\aFunc_{t} - \underline\aNrm)^{1-\wealthShare}}{1-\wealthShare}
\end{equation}

### Non-separable Utility

\begin{equation}
  \uFunc(\cNrm_{t}, \aNrm_{t}) = \frac{(\cNrm_{t}^{1-\wealthShare} (\aNrm_{t}- \underline\aNrm)^\wealthShare)^{1-\CRRA}}{(1-\CRRA)} 
\end{equation}

### Generalized Endogenous Grid Method

\begin{align}
  \vFunc(\mRat) & = \max_{\cNrm} ~ \uFunc(\cNrm, \aNrm) + \DiscFac \wFunc(\aRat) 
  \\ & \text{s.t.} 
  \\ \aNrm & = \mRat-\cNrm
\end{align}

\begin{equation}
  \uFunc_{c}'(\cNrm^*, \aNrm^*) - \uFunc_{a}'(\cNrm^*, \aNrm^*) = \DiscFac \wFunc'(\aNrm^*) 
\end{equation}

\begin{equation}
  f_{a}(\cNrm) = \uFunc_{c}'(\cNrm, \aNrm) - \uFunc_{a}'(\cNrm, \aNrm) = \xFer_{a}
\end{equation}

\begin{equation}
  f_{a}^{-1}(\xFer_{a}) = \cNrm
\end{equation}

\begin{equation}
  g(\aNrm^*) = f_{a^*}^{-1} \big( \DiscFac \wFunc'(\aNrm^*) \big) = \cNrm^*
\end{equation}

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
[](doi:10.1198/073500103288619007)