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
    line_spacing: doublespacing
---

+++ {"part": "abstract"}

Heterogeneous Agent Models (HAM) are a powerful tool for understanding the effects of monetary and fiscal policy on the economy. However, state of the art frameworks such as Heterogeneous Agent New Keynsian (HANK) models have been unable to replicate the observed hoarding of wealth at the very top of the distribution and generally lack important life cycle properties such as time-varying mortality and income risk. On the one hand, the inability to pin down wealth at the tail of the distribution has been a problem for HANK models precisely because it has implications for the transmission of monetary and fiscal policy. On the other hand, agents in HANK are generally conceived as perpetual youth with infinite horizons and without age-specific profiles of mortality and income risk. This is problematic as it ignores the effects of these policies on potentially more affected communities, such as young families with children or the low-wealth elderly. In this paper, I investigate the effects of both life cycle considerations as well as wealth in the utility on the structural estimation of HAMs. Structural estimation is the first step in evaluating the effect of monetary and fiscal policies in a HANK framework, and my hope is that this paper will lead to better models of the economy that can be used to inform policy..

+++

+++ {"part": "acknowledgements"}

I would like to thank my advisor, Chris Carroll, for his guidance and support throughout this project, as well as the members of the Econ-ARK team for providing a great collaborative community to work in.

+++

# Introduction

# Life Cycle Incomplete Markets Models

An important extension to the Standard Incomplete Markets (SIM) model is the Life Cycle Incomplete Markets (LCIM) model as in [](doi:10.1198/073500103288619007), [](doi:10.1111/1468-0262.00269), and [](doi:10.1111/1467-937X.00092), among others. The LCIM model is a natural extension to the SIM model that allows for age-specific profiles of preferences, mortality, and income risk. 

## The Baseline Model

The agent's objective is to maximize present discounted utility from consumption over the life cycle with a terminal period of $T$:

\begin{equation}
  \vFunc_{t}(\pLvl_{t},\mLvl_{t})  =    \max_{\{\cFunc\}_{t}^{T}} ~ \uFunc(\cLvl_{t})+\Ex_{t}\left[\sum_{n=1}^{T-t} {\beth}^{n} \Alive_{t}^{t+n}\hat{\DiscFac}_{t}^{t+n} \uFunc(\cLvl_{t+n}) \right]   \label{eq:lifecyclemax}
\end{equation}

where $\pLvl_{t}$ is the permanent income level, $\mLvl_{t}$ is total market resources, $\cLvl_{t}$ is consumption, and  

\begin{align}
    \beth & :  \text{time-invariant `pure' discount factor}
    \\ \Alive _{t}^{t+n} & :  \text{probability to }\Alive\text{ive until age $t+n$ given alive at age $t$}
    \\ \hat{\DiscFac}_{t}^{t+n} & :  \text{age-varying discount factor between ages $t$ and $t+n$.}
\end{align}

It will be convenient to work with the problem in permanent-income-normalized form as in [](doi:10.3386/w10867), which allows us to reduce a 2 dimensional problem of permanent income and wealth into a 1 dimensional problem of wealth normalized by permanent income. The recursive Bellman equation can be expressed as: 

\begin{align}
    {\vFunc}_{t}({m}_{t}) & = \max_{\cNrm_{t}} ~ \uFunc(\cNrm_{t})+\beth\Alive_{t+1}\hat{\DiscFac}_{t+1}
    \Ex_{t}[(\PermShk_{t+1}\PermGroFac_{t+1})^{1-\CRRA}{\vFunc}_{t+1}({m}_{t+1})]
    \\ & \text{s.t.} & \nonumber
    \\ \aNrm_{t} & = {m}_{t}-\cNrm_{t} \nonumber
    \\ {m}_{t+1} & = \aNrm_{t}\underbrace{\left(\frac{\Rfree}{\PermShk_{t+1}\PermGroFac_{t+1}}\right)}_{\equiv \RNrm_{t+1}}
    + ~\TranShkEmp_{t+1}
\end{align}

where $\cNrm$, $\aNrm$, and $\mNrm$ are consumption, assets, and market resources normalized by permanent income, respectively, and

\begin{align}
  \PermShk_{t+1} & :  \text{mean-one shock to permanent income}
    \\ \PermGroFac_{t+1} & :  \text{permanent income growth factor}
    \\ \TranShkEmp_{t+1} & :  \text{transitory shock to permanent income}
    \\ \RNrm_{t+1} & :  \text{permanent income growth normalized return factor}
\end{align}

and all other variables are defined as above. The transitory and permanent shocks to income are defined as:

\begin{align}
\TranShkEmp_{s}  = &
    \begin{cases}
        0\phantom{/\pZero} & \text{with probability $\pZero>0$}
        \\ \xi_{s}/\pZero & \text{with probability $(1-\pZero)$, where
            $\log \xi{s}\thicksim \mathcal{N}(-\sigma_{\xi}^{2}/2,\sigma_{\xi}^{2})$}
    \end{cases}
    \\ \phantom{/\pZero} \\ & \text{and }  \log \PermShk_{s}   \thicksim \mathcal{N}(-\sigma_{\PermShk}^{2}/2,\sigma_{\PermShk}^{2}).
  \end{align}


## Wealth in the Utility Function

A simple extension to the Life Cycle Incomplete Markets (LCIM) model is to include wealth in the utility function. [](doi:10.3386/w6549) argues that models in which the only driver of wealth accumulation is consumption smoothing are not consistent with the saving behavior of the wealthiest households. Instead, they propose a model in which households derive utility from their level of wealth itself or they derive a flow of services from political power and social status, calling it the `Capitalist Spirit' model. In turn, we can add this feature to the LCIM model by adding a utility function with consumption and wealth as follows:

\begin{align}
    {\vFunc}_{t}({m}_{t}) & = \max_{\cNrm_{t}} ~ \uFunc(\cNrm_{t}, \aNrm_{t})+\beth\Alive_{t+1}\hat{\DiscFac}_{t+1}
    \Ex_{t}[(\PermShk_{t+1}\PermGroFac_{t+1})^{1-\CRRA}{\vFunc}_{t+1}({m}_{t+1})]
    \\ & \text{s.t.} & \nonumber
    \\ \aNrm_{t} & = {m}_{t}-\cNrm_{t} \nonumber
    \\ {m}_{t+1} & = \aNrm_{t}\RNrm_{t+1}+ ~\TranShkEmp_{t+1}
\end{align}

### Separable Utility

[](doi:10.3386/w6549) presents extensive empirical and informal evidence for a LCIM model with wealth in the utility function. Specifically, the paper uses a utility that is separable in consumption and wealth:

\begin{equation}
    \uFunc(\cNrm_{t}, \aNrm_{t}) = \frac{\cNrm_{t}^{1-\CRRA}}{1-\CRRA}
    + \kapShare_{t} \frac{(\aFunc_{t} - \underline\aNrm)^{1-\wealthShare}}{1-\wealthShare}
\end{equation}

where $\kapShare$ is the relative weight of the utility of wealth and $\wealthShare$ is the relative risk aversion of wealth. 

### Non-separable Utility

A different model that we will explore is one in which the utility function is non-separable in consumption and wealth; i.e. consumption and wealth are complimentary goods. In the case of the LCIM model, this dynamic complementarity drives the accumulation of wealth not only for the sake of wealth itself, but also because it increases the marginal utility of consumption. 

\begin{equation}
    \uFunc(\cNrm_{t}, \aNrm_{t}) = \frac{(\cNrm_{t}^{1-\wealthShare} (\aNrm_{t} - \underline\aNrm)^\wealthShare)^{1-\CRRA}}{(1-\CRRA)}
\end{equation}

# Solution Methods

For a brief departure, let's consider how we solve these problems generally. Define the post-decision value function as:

\begin{align}
    \DiscFac_{t+1} \wFunc_{t}(\aNrm_{t}) & = \beth\Alive_{t+1}\hat{\DiscFac}_{t+1}
    \Ex_{t}[(\PermShk_{t+1}\PermGroFac_{t+1})^{1-\CRRA}{\vFunc}_{t+1}({m}_{t+1})]
    \\ & \text{s.t.}
    \\ {m}_{t+1} & = \aNrm_{t}\RNrm_{t+1}+ ~\TranShkEmp_{t+1}
\end{align}

For our purposes, it will be useful to simplify the notation a bit by dropping time subscripts. The recursive problem can then be written as:

\begin{align}
    \vFunc(\mRat) & = \max_{\cNrm} ~ \uFunc(\cNrm, \aNrm) + \DiscFac \wFunc(\aRat)
    \\ & \text{s.t.}
    \\ \aNrm & = \mRat-\cNrm
\end{align}

## Endogenous Grid Method, Abridged

In the standard incomplete markets (SIM) model, the utility function is simply $\uFunc(\cNrm)$ and the Euler equation is $\uFunc'(\cNrm) = \DiscFac \wFunc'(\aNrm)$, which is a necessary and sufficient condition for an internal solution of the optimal choice of consumption. If $\uFunc(\cNrm)$ is differentiable and its marginal utility is invertible, then the Euler equation can be inverted to obtain the optimal consumption function as $\cNrm(\aNrm) = \uFunc'^{-1}(\DiscFac \wFunc'(\aNrm))$. Using an _exogenous_ grid of post-decision savings $\aMat$, we can obtain an _endogenous_ grid of market resources $\mMat$ by using the budget constraint $\mNrm(\aMat) = \aMat + \cNrm(\aMat)$ such that this collection of grids satisfy the Euler equation. This is the endogenous grid method (EGM) of [](doi:10.1016/j.econlet.2005.09.013).

In the presence of wealth in the utility function, the Euler equation is more complicated and may not be invertible in terms of optimal consumption. Consider the first order condition for an optimal combination of consumption and savings, denoted by $^*$:

\begin{equation}
    \uFunc_{c}'(\cNrm^*, \aNrm^*) - \uFunc_{a}'(\cNrm^*, \aNrm^*) = \DiscFac \wFunc'(\aNrm^*)
\end{equation}

If the utility of consumption and wealth is additively separable, then the Euler equation can be written as $\uFunc_{c}'(\cNrm) = \uFunc_{a}'(\aNrm) + \DiscFac \wFunc'(\aNrm)$. This makes sense, as the agent will equalize the marginal utility of consumption with the marginal utility of wealth today plus the discounted marginal value of wealth tomorrow. In this case, the EGM is simple: we can invert the Euler equation to obtain the optimal consumption policy as $\cNrm(\aNrm) = \uFunc_{c}'^{-1}\big(\uFunc_{a}'(\aNrm) + \DiscFac \wFunc'(\aNrm)\big)$. We can proceed with EGM as usual, using the budget constraint to obtain the endogenous grid of market resources $\mNrm(\aMat) = \aMat + \cNrm(\aMat)$.

## Generalized Endogenous Grid Method

When the utility of consumption and wealth is not additively separable, the Euler equation is not analytically invertible for the optimal consumption policy. The usual recourse is to use a root-finding algorithm to obtain the optimal consumption policy for each point on the grid of market resources, which turns out to be more efficient than grid search maximization. As far as I am aware, this is the first paper to avoid root-finding or grid search maximization by using an EGM-like approach to solve a problem with a non-invertible Euler equation.

Holding $\aNrm$ constant, define a function $f_{a}$ as the difference between the marginal utility of consumption and the marginal utility of wealth, and using an _exogenous_ grid of $\cMat$, label the difference as $[\xFer_{a}]$:

\begin{equation}
    f_{a}(\cMat) = \uFunc_{c}'(\cMat, \aNrm) - \uFunc_{a}'(\cMat, \aNrm) = [\xFer_{a}]
\end{equation}

So far, we haven't done much, as $f_a$ is still not analytically invertible. However, as we have held $\aNrm$ constant, this is now a single variable function with a single output. We can easily create a linear interpolator for the inverse of this function by reversing output with input appropriately. This is the key step that allows us to avoid root-finding or grid search maximization. We will call this function $\hat{f}_{a}^{-1}$, as it is an approximation of the inverse of $f_{a}$ which could still introduce some error[^f_error].

[^f_error]: A main difference between EGM and GEGM is that EGM uses the exact inverse of the Euler equation, thereby giving the unique optimal consumption policy for each point on the grid of post-decision savings that exactly satisfies the Euler equation. GEGM uses an approximation of the inverse of the Euler equation, thereby giving an approximate optimal consumption policy for each point on the grid of market resources. With careful grid choice, the approximation error can be made arbitrarily small.

\begin{equation}
    \hat{f}_{a}^{-1}([\xFer_{a}]) = \cMat
\end{equation}

We can now construct the function $g(\aNrm)$ as the composition of $\hat{f}_{a}^{-1}$ and discounted marginal value of wealth $\DiscFac \wFunc'$, which should provide an approximation of the optimal consumption policy for each point on the grid of post-decision savings:

\begin{equation}
    g(\aNrm^*) = \hat{f}_{a^*}^{-1} \big( \DiscFac \wFunc'(\aNrm^*) \big) = \cNrm^*
\end{equation}

This completes the modified step in the Generalized Endogenous Grid Method (GEGM). If this looks familiar, it is because it exactly parallels the EGM method. In the SIM model, $f_a(\cNrm,\aNrm) = \uFunc'(\cNrm)$, which does not depend on $\aNrm$, so we can drop it. The inverse of this function is exactly $f^{-1}(\xFer) = \uFunc'^{-1}(\xFer)$, so we don't need an approximating interpolator. The composition of this function with the discounted marginal value of wealth exactly provides the consumption policy as $g(\aNrm) = \uFunc'^{-1} \big( \DiscFac \wFunc'(\aNrm) \big) = \cNrm$. GEGM is a generalization of EGM to the case where the Euler equation is not analytically invertible.

An additional feature of GEGM is that the inverse interpolating function $\hat{f}_{a}^{-1}$ only needs to be constructed once and can be used for all backward iterations, as long as the utility function isn't time-varying. This is because the inverse interpolating function only depends on the marginal utilities of consumption and wealth, whose parameters are constant in our specification. This also means that we can construct this interpolating function from the onset of our process to have arbitrary precision by optimally choosing the grid of $\cMat$. Computationally, this adds just one additional step to the standard EGM at the beginning of the process, inheriting its substantial improvements in speed and accuracy relative to root finding and grid search maximization[^speed].

[^speed]: If additional precision is needed, the resulting $\hat{\cNrm}$ can be used as the initial guess for a root-finding algorithm with arbitrary precision, which should converge in few iterations due to the proximity of the initial guess to the true solution.

# Quantitative Strategy

## Calibration

## Estimation

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