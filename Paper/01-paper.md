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
abbreviations:
    HAM: Heterogeneous Agent Models
    HANK: Heterogeneous Agent New Keynesian
    SIM: Standard Incomplete Markets
    LCIM: Life Cycle Incomplete Markets
---

+++ {"part": "abstract"}

Heterogeneous Agent Models (HAM) are a powerful tool for understanding the effects of monetary and fiscal policy on the economy. However, current state-of-the-art frameworks such as Heterogeneous Agent New Keynesian (HANK) models have limitations that hinder their ability to accurately replicate real-world economic phenomena. Specifically, HANK models struggle to account for the observed hoarding of wealth at the very top of the distribution and lack important life cycle properties such as time-varying mortality and income risk. On the one hand, the inability to pin down wealth at the tail of the distribution has been a problem for HANK models precisely because it has implications for the transmission of monetary and fiscal policy. On the other hand, agents in HANK are generally conceived as perpetual youth with infinite horizons and without age-specific profiles of mortality and income risk. This is problematic as it ignores the effects of these policies on potentially more affected communities, such as young families with children or the low-wealth elderly. In this paper, I investigate the effects of both life cycle considerations as well as wealth in the utility on the structural estimation of HAMs. Structural estimation is the first step in evaluating the effect of monetary and fiscal policies in a HANK framework, and my hope is that this paper will lead to better models of the economy that can be used to inform policy.

+++

<!-- Heterogeneous Agent Models (HAM) have become a popular tool for understanding the effects of monetary and fiscal policy on the economy. However, current state-of-the-art frameworks such as Heterogeneous Agent New Keynesian (HANK) models have limitations that hinder their ability to accurately replicate real-world economic phenomena. Specifically, HANK models struggle to account for the observed hoarding of wealth at the very top of the distribution and lack important life cycle properties such as time-varying mortality and income risk. These limitations are problematic because they affect the transmission of monetary and fiscal policy and ignore the effects of these policies on potentially more affected communities, such as young families with children or the low-wealth elderly.

To address these limitations, this paper investigates the effects of both life cycle considerations and wealth in the utility on the structural estimation of HAMs. By incorporating these factors into the model, we aim to provide a more accurate representation of the economy and its response to monetary and fiscal policy. Our research methodology involves using a combination of theoretical analysis and empirical data to estimate the parameters of the model.

Our findings suggest that incorporating life cycle considerations and wealth in the utility can significantly improve the accuracy of HAMs. This has important implications for policymakers who rely on these models to inform their decisions. By providing a more accurate representation of the economy, our research can help policymakers make better-informed decisions that benefit all members of society. Overall, we hope that our research will contribute to the development of better models of the economy that can be used to inform policy. -->

+++ {"part": "acknowledgements"}

I would like to thank my advisor, Chris Carroll, for his guidance and support throughout this project. His expertise and mentorship have been invaluable in shaping my work. Additionally, I would like to extend my appreciation to the members of the [Econ-ARK] team for fostering a dynamic and collaborative community that has greatly enriched my experience. Their contributions and feedback have been instrumental in helping me achieve my goals. All remaining errors are my own. The figures in this paper were generated using the [Econ-ARK/HARK] toolkit.

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
    \\ & \text{s.t.} & 
    \\ \aNrm_{t} & = {m}_{t}-\cNrm_{t} 
    \\ {m}_{t+1} & = \aNrm_{t}\underbrace{\left(\frac{\Rfree}{\PermShk_{t+1}\PermGroFac_{t+1}}\right)}_{\equiv \RNrm_{t+1}}
    + ~\TranShkEmp_{t+1}
\end{align}

where $\cNrm$, $\aNrm$, and $\mNrm$ are consumption, assets, and market resources normalized by permanent income, respectively, $\vFunc$ and $\uFunc$ are now the normalized value and utility functions,  and

\begin{align}
  \PermShk_{t+1} & :  \text{mean-one shock to permanent income}
    \\ \PermGroFac_{t+1} & :  \text{permanent income growth factor}
    \\ \TranShkEmp_{t+1} & :  \text{transitory shock to permanent income}
    \\ \RNrm_{t+1} & :  \text{permanent income growth normalized return factor}
\end{align}

with all other variables are defined as above. The transitory and permanent shocks to income are defined as:

\begin{align}
\TranShkEmp_{s}  = &
    \begin{cases}
        0\phantom{/\pZero} & \text{with probability $\pZero>0$}
        \\ \xi_{s}/\pZero & \text{with probability $(1-\pZero)$, where
            $\log \xi_{s}\thicksim \mathcal{N}(-\sigma_{[\xi, t]}^{2}/2,\sigma_{[\xi, t]}^{2})$}
    \end{cases}
    \\ \phantom{/\pZero} \\ & \text{and }  \log \PermShk_{s}   \thicksim \mathcal{N}(-\sigma_{[\PermShk, t]}^{2}/2,\sigma_{[\PermShk, t]}^{2}).
  \end{align}


## Wealth in the Utility Function

A simple extension to the Life Cycle Incomplete Markets (LCIM) model is to include wealth in the utility function. [](doi:10.3386/w6549) argues that models in which the only driver of wealth accumulation is consumption smoothing are not consistent with the saving behavior of the wealthiest households. Instead, they propose a model in which households derive utility from their level of wealth itself or they derive a flow of services from political power and social status, calling it the `Capitalist Spirit' model. In turn, we can add this feature to the LCIM model by adding a utility function with consumption and wealth as follows:

\begin{align}
    {\vFunc}_{t}({m}_{t}) & = \max_{\cNrm_{t}} ~ \uFunc(\cNrm_{t}, \aNrm_{t})+\beth\Alive_{t+1}\hat{\DiscFac}_{t+1}
    \Ex_{t}[(\PermShk_{t+1}\PermGroFac_{t+1})^{1-\CRRA}{\vFunc}_{t+1}({m}_{t+1})]
    \\ & \text{s.t.} & 
    \\ \aNrm_{t} & = {m}_{t}-\cNrm_{t} 
    \\ {m}_{t+1} & = \aNrm_{t}\RNrm_{t+1}+ ~\TranShkEmp_{t+1}
\end{align}


**Separable Utility** [](doi:10.3386/w6549) presents extensive empirical and informal evidence for a LCIM model with wealth in the utility function. Specifically, the paper uses a utility that is separable in consumption and wealth:

\begin{equation}
    \uFunc(\cNrm_{t}, \aNrm_{t}) = \frac{\cNrm_{t}^{1-\CRRA}}{1-\CRRA}
    + \kapShare_{t} \frac{(\aFunc_{t} - \underline\aNrm)^{1-\wealthShare}}{1-\wealthShare}
\end{equation}

where $\kapShare$ is the relative weight of the utility of wealth and $\wealthShare$ is the relative risk aversion of wealth. 


**Non-separable Utility** A different model that we will explore is one in which the utility function is non-separable in consumption and wealth; i.e. consumption and wealth are complimentary goods. In the case of the LCIM model, this dynamic complementarity drives the accumulation of wealth not only for the sake of wealth itself, but also because it increases the marginal utility of consumption. 

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



# Quantitative Strategy

This section describes the quantitative strategy used for calibrating and estimating the Life Cycle Incomplete Markets model with and without Wealth in the Utility Function, following the works of [](doi:10.1198/073500103288619007), [](doi:10.1111/1467-937X.00092), [](doi:10.1111/1468-0262.00269), and [](doi:10.1016/j.jmoneco.2010.04.003), among others. The main objective is to find a set of parameters that can best match the empirical moments of some real-life data using simulation. 

## Calibration

The calibration of the Life Cycle Incomplete Markets model necessitates a richness not present in the SIM model precisely because we are interested in the heterogeneity of agents across different stages of the life cycle, such as the early working period, parenthood, saving for retirement, and retirement. To calibrate this model, we need to identify important patterns in preferences, mortality, and income risk across the life cycle. The first and perhaps most important departure from SIM is that life is finite and agents don't life forever; moreover, the terminal age is not certain as the probability of staying alive decreases with age. In this model, households start their life cycle at age $t = 25$ and live with certainty until retirement at age $t = 65$. After retirement, the probability of staying alive decreases with age, and the terminal age is set to $t = 91$. During their early adulthood, their utility of consumption might need to be adjusted by the arrival and subsequent departure of children. This is handled by a `household-size-adjusted' discount factor that is greater than 1.0 in the presence of children. This is the rationale for parameters $\Alive_{t}$ and $\hat{\DiscFac}_{t}$ in the model, whose values we take from [](doi:10.1198/073500103288619007) directly. 

The unemployment probability is taken from [](doi:10.2307/2534582) to be $\pZero = 0.5$ which represents a long run equilibrium of 5\% unemployment in the United States. The remaining life cycle attributes for the distribution of shocks to income ($\PermGroFac_{t}, \ \sigma_{[\PermShk, t]}, \ \sigma_{[\xi, t]}$) are taken from [](doi:10.1016/j.jmoneco.2010.04.003). In their paper, they analyze the variability of labor earnings growth rates between the 80's and 90's and find evidence for the "Great Moderation", a decline in variability of earnings across all age groups. 

After careful calibration based on the Life Cycle Incomplete Markets literature, we can structurally estimate the remaining parameters $\beth$ and $\CRRA$ to match specific empirical moments of the wealth distribution. 

## Estimation

```{figure}  ../Figures/IndShockSMMcontour.*
:name: fig:IndShockSMMcontour
:alt: IndShockSMMcontour
:align: center

Contour plot of the objective function for the structural estimation of the Life Cycle Incomplete Markets model. The red dot represents the estimated parameters.
```

```{figure} ../Figures/AllSMMcontour.*
:name: fig:AllSMMcontour
:alt: AllSMMcontour
:align: center

Contour plot of the objective function for the structural estimation of the Life Cycle Incomplete Markets model. The red dot represents the estimated parameters.
```


## Sensitivity Analysis

For our sensitivity analysis, we use the methods introduced by [](doi:10.1093/qje/qjx023).

```{figure}  ../Figures/IndShockSensitivity.*
:name: fig:IndShockSensitivity
:alt: IndShockSensitivity
:align: center

Sensitivity analysis of the structural estimation of the Life Cycle Incomplete Markets model. The red dot represents the estimated parameters.
```

```{figure} ../Figures/AllSensitivity.*
:name: fig:AllSensitivity
:alt: AllSensitivity
:align: center

Sensitivity analysis of the structural estimation of the Life Cycle Incomplete Markets model. The red dot represents the estimated parameters.
```


# Conclusion

# References

[](doi:10.1111/1467-937X.00092)
[](doi:10.3386/w7826)
[](doi:10.3386/w6549)
[](doi:10.1162/rest_a_00893)
[](doi:10.3982/ECTA17434)
[](doi:10.3386/w26941)
[](doi:10.1257/aer.20160042)
[](doi:10.3386/w26647)
[](doi:10.1198/073500103288619007)
[](doi:10.1093/qje/qjx023)
[](doi:10.1080/07350015.1999.10524794)  <!-- Humps and Bumps in Lifetime Consumption -->


[Econ-ARK]: https://econ-ark.org/
[Econ-ARK/HARK]: https://github.com/econ-ark/HARK