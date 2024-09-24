---
title: "Integro-Difference Equation Models in JAX"
author: "Evan Tate Paterson Hughes"
header-includes:
  - \renewcommand*{\vec}[1]{\boldsymbol{\mathbf{#1}}}
output:
  pdf_document:
    keep_tex: true
---

# The Model

The Integro-Difference Equation model, sometimes abbreviated to IDE or, to avoid confusion, IDEM, is a hierarchical dynamic spatio-temporal model, which can be written

\begin{align}
\begin{split}
Z(\vec s;t) &= Y(\vec s;t) + \epsilon_t(\vec s)\\
Y(\vec s;t+1) &= \int_{\mathcal D_s} \kappa(s,r;t) Y(r;t) d\vec r + \eta_t(\vec s).\label{eq:IDeq}
\end{split}
\end{align}

Here, $Y$, the process, defined over the space $\mathcal D_s$ and discrete time $t=1, \dots, T$ evolves dynamically with respect to discrete time, according to a weighted integral of it's previous state. There is also with a non-dynamical component $\eta_t(\vec s)$ and measurement error/noise process for the data, $\epsilon_t(\vec)$, all of which are assumed mutually independent in time.

For more rigorous treatment of this model, there are many sources for the subject \citep[most prominently, both][, and @wikle2019spatio for a treatment in R]{cressie2015statistics}.

This project could also be understood as a re-implementation of Andrew Zammit Mangion's `R` package `IDE` \citep{zammit2022IDE}, though the hope is that the functionality and speed will surpass that implementation.
