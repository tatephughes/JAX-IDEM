---
author: Evan Tate Paterson Hughes
bibliography:
- bibliography.bib
classoption: letterpaper
documentclass: article
header-includes:
- "`\\usepackage{amsmath,amsfonts,amsthm,amssymb,bm,bbm,tikz,tkz-graph, graphicx, subcaption, mathtools, algpseudocode}`{=latex}"
- "`\\usepackage[cache=false]{minted}`{=latex}"
- "`\\usetikzlibrary{arrows}`{=latex}"
- "`\\usetikzlibrary{bayesnet}`{=latex}"
- "`\\usetikzlibrary{matrix}`{=latex}"
- "`\\usepackage[margin=1in]{geometry}`{=latex}"
- "`\\usepackage[english]{babel}`{=latex}"
- "`\\newtheorem{theorem}{Theorem}[section]`{=latex}"
- "`\\newtheorem{corollary}[theorem]{Corollary}`{=latex}"
- "`\\newtheorem{lemma}[theorem]{Lemma}`{=latex}"
- "`\\newtheorem{definition}[theorem]{Definition}`{=latex}"
- "`\\newtheorem*{remark}{Remark}`{=latex}"
- "`\\DeclareMathOperator{\\E}{\\mathbb E}`{=latex}"
- "`\\DeclareMathOperator{\\prob}{\\mathbb P}`{=latex}"
- "`\\DeclareMathOperator{\\var}{\\mathbb V\\mathrm{ar}}`{=latex}"
- "`\\DeclareMathOperator{\\cov}{\\mathbb C\\mathrm{ov}}`{=latex}"
- "`\\DeclareMathOperator{\\cor}{\\mathbb C\\mathrm{or}}`{=latex}"
- "`\\DeclareMathOperator{\\normal}{\\mathcal N}`{=latex}"
- "`\\DeclareMathOperator{\\invgam}{\\mathcal{IG}}`{=latex}"
- "`\\newcommand*{\\mat}[1]{\\bm{#1}}`{=latex}"
- "`\\newcommand{\\norm}[1]{\\left\\Vert #1 \\right\\Vert}`{=latex}"
- "`\\renewcommand*{\\vec}[1]{\\boldsymbol{\\mathbf{#1}}}`{=latex}"
title: Spatio-Temporal Integro-Difference Equation models in Python
---

`\usepackage{amsmath,amsfonts,amsthm,amssymb,bm,bbm,tikz,tkz-graph, graphicx, subcaption, mathtools, algpseudocode}`{=latex}

`\usepackage[cache=false]{minted}`{=latex}

`\usetikzlibrary{arrows}`{=latex}

`\usetikzlibrary{bayesnet}`{=latex}

`\usetikzlibrary{matrix}`{=latex}

`\usepackage[margin=1in]{geometry}`{=latex}

`\usepackage[english]{babel}`{=latex}

`\newtheorem{theorem}{Theorem}[section]`{=latex}

`\newtheorem{corollary}[theorem]{Corollary}`{=latex}

`\newtheorem{lemma}[theorem]{Lemma}`{=latex}

`\newtheorem{definition}[theorem]{Definition}`{=latex}

`\newtheorem*{remark}{Remark}`{=latex}

`\DeclareMathOperator{\E}{\mathbb E}`{=latex}

`\DeclareMathOperator{\prob}{\mathbb P}`{=latex}

`\DeclareMathOperator{\var}{\mathbb V\mathrm{ar}}`{=latex}

`\DeclareMathOperator{\cov}{\mathbb C\mathrm{ov}}`{=latex}

`\DeclareMathOperator{\cor}{\mathbb C\mathrm{or}}`{=latex}

`\DeclareMathOperator{\normal}{\mathcal N}`{=latex}

`\DeclareMathOperator{\invgam}{\mathcal{IG}}`{=latex}

`\newcommand*{\mat}[1]{\bm{#1}}`{=latex}

`\newcommand{\norm}[1]{\left\Vert #1 \right\Vert}`{=latex}

`\renewcommand*{\vec}[1]{\boldsymbol{\mathbf{#1}}}`{=latex}

::: {.BOILERPLATE .drawer}
```{=org}
#+cite_export: natbib authoryear authoryear
```
```{=org}
#+EXPORT_EXCLUDE_TAGS: noexport
```
:::

# The Model

The Integro-Difference Equation model, sometimes abbreviated to IDE or,
to avoid confusion, IDEM, is a hierarchical dynamic spatio-temporal
model, which can be written

```{=latex}
\begin{align}
\begin{split}
Z(\vec s;t) &= Y(\vec s;t) + \epsilon_t(\vec s)\\
Y(\vec s;t+1) &= \int_{\mathcal D_s} \kappa(s,r;t) Y(r;t) d\vec r + \eta_t(\vec s).\label{eq:IDeq}
\end{split}
\end{align}
```
Here, $Y$, the process, defined over the space $\mathcal D_s$ and
discrete time $t=1, \dots, T$ evolves dynamically with respect to
discrete time, according to a weighted integral of it\'s previous state.
There is also with a non-dynamical component $\eta_t(\vec s)$ and
measurement error/noise process for the data, $\epsilon_t(\vec)$, all of
which are assumed mutually independent in time.

For more rigorous treatment of this model, there are many sources for
the subject \[@most prominently, both \@cressie2015statistics, and
\@wikle2019spatio for a treatment in R;\].

This project could also be understood as a re-implementation of Andrew
Zammit Mangion\'s `R` package `IDE` [@zammit2022IDE], though the hope is
that the functionality and speed will surpass that implementation.
