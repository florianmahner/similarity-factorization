# About project
This is a computational cognitive neuroscience study which compares IT representations between macaque and human data. We perform within and cross-species analyses, comparing multi-unit responses from monkey IT and fMRI responses from human IT. Neural responses to an intersecting set of 8640 stimuli (720 categories with 12 exemplars) were analyzed using CCA, targeting shared patterns across and within primates. . We implement three CCA models: a universal model including all five subjects (two monkeys, three humans) to home in on shared representations, and species-specific models that exclude information from the other species.

# Coding overheads
- config/config.toml defines a series of overhead settings
- config/paths.py sets paths
- functions/plotting.py defines a consistent visual style for publication-ready figures
- functions/ contains functions for general algorithms for computation
- data/ contains neural data, images, DNN features, THINGS meta-data and a plethora of other resources
- figures/ is where we save figures
- model/ contains scripts that implement central analyses, often calling functions from /functions/
- viz/ contains visualization scripts, often tied in logic to /model/

# Getting started
While I outline the general structure and analysis logic above, in the beginning, i.e. if we just start an interaction, get a bird-eye view of my code and analysis logic. You are actively encouraged to run bits of code to inspect the dimensionality, shape, keys, and contents of data files as this will mitigate issues down the road. You can do this either directly in the terminal, or through temporary test scripts that are cleaned up after execution. There are many data types and formats, so inspection and testing is encouraged. This also avoids bloated code with excessive conditionals.

# Data format
8640 INTERSECTING stimuli, 720 categories. 
Monkey-only CCA model: 8640x77 (derived and averaged across 2 monkey views)
Human-only CCA model: 8640x30 (3 human views)
Cross-species CCA model 8640x79 (5 views in total)
Feature spaces were:
Human 01: Shape: (9840, 5064) Human 02: Shape: (9840, 4076) Human 03: Shape: (9840, 3327), monkey
Monkey N: Shape: (22248, 250) Monkey F: Shape: (22248, 313)

# Coding style
Write concise code with moderately short variable and function names that are somewhat descriptive. Example of bad variable name: nv, number_of_views. good: n_views. Example of bad function name: compute_view_component_correlations, good: viewcomp_corr. Avoid excessive white space and error checks; do not take up too many lines. However, do not compromise on quality.  Under no circumstances must you use emojis. Try to adopt the coding style already used in the codebase.
we must emphasize simplicity and a streamlined approach. That is, consider us to effectively be in  a setting up phase in which we are building a generalizable and workable infrastructure to do experiments. As such, the code itself must be non-bloated and concise, and the logic must be intuitive and simple.
I prefer having relatively modular scripts, with key choices specific to any one script defined at the top. I prefer this over more CLI-based approaches and overly many interlocking functions.

# Editing logic
Please make surgical and precise changes that minimally affect the codebase while still getting the job done. Spend extra time if you need to to find a solution that implements concise code. It is one thing to write code concisely, but it is another thing to know a compact solution.

# Code execution
Code can be ran with conda activate thingsprimate.

# Generally important
- When just getting started on a query or line of work, first get a high-level overview of this project by quickly moving through the codebase.
- It is critical that you adopt my current coding style. Never use emojis; stay matter-of-fact.

## Parallel Processing:
- Use joblib with config.analysis.n_jobs for parallelization
- Implement per-feature/per-layer parallel loops in ridge_component.py and ridge_species.py
- Control parallel processing via config.analysis.n_jobs with minimal code changes
- Compact progress update prints are always appreciated if they are easy to add in a line or two

## Environment & Paths:
- Run commands with 'conda activate thingsprimate'
- Use parallel processing with config.n_jobs CPU cores
- Specify key paths in central config.toml file and use predefined paths
As you prepare to make coding changes, make sure to get a proper understanding of data structures; including their dimensionality, row and column structure. This will avoid errors down the road.