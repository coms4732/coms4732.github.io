---
title: "HW3: Starter Code"
layout: default
permalink: /hw3_startercode/
nav_exclude: true
published: true
---

<header>
  <h1>
    HW3: Starter Code<br>
    <a href="/hw3/">← Back to Homework 3</a>
  </h1>
</header>

## Repository

Clone the starter code from the GitHub repository:

**[https://github.com/coms4732/hw3_starter/](https://github.com/coms4732/hw3_starter/)**

## Setup

Optional: create a conda env

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Your Task:

- Adjust the pipeline hyperparameters (e.g., SIFT parameters, NNDR threshold, RANSAC iterations, Sampson distance threshold, etc.) via the `Config` object at the top of `main.py`. 
  - Steps 2 and 3 simply involve setting the parameters and running `main.py`.
- Implement `intrinsics.py` for step 1
- Implement `ransac.py` for step 4

There are visualizations (required ones, as well as ones for debugging) outputted for each step of the pipeline (assuming they're implemented properly) each time you run `python main.py`. They are saved in a directory with `YYYY-MM-DD-HH-MM` as the directory name.