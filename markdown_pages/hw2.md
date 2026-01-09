---
title: Homework 2
layout: default
permalink: /hw2/
toc: true
---

<style>
  h1 {
    font-size: x-large;
  }

  h1 a {
    font-size: medium;
  }

  h1 img {
    float: left;
    padding-right: 1em;
  }

  h2 {
    font-size: x-large;
    text-align: left;
    font-variant: small-caps;
  }

  h2 b {
    font-size: large;
    font-variant: normal;
    color: red;
  }

  h2 i {
    font-size: large;
    font-variant: normal;
    font-style: italic;
    font-weight: normal;
  }

  h3 {
    font-size: large;
    font-variant: small-caps;
    margin: 1em 0 0 0;
  }

  /* Ensure h5 is at least as large as paragraph text */
  h5 {
    font-size: 1em;
  }

  p {
    margin: 0 1em 0.5em 1em;
  }

  ul,
  ol {
    margin: 0.5em 0 0.5em 1em;
  }

  li {
    margin: 0;
  }

  /* Rubric layout styles (scoped to this page) */
  .rubric {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .rubric-col {
    width: 48%;
    min-width: 320px;
  }

  .rubric-indent {
    margin-left: 0.5rem;
  }

  .rubric-note {
    font-size: 0.95em;
  }

  .rubric h5 {
    margin: 0.25em 0;
  }
</style>

<header>
  <h1>
    Homework 2<br>
    <a href="../../">COMS4732: Computer Vision 2</a>
  </h1>
</header>

<h2 style="text-align: center;">
  <div style="display: flex; justify-content: center; gap: 1em; align-items: center; flex-wrap: wrap;">
    <img src="/hws/hw2/image002.gif" alt="Feature Matching Example">
    <img src="/hws/hw2/image003.gif" alt="Feature Matching Example 2">
  </div><br>
  AUTOMATIC FEATURE MATCHING ACROSS IMAGES<br>
  <b style="color:#9E0000">Due Date: TBD</b>
</h2>

## Background

This assignment will involve creating a system for automatically detecting corresponding features in 2 images, as well as learning how to read and implement a research paper.

The project will consist of the following steps:

- **B.1**: Detecting corner features in an image (20 pts)
- **B.2**: Extracting a Feature Descriptor for each feature point (20 pts)
- **B.3**: Matching these feature descriptors between two images (20 pts)
- **B.4**: Bells & Whistles

For the following sections, we will follow the paper ["Multi-Image Matching using Multi-Scale Oriented Patches"](https://cal-cs180.github.io/fa25/hw/proj3/Papers/MOPS.pdf) by Brown et al. but with several simplifications. Read the paper first and make sure you understand it, then implement the algorithm.

## B.1: Harris Corner Detection

- Start with Harris Interest Point Detector (Section 2). We won't worry about multi-scale – just do a single scale. Also, don't worry about sub-pixel accuracy. Re-implementing Harris is a thankless task – so you can use my sample code: [harris.py](harris.py). Include on your webpage a figure of the Harris corners overlaid on the image.

- Implement Adaptive Non-Maximal Suppression (ANMS, Section 3). Include on your webpage a figure of the chosen corners overlaid on the image. This section has multiple moving parts; you may need to read it a few times. You may want to skip this step and come back to it; just choose a random set of corners instead in the meantime.

**Deliverables:** Show detected corners overlaid on image, with and without ANMS.

## B.2: Feature Descriptor Extraction

Implement Feature Descriptor extraction (Section 4 of the paper). Don't worry about rotation-invariance – just extract axis-aligned 8x8 patches. Note that it's extremely important to sample these patches from the larger 40x40 window to have a nice big blurred descriptor. Don't forget to bias/gain-normalize the descriptors. Ignore the wavelet transform section.

**Deliverables:** Extract normalized 8x8 feature descriptors. Show several extracted features.

## B.3: Feature Matching

Implement Feature Matching (Section 5 of the paper). That is, you will need to find pairs of features that look similar and are thus likely to be good matches. For thresholding, use the simpler approach due to Lowe of thresholding on the ratio between the first and the second nearest neighbors. Consult Figure 6b in the paper for picking the threshold. Ignore Section 6 of the paper.

**Deliverables:** Show matched features between image pairs. 

## B.5: Bells & Whistles <span style="color:#9E0000">TBD IF ASSIGNING OR NOT</span>

**Choose one:**

- **Multiscale processing:** Add multiscale processing for corner detection and feature description.

- **Rotation invariance:** Add rotation invariance to the feature descriptors.

## Deliverables

- You must turn in your code and project webpage as described [here](https://cal-cs180.github.io/fa25/hw/submitting.html). You should include (at minimum) the deliverables specified for each exercise.

- Submit your webpage public URL to the class gallery filling out [this](https://docs.google.com/forms/d/e/1FAIpQLSeCnEZKsefuGIh0V3kAIfFfRxnXTcUAUQ9Ld3XyC-LvHhGqaQ/viewform?usp=header) form.

- Your submission for part B should include the contents of part A and B.