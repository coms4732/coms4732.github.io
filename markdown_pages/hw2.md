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

We will follow the paper ["Multi-Image Matching using Multi-Scale Oriented Patches"](https://cal-cs180.github.io/fa25/hw/proj3/Papers/MOPS.pdf) by Brown et al. but with several simplifications. Read the paper first and make sure you understand it, then implement the algorithm.

## Step 0: Taking photos (0 points, required)

Take your 2 photos as you would a panoramic, keeping in mind to keep your camera level, only rotating your camera but not translating it.

## Step 1: Harris Corner Detection (5 points)

- Start with Harris Interest Point Detector (Section 2). We won't worry about multi-scale – just do a single scale. Also, don't worry about sub-pixel accuracy. Re-implementing Harris is a thankless task – so you can use our sample code: [harris.py](harris.py).

![Step 1 Deliverable](/hws/hw2/step1_deliverable.png)

**Deliverables:** Show your 2 images, as-is, side-by-side. Also, show detected corners overlaid on your set of images side-by-side.
## Step 2: Adaptive Non-Maximal Suppression (ANMS) (15 points)

- Implement Adaptive Non-Maximal Suppression (ANMS, Section 3). Include in your submission a figure of the chosen corners overlaid on the image. This section has multiple moving parts; you may need to read it a few times. You may want to skip this step and come back to it; just choose a random set of corners instead in the meantime.

![Step 2 Deliverable](/hws/hw2/step2_deliverable.png)

**Deliverables:** Show chosen corners overlaid on your set of images side-by-side after applying ANMS.

## Step 3: Feature Descriptor Extraction (5 points)

Implement Feature Descriptor extraction (Section 4 of the paper). Don't worry about rotation-invariance – just extract axis-aligned 8x8 patches. Note that it's extremely important to sample these patches from the larger 40x40 window to have a nice big blurred descriptor. Don't forget to bias/gain-normalize the descriptors. Ignore the wavelet transform section.

**Deliverables:** deliverable is part of step 4

## Step 4: Feature Matching (15 points)

Implement Feature Matching (Section 5 of the paper). That is, you will need to find pairs of features that look similar and are thus likely to be good matches. There are 2 approaches for this:
1. Nearest neighbor matching: Find the nearest neighbor for each feature in the other image and check if the distance is less than some threshold.
2. Lowe's ratio test / NN distance ratio (NNDR): 
  - Threshold on the ratio between the first and the second nearest neighbors. Consult Figure 6b in the paper for picking the threshold. Ignore Section 6 of the paper.
  - The idea here is that a good descriptor from img2 should be significantly better than the other img2 descriptor candidates while still being close to the img1 descriptor.

You will implement the latter: for your set of images, display the NNDR histogram and highlight the threshold you used.

<div style="display: flex; justify-content: center; gap: 1em; align-items: center; flex-wrap: wrap;">
  <img src="/hws/hw2/step4.1_deliverable.png" alt="Step 4.1 Deliverable" style="max-width: 45%; height: auto;">
  <img src="/hws/hw2/step4.2_deliverable.png" alt="Step 4.2 Deliverable" style="max-width: 45%; height: auto;">
</div>

![Step 4.3 Deliverable](/hws/hw2/step4.3_deliverable.png)

**Deliverables:** 
1. Display the NNDR histogram and highlight the threshold you used. Also specify which similarity metric you used (e.g. SSD, NCC, etc.).
2. Visualize the 5 best feature matches between the 2 images (no worries if you don't have 10 matches, just show as many as you can).
   1. the first column should be the feature descriptor for img1's feature.
   2. the second column should be the 1NN feature descriptor from img2.
   3. the third column should be the 2NN feature descriptor from img2.
3. Color-code the matched features across both images and display them side-by-side. Also, put a number next to each feature to indicate the match index.

## Step 5: RANSAC to estimate the homography (40 points)

- Use 4-point RANSAC as described in class to compute robust homography estimates. Using your best homography estimate, visualize the inliers among the homography applied on all points. 

![Step 5 Deliverable](/hws/hw2/step5_deliverable.png)

**Deliverables:** Show the inliers associated with the best homography found with RANSAC.

## Bells & Whistles (Optional)

- **Image Panoramic:** Create a panoramic image from the 2 images by stitching them together using the homography and blending the images together.

- **Rotation invariance:** Add rotation invariance to the MOPS feature descriptors.

- **Support for 3+ images:** Implement support for 3+ images by using the homography to stitch together more than 2 images.

## Deliverables

You must submit your code and visualizations outlined above for **2 different scenes / pairs of images**. You *cannot* use the example provided by the staff. An example for one scene is shown below:

<div style="display: flex; flex-direction: column; gap: 1em; align-items: center;">
  <img src="/hws/hw2/step4.2_deliverable.png" alt="Step 4.2 Deliverable - NNDR Distribution" style="max-width: 80%; height: auto;">
  <div style="display: flex; justify-content: center; gap: 1em; align-items: stretch; flex-wrap: wrap; width: 100%;">
    <div style="flex: 1; max-width: 48%; display: flex;">
      <img src="/hws/hw2/step4.1_deliverable.png" alt="Step 4.1 Deliverable - Top Feature Matches" style="width: 100%; height: 100%; object-fit: contain;">
    </div>
    <div style="flex: 1; max-width: 48%; display: flex;">
      <img src="/hws/hw2/entire_deliverable.png" alt="Entire Pipeline Deliverable" style="width: 100%; height: 100%; object-fit: contain;">
    </div>
  </div>
</div>

## Hints
- You can use LLMs to implement any visualization code you wish.
- If you would like to visually debug using the staff example, the images can be found here: [img1.jpg](/hws/hw2/north1.jpg) and [img2.jpg](/hws/hw2/north2.jpg) 
- Some hyperparameters (such as the RANSAC NNDR threshold) may need to be tuned for each image pair. For references, the staff solution for the example provided in this webpage used the following:

| Hyperparameter | Value | Description |
|----------------|---------------|-------------|
| `harris_corner_edge_discard` | 20 | Number of pixels to discard from image edges when detecting Harris corners |
| `anms_c_robust` | 0.8 | Robustness parameter for Adaptive Non-Maximal Suppression (ANMS) |
| `anms_num_points` | 250 | Number of interest points to retain after ANMS |
| `feature_matching_metric` | "NCC" | Metric for comparing feature descriptors (e.g., NCC, SSD) |
| `feature_matching_ratio_threshold` | 0.85 | NNDR (Nearest Neighbor Distance Ratio) threshold for feature matching |
| `ransac_s` | 4 | Number of correspondences to sample per RANSAC iteration |
| `ransac_epsilon` | 1.0 | Distance threshold (in pixels) for RANSAC inlier classification |
| `ransac_num_iters` | 15000 | Number of iterations to run RANSAC |

## Acknowledgements

This assignment is based on [Alyosha Efros's version at Berkeley](https://cal-cs180.github.io/fa25/hw/proj3/partB.html).