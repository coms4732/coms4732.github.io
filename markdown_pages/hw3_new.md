---
title: Homework 3
layout: default
permalink: /hw3_new/
toc: true
nav_exclude: true
published: true
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
MathJax = {
tex: {
inlineMath: [['$', '$'], ['\\(', '\\)']],
displayMath: [['$$', '$$'], ['\\[', '\\]']],
processEscapes: true
}
};
</script>

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

  /* Collapsible section styles */
  details.section {
    margin: 0.5em 0;
  }

  details.section > summary {
    font-size: x-large;
    text-align: left;
    font-variant: small-caps;
    font-weight: bold;
    cursor: pointer;
    padding: 0.25em 0;
    list-style: revert;
  }

  details.section > summary::-webkit-details-marker {
    display: initial;
  }
</style>

<header>
  <h1>
    Homework 3<br>
    <a href="../../">COMS4732: Computer Vision 2</a>
  </h1>
</header>

<h2 style="text-align: center;">
  SIMPLE STRUCTURE FROM MOTION<br>
  <b style="color:#9E0000">Due Date: Thursday, March 5 11:59PM ET</b>
</h2>

<div style="text-align: center; margin: 1em 0;">
  <img src="/hws/hw3/assets/problem_statement_new.png" alt="Problem statement diagram" style="max-width: 100%;">
</div>

<span style="color: darkgreen;">**Starter code can be found [here](https://github.com/coms4732/hw3_starter_testing/).**</span>

<p style="font-size: 0.85em; margin: 0.5em 1em;">
  <a href="javascript:void(0)" id="toggle-all" onclick="(function(){var d=document.querySelectorAll('details.section'),open=d[0]&&!d[0].open;d.forEach(function(el){el.open=open});document.getElementById('toggle-all').textContent=open?'Collapse all':'Expand all'})()">Collapse all</a>
</p>

<details open class="section" markdown="1">
<summary id="overview">Overview</summary>
In this assignment, you'll build components of a simple [Structure from Motion (SfM)](https://en.wikipedia.org/wiki/Structure_from_motion) pipeline that estimates correspondences, camera position/motion, and 3D scene points from a pair of images. By the end of this assignment, you'll be able to generate a sparse point cloud and visualize the rotation and translation that relates the 2 cameras.

For full credit, you only have to achieve a working solution on the staff-provided images ([img1_1280x960.jpeg](/hws/hw3/assets/img1_1280x960.jpeg), [img2_1280x960.jpeg](/hws/hw3/assets/img2_1280x960.jpeg)). If you wish to improve the pipeline on your own and potentially achieve a denser point cloud, you may want to utilize the original-resolution images ([img1_5712x4284.jpeg](/hws/hw3/assets/img1_5712x4284.jpeg), [img2_5712x4284.jpeg](/hws/hw3/assets/img2_5712x4284.jpeg)).

<div style="text-align: center; margin: 1em 0;">
  <div style="display: flex; justify-content: center; gap: 1em; align-items: center; flex-wrap: wrap;">
    <img src="/hws/hw3/assets/img1_5712x4284.jpeg" alt="Image 1" style="max-width: 48%; height: auto;">
    <img src="/hws/hw3/assets/img2_5712x4284.jpeg" alt="Image 2" style="max-width: 48%; height: auto;">
  </div>
  <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.9em;">Ryan's living room out of equilibrium</p>
</div>

If you're interested in why this setup is the way it is and how to optimally capture your own scene, see instructions below on [capturing your own scene](#capturing-your-own-scene).

</details>

<details open class="section" markdown="1">
<summary>Step 1: Computing Camera Intrinsics (15 points)</summary>

This assignment involves stereo on *calibrated cameras*, meaning with known [camera intrinsics](https://en.wikipedia.org/wiki/Camera_resectioning). Note that the more general version of SfM uses uncalibrated cameras–meaning we don't know the intrinsics–though we leave this as an exercise to the reader.

Step 1 of this pipeline involves computing the intrinsic matrix $K$ for the camera associated with the images used:

$$K = \begin{bmatrix} f_\text{px} & 0 & c_x \\ 0 & f_\text{px} & c_y \\ 0 & 0 & 1 \end{bmatrix}, \qquad c_x = \frac{W_\text{px}}{2},\quad c_y = \frac{H_\text{px}}{2}$$

  where $f_\text{px}$ is the camera focal length in terms of number of pixels, $W_\text{px}$ is the number of horizontal pixels and $H_\text{px}$ is the number of vertical pixels, and $(c_x, c_y)$ is the principal point at the center of the image plane (i.e., center of image).

For the staff-provided images (captured with an iPhone 15 Pro on the main camera / 1x zoom):
- physical focal length: $f_\text{mm} = 6.765$mm
- physical camera sensor height: $h_\text{mm} = 7.318$mm
- physical camera sensor width:  $w_\text{mm} = 9.757$mm

From these physical parameters, we need to compute the focal length in pixels:

$f_\text{px} = f_\text{mm} \cdot \frac{W_\text{px}}{w_\text{mm}}$

Note that if we assume that the aspect ratio of the image width and height is the same as the aspect ratio of the sensor width and sensor height, then the pixels are square. Meaning, we could have equivalently computed $f_\text{px}$ from the image height and sensor height.

From inspecting the image directly, we also have access to the image height $h_\text{px}=960$ and image width $w_\text{px} = 1280$. Thus, we have everything needed to compute our intrinsics matrix $K$.

<span style="color: red;">**[Deliverable]**</span>
- Implement `intrinsics.py`
- Report what the $K$ matrix is for the staff-provided images.
  - Make sure to indicate which image resolution you're using.
  - For chance of partial credit, show your work.

</details>

<details open class="section" markdown="1">
<summary>Step 2: Detecting Features (10 points)</summary>

In HW2 we used a Harris Corner detector to compute candidate features that we'd want to use as correspondences between the two images. Because the motion between the two images in this assignment is more extreme (rotation + translation), we need a more robust feature descriptor. As such, we use [SIFT features](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform).

<span style="color: red;">**[Deliverable]**</span>
- Show a side-by-side of the SIFT features detected in your two images.
- Report the SIFT parameters used to compute these features and the resulting number of features per image.

<div style="text-align: center; margin: 1em 0;">
  <img src="/hws/hw3/assets/step2.png" alt="SIFT features detected in both images" style="max-width: 100%; height: auto;">
  <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.9em;">SIFT features detected in both images, side-by-side</p>
</div>

</details>

<details open class="section" markdown="1">
<summary>Step 3: Feature Matching (15 points)</summary>

In HW2, we created feature descriptors then subsequently did an $n^2$ search between the features in image 1 and image 2, using a metric like SSD or NCC to compare the features. We also applied the ratio test between the 1NN and 2NN features in image 2 for a feature in image 1. This is the nearest neighbor distance ratio (NNDR).

In this HW, the OpenCV SIFT API `cv2.SIFT_create` directly returns keypoints and feature descriptors––all that's left is to match the features between these images. We use an L2 loss to compare these features, as is typically done with SIFT. Since you've already implemented this in HW2, we provide code that uses a C++-optimized feature matching algorithm: `cv2.BFMatcher`. You still need to provide an NNDR threshold, like we did in HW2.

<span style="color: red;">**[Deliverables]**</span>
- Show the histogram of NNDR's for the scene. Indicate what threshold you used.
- Show the top 5 feature descriptors according to NNDR.
- Visualize side-by-side the correspondences. It's okay if it's slightly unclear what corresponds to what.

<div style="display: flex; justify-content: center; align-items: center; gap: 1em; margin: 1em 0;">
  <div style="flex: 1; text-align: center; min-width: 0;">
    <img src="/hws/hw3/assets/step3.2.png" alt="Top 5 feature matches by NNDR" style="max-width: 100%; height: auto;">
    <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.8em;">Top 5 feature matches by NNDR</p>
  </div>
  <div style="flex: 1; display: flex; flex-direction: column; gap: 1em; min-width: 0;">
    <div style="text-align: center;">
      <img src="/hws/hw3/assets/step3.1.png" alt="NNDR histogram" style="max-width: 100%; height: auto;">
      <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.8em;">NNDR histogram</p>
    </div>
    <div style="text-align: center;">
      <img src="/hws/hw3/assets/step3.3.png" alt="Matched correspondences between images" style="max-width: 100%; height: auto;">
      <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.8em;">Matched correspondences</p>
    </div>
  </div>
</div>

</details>

<details open class="section" markdown="1">
<summary>Step 4: RANSAC to estimate R and t (30 points)</summary>

Now we must estimate the rotation matrix $R$ and translation vector $t$ that relate camera 2 (from image 2) to camera 1 (from image 1). We treat camera 1's center as the origin. We do this via [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) and [epipolar geometry](https://en.wikipedia.org/wiki/Epipolar_geometry). You are not expected to understand the epipolar geometry well, but **you are expected to understand how RANSAC can be used as a general method to estimate the parameters of some mathematical model for which we have data that contains inliers and outliers.** As such, you will be implementing parts of RANSAC in this HW.

The mathematical model we hope to estimate is the rotation and translation that relates these two images. Using epipolar geometry, there is an [essential matrix](https://en.wikipedia.org/wiki/Essential_matrix) $E$ that relates the features between the images. We can use RANSAC to estimate $E$ and decompose it into $R$ and $t$ that relate the cameras. If you'd like to learn more about the underlying math, see the [appendix](#rt-ambiguity).

<div style="border: 1.5px solid #333; padding: 1em 1.5em; margin: 1em 0; background: #fafafa; font-family: serif; font-size: 0.97em; line-height: 1.6;">
<p style="margin: 0 0 0.5em 0; font-weight: bold; font-size: 1.05em; border-bottom: 1px solid #333; padding-bottom: 0.3em;">Algorithm 1: RANSAC for Essential Matrix Estimation</p>

<p style="margin: 0.3em 0;"><b>Input:</b> correspondence pairs $\{(p_1^i, p_2^i)\}_{i=1}^{N}$ from Step 3, camera intrinsics $K$, number of iterations $T$, Sampson distance threshold $\epsilon$</p>
<p style="margin: 0.3em 0 0.8em 0;"><b>Output:</b> rotation $R$, translation $t$, inlier mask</p>

<p style="margin: 0.2em 0 0.2em 0;">1: &ensp; best_inliers &larr; 0; &ensp; $E^*$ &larr; None</p>
<p style="margin: 0.2em 0 0.2em 0;">2: &ensp; <b>for</b> iter = 1 <b>to</b> $T$ <b>do</b></p>
<p style="margin: 0.2em 0 0.2em 2em;">3: &ensp; Randomly sample 8 correspondences &ensp;<span style="color: #888; font-style: italic; font-size: 0.9em;">(from the <a href="https://en.wikipedia.org/wiki/Eight-point_algorithm">8-point algorithm</a>)</span></p>
<p style="margin: 0.2em 0 0.2em 2em;">4: &ensp; Estimate essential matrix $E$ from the 8 samples</p>
<p style="margin: 0.2em 0 0.2em 2em;">5: &ensp; Decompose $E$ into 4 candidate $(R, t)$ pairs and run cheirality check on the sample &ensp;<span style="color: #888; font-style: italic; font-size: 0.9em;">(see <a href="#rt-ambiguity">appendix A.2</a>)</span></p>
<p style="margin: 0.2em 0 0.2em 2em;">6: &ensp; <b>if</b> no valid $(R, t)$ found, <b>continue</b> to next iteration</p>
<p style="margin: 0.2em 0 0.2em 2em;">7: &ensp; Compute inliers over <i>all</i> correspondences: $\;\mathcal{I} \leftarrow \{i : d_{\text{Sampson}}^2(E,\, p_1^i,\, p_2^i) < \epsilon \}$ &ensp;<span style="color: red; font-weight: bold; font-size: 0.9em;">← implement</span> &ensp;<span style="color: #888; font-style: italic; font-size: 0.9em;">(see <a href="#sampson-distance">appendix A.1</a>)</span></p>
<p style="margin: 0.2em 0 0.2em 2em;">8: &ensp; <b>if</b> $|\mathcal{I}|$ > best_inliers <b>then</b> &ensp;<span style="color: red; font-weight: bold; font-size: 0.9em;">← implement</span></p>
<p style="margin: 0.2em 0 0.2em 4em;">9: &ensp; best_inliers &larr; $|\mathcal{I}|$; &ensp; save $E^*$</p>
<p style="margin: 0.2em 0 0.2em 2em;">10: &ensp; <b>end if</b></p>
<p style="margin: 0.2em 0 0.2em 0;">11: &ensp; <b>end for</b></p>
<p style="margin: 0.2em 0 0.2em 0;">12: &ensp; Recompute inliers $\mathcal{I}^*$ from $E^*$ over all correspondences &ensp;<span style="color: red; font-weight: bold; font-size: 0.9em;">← implement</span></p>
<p style="margin: 0.2em 0 0.2em 0;">13: &ensp; Re-estimate $E$ from $\mathcal{I}^*$ &ensp;<span style="color: red; font-weight: bold; font-size: 0.9em;">← implement</span></p>
<p style="margin: 0.2em 0 0.2em 0;">14: &ensp; Recover $R, t$ from refined $E$ via decomposition + cheirality</p>
<p style="margin: 0.2em 0 0 0;">15: &ensp; <b>return</b> $R, t$, inlier mask</p>
</div>

<span style="color: red;">**[Deliverables]**</span>
- Complete the `ransac.py` implementation.
- Report RANSAC parameters used: number of iterations $T$, Sampson distance threshold $\epsilon$, and the final number of inliers found.
- Visualize the RANSAC convergence (best inlier count over iterations).
- Visualize the matched correspondences after RANSAC (inliers only).

<div style="text-align: center; margin: 1em 0;">
  <img src="/hws/hw3/assets/step4_ransac_convergence.png" alt="RANSAC convergence plot" style="max-width: 70%; height: auto;">
  <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.9em;">RANSAC convergence: best inlier count over iterations</p>
</div>

<div style="text-align: center; margin: 1em 0;">
  <img src="/hws/hw3/assets/step4_correspondences_after_ransac.png" alt="Matches after RANSAC" style="max-width: 80%; height: auto;">
  <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.9em;">Feature correspondences after RANSAC (inliers only)</p>
</div>

</details>

<details open class="section" markdown="1">
<summary>Step 5: Triangulating Inliers to generate Point Cloud (0 points, done for you)</summary>

Now that we have the camera poses ($R$ and $t$ from Step 4) and the inlier correspondences, we can recover the 3D positions of the matched points via [triangulation](https://en.wikipedia.org/wiki/Triangulation_(computer_vision)). This part is already done for you, but you will need to load the outputs into [Viser](https://viser.studio/) and show screenshots of the sparse point cloud. If the previous steps are implemented properly and `main.py` executes, a viser command will be outputted at the very end of the script.

If you'd like to learn more about the underlying math, see [the appendix](#triangulation-dlt).


<span style="color: darkgreen;">**[Hints]**</span>

When debugging, compare your point cloud with the input images and ask yourself:
- Are your inliers after RANSAC good enough? Do they actually correspond to the same feature across the images?
- Do the different surfaces at different depths seem right?
- Do planar surfaces in the image look planar in the point cloud?
- Do the cameras look right? Is camera 2 rotated and translated in a way consistent with the real scene?

Additionally:
- `step5_pipeline_grid.png` that's produced by the pipeline gives you a comprehensive overview of our SfM system, including the epipolar lines associated with our estimated $E$ matrix. Ask: do the epipolar lines look correct?
- You shouldn't need to touch the triangulation-related parameters in the config, but they are accessible in case you'd like to adjust them
- Triangulation will cause you to lose some points from your initial set of RANSAC inliers. This is expected. For reference: staff solution lost ~10 points after performing triangulation.

<span style="color: red;">**[Deliverables]**</span>

3 screenshots of the Viser server showing the point cloud from different angles:
- All 3 of them should show the cameras so we can observe the rotation + translation that relates them and have a reference across the 3 images.
- All 3 of them should **have a caption/description of what the identifiable 'landmarks' from the original images are visible**, since your point cloud will likely be sparse and hard to understand at first glance.
  - A point cloud of even, e.g., 20 points is valid, so long as you can justify why it's consistent with the geometry of the real-world scene.



<div style="display: flex; justify-content: center; gap: 1em; margin: 1em 0;">
  <div style="flex: 1; text-align: center; min-width: 200px; display: flex; flex-direction: column;">
    <div style="height: 280px; display: flex; align-items: center; justify-content: center;">
      <img src="/hws/hw3/assets/pc1.png" alt="Point cloud angle 1" style="max-width: 100%; max-height: 100%; object-fit: contain;">
    </div>
    <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.9em;">... I see bookshelf, poster, coffee table ...</p>
  </div>
  <div style="flex: 1; text-align: center; min-width: 200px; display: flex; flex-direction: column;">
    <div style="height: 280px; display: flex; align-items: center; justify-content: center;">
      <img src="/hws/hw3/assets/pc2.png" alt="Point cloud angle 2" style="max-width: 100%; max-height: 100%; object-fit: contain;">
    </div>
    <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.9em;"> ... the right camera looks correctly positioned to the top right of camera 1, with correct rotation applied ...</p>
  </div>
  <div style="flex: 1; text-align: center; min-width: 200px; display: flex; flex-direction: column;">
    <div style="height: 280px; display: flex; align-items: center; justify-content: center;">
      <img src="/hws/hw3/assets/pc3.png" alt="Point cloud angle 3" style="max-width: 100%; max-height: 100%; object-fit: contain;">
    </div>
    <p style="margin: 0.5em 0 0 0; font-style: italic; font-size: 0.9em;"> ... the poster looks planar from this angle, as expected ...</p>
  </div>
</div>

</details>

<details class="section" markdown="1">
<summary>Assignment Deliverables</summary>

As with HW1 and HW2, you must submit both your code and a webpage written as an `index.html` with pointers to images/assets. You must also submit a `README.md` file that outlines how to run your code. Lastly, submit a PDF version of your webpage. **Failure to submit any of these will result in lost points**.

For the staff-provided images, report the following:
  - Your random seed chosen
  - Camera intrinsics matrix $K$ used in the pipeline
  - SIFT number of points detected
  - Feature matching nearest neighbor distance ratio (NNDR) threshold used
  - RANSAC parameters:
      - number of RANSAC iterations
      - $\epsilon$ parameter used
  - Any other staff-provided parameters that you modified.
- Visualizations of:
  - SIFT features
  - NNDR histogram, with threshold overlaid
  - Top 5 features matched, ranked by NNDR
  - Correspondences between image 1 and 2 *before* RANSAC
  - Correspondences between image 1 and 2 *after* RANSAC
  - RANSAC convergence.
  - At least 3 screenshots of sparse point cloud with 'landmarks' described.

</details>

<details class="section" markdown="1">
<summary>Extra Credit</summary>

<details markdown="1">
<summary id="capturing-your-own-scene" style="font-size: large; font-variant: small-caps; font-weight: bold; cursor: pointer;">Capture your own scene (10 points, competition)</summary>

You are required to use the staff-provided images (provided in the [overview](#overview)) for full credit on this assignment.

For extra credit, reconstruct your own scene with our simple SfM pipeline on 2 images of your own. You get 10 points if you can get a decent reconstruction, and we will also be voting on the best custom scenes! **For this section, stick to the current pipeline. You are allowed to modify: scene contents, image resolution, and config hyperparameters.**

You'll need to provide the parameters for your own camera used. If this is difficult to access online, you can try accessing the image metadata and recover the parameters.

Advice:
- Lock the exposure and focus of your camera before taking photos. This will make it easier to find common features across the scene images.
- Choose a scene with multiple levels of depth (2+). Failing to do so will cause the model to only pick planar correspondences, which don't exhibit much disparity/parallax across camera angles. This makes it significantly harder to determine the geometry that relates these two photos (since we don't exhibit much change between them). Thus, we want a scene with varying levels of depth so that we see parallax at varying distances.
    - this also means that we don't only want to rotate our camera but translate as well. You will need to find a sweet spot between how much or how little translation to use. In a realistic pipeline you'd have ≫ 2 images taken and wouldn't have to worry about this as much.
- Choose a scene with lots of textured objects. The SIFT feature descriptor used is targeting textures and your pipeline's performance will markedly increase with the number of textured objects (ideally with textures at different distances in the scene).

**More details on competition to come**.

</details>


<details markdown="1">
<summary style="font-size: large; font-variant: small-caps; font-weight: bold; cursor: pointer;">Improve the staff solution (10 points, competition) </summary>

Improve on the staff-provided scene by adjusting the pipeline hyperparameters. You will likely need to use the higher resolution image as well. You'll be awarded 10 points if you can get a noticeable improvement, and even more goodies if ranked high in the competition!

**More details on competition to come**.

</details>

</details>


---

##### Appendix

<h3 id="epipolar-geometry-stereo">A. Epipolar geometry for stereo</h3>

<div style="text-align: center; margin: 1em 0;">
  <figure style="display: inline-block; margin: 1em 0;">
    <img src="/hws/hw3/assets/correspondences_to_pose.png" alt="Correspondences to pose diagram" style="max-width: 500px; width: 100%;">
    <figcaption style="margin-top: 0.5em; font-style: italic;">
      <strong>Figure 2:</strong> Now that we have correspondences between images, we aim to recover the motion of both cameras–i.e., the rotation $R$ and translation $t$ between our cameras
    </figcaption>
  </figure>
</div>

After getting our correspondences from [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) + feature matching, we now aim to estimate the pose of each camera. We do this via [epipolar geometry](https://en.wikipedia.org/wiki/Epipolar_geometry), which is the geometry that relates the correspondences and camera centers of the two images.

<div style="text-align: center; margin: 1em 0;">
  <figure style="display: inline-block; margin: 1em 0;">
    <img src="/hws/hw3/assets/epipolar1.png" alt="Epipolar geometry setup" style="max-width: 100%;">
    <figcaption style="margin-top: 0.5em; font-style: italic;">
      Epipolar geometry relates two camera views of the same 3D point
    </figcaption>
  </figure>
</div>

<h4 id="epipolar-constraint">The epipolar constraint</h4>

Suppose 2D points $x$ and $x'$ are projections of a 3D point $X$ onto the image planes of two cameras. The epipolar geometry relates these two points via the [essential matrix](https://en.wikipedia.org/wiki/Essential_matrix) $E$: ${x'}^\intercal E x= 0$

To see how we got here, we first observe that $x'$ is related to $x$ by a rotation and translation:

$$x' = Rx + t$$

where $t$ is the translation from camera 1's center of projection $O$ to camera 2's center of projection $O'$. $R$ is the rotation from camera 1's coordinate system to camera 2's coordinate system.

We observe that $x$, $x'$, and $t$ are all coplanar and would like to encode this constraint in a meaningful way. We define the normal vector to the plane as $n = t \times x'$ and substitute in the expression for $x'$:

$$
\begin{aligned}
n &= t \times x' \\
n &= t \times x' \\
n &= t \times (Rx + t) \\
n &= t \times (Rx + t) \quad (\text{since } t \times t = 0) \\
n &= t \times Rx
\end{aligned}
$$

Lastly, by definition of $n$:

$$
\begin{aligned}
x' \cdot n &= 0 \\
x' \cdot (t \times (Rx)) &= 0 \\
{x'}^\intercal [t_{\leftrightarrow}] R x &= 0 \\
{x'}^\intercal Ex &= 0
\end{aligned}
$$

Where $[t_{\leftrightarrow}]$ is a [skew-symmetric matrix](https://en.wikipedia.org/wiki/Skew-symmetric_matrix) representing the [cross product](https://en.wikipedia.org/wiki/Cross_product) of $t$ with $Rx$ and $E = [t_{\leftrightarrow}]R$ is the [essential matrix](https://en.wikipedia.org/wiki/Essential_matrix).

Alas, we have arrived at our epipolar constraint: for every $(x, x')$ correspondence pair, we have an epipolar constraint relating the two by a rotation and translation: ${x'}^\intercal Ex = 0$.

<h4 id="constructing-system">Constructing a system of equations to solve for $E$</h4>

We can construct a system of equations for every $(x, x')$ correspondence pair by unrolling $E$ into a vector of variables (recall we use [homogeneous coordinates](https://en.wikipedia.org/wiki/Homogeneous_coordinates) to describe these 2D points):

$$x = (u, v, 1)^\intercal, \quad x' = (u', v', 1)$$

$$
\begin{bmatrix} u' & v' & 1 \end{bmatrix}
\begin{bmatrix}
e_{11} & e_{12} & e_{13} \\
e_{21} & e_{22} & e_{23} \\
e_{31} & e_{32} & e_{33}
\end{bmatrix}
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = 0
$$

$$
\implies \begin{bmatrix} u'u & u'v & u' & v'u & v'v & v' & u & v & 1 \end{bmatrix}
\begin{bmatrix}
e_{11} \\ e_{12} \\ e_{13} \\ e_{21} \\ e_{22} \\ e_{23} \\ e_{31} \\ e_{32} \\ e_{33}
\end{bmatrix} = 0
$$

Thus, for each $(x, x')$ pair, we have a constraint imposed on the vector $e$ and can construct our system of equations given a set of correspondences:

$$Ae = 0$$

<h5 id="dof">Linear System's Degrees of freedom (DOF)</h5>

The rotation matrix $R$ has 3 DOF (pitch, roll, and yaw) and the translation $t$ has 3 DOF ($x, y, z$), bringing us to 6 DOF. We note that a 3D vector $t = [x, y, z]^\intercal$ can be thought of as having a direction (2 DOF) and a magnitude (1 DOF).

Note, however, that the translation between $O$ and $O'$ is ambiguous in magnitude/scale. To see why, suppose we scale $t$ by some factor $\lambda$. Then the essential matrix becomes

$$E' = [(\lambda t)_{\leftrightarrow}]R = \lambda ([t_{\leftrightarrow}] R) = \lambda E$$

and substituting into the epipolar constraint gets us

$$
\begin{aligned}
{x'}^\intercal E' x &= {x'}^\intercal (\lambda E) x \\
&= \lambda ({x'}^\intercal E x) \\
&= \lambda \cdot 0 \\
&= 0
\end{aligned}
$$

Regardless of scale factor $\lambda$, we cannot recover the magnitude of $t$ from the epipolar constraint alone and therefore can only encode 2 DOF about $t$. When accounting for this, we see that we have 5 DOF in total: 3 from $R$ and 2 from $t$.

<h3 id="ransac-camera-motion">RANSAC for camera motion estimation</h3>

We'll solve our system of equations $Ae = 0$ with [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus). At a high level, we iteratively estimate $e$ from a collection of correspondences (some of which are noisy/outliers) for some $T$ iterations until $e$ is 'good enough'. We say $e$ is good enough when it can accurately enough relate the rotation and translation between correspondences from images 1 and 2: we estimate randomly sample $n$ correspondences and estimate $e$ from them, apply it on some feature in image 1, and see if it lands within some $\epsilon$ distance to the corresponding feature in image 2. If it's within $\epsilon$ distance, we say that this correspondence pair is an *inlier*–otherwise, it's an *outlier*. The best $e$ will be the one that, after $T$ iterations, yields the highest number of inliers. Optionally, at the very end of the $T$ iterations, we can compute the inliers using our $e$ estimate, then *fit* $e$ *again* on the inliers only, yielding an even better $e$ estimate. We can then decompose $e$ into our $R$ and $t$ components.


### Acknowledgements

This assignment was created by Ryan Tabrizi with helpful feedback from Jordan Lin and Aleksander Holynski.

<!-- [Previous version of HW3](/hw3_old/) -->
