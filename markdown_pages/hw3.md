---
title: Homework 3
layout: default
permalink: /hw3/
toc: true
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
</style>

<header>
  <h1>
    Homework 3<br>
    <a href="../../">COMS4732: Computer Vision 2</a>
  </h1>
</header>

<h2 style="text-align: center;">
  <!-- <div style="display: flex; justify-content: center; gap: 1em; align-items: center; flex-wrap: wrap;">
    <img src="/hws/hw2/image002.gif" alt="Feature Matching Example">
    <img src="/hws/hw2/image003.gif" alt="Feature Matching Example 2">
  </div><br> -->
  Simple Structure from Motion<br>
  <b style="color:#9E0000">Due Date: TBD</b>
</h2>

# Background

This assignment will involve determining the 3D position, or pose, of multiple cameras in a scene. We will build off of homework 2's feature matching algorithm and modify RANSAC.

<br>
**Important:** this assignment largely depends on homework 2. Please make sure you have completed it before starting this assignment.
<br>
<br>
**Important:** the overview isn't exhaustive. You are expected to have attended lecture.

## Problem statement

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 1em 0;">
    <img src="/hws/hw3/assets/problem_statement.png" alt="Problem statement diagram" style="max-width: 100%;">
    <figcaption style="margin-top: 0.5em; font-style: italic;">
      <strong>Figure 1:</strong> Left: 3D problem space: you may or may not have 2D correspondences, 3D camera positions/motion, or 3D structure for a given 3D scene. <br> Right: this homework will focus on when we know nothing about the scene.
    </figcaption>
  </figure>
</div>

In the previous assignment, we computed correspondences between features in two images. The goal of this assignment is to similarly find correspondences between features in multiple images that will then be used to estimate the camera poses in the scene. Lastly, we will perform triangulation to also use the camera poses to estimate the 3D structure of the scene.

### Epipolar Geometry





# Step 1: Feature extraction

In HW2 we used Harris corners as a simple feature. Because the task of SfM involves finding correspondences between images whose cameras have been translated and rotated relative to each other, we need to use a more robust feature, particularly one that is invariant to rotation. We will use SIFT (Scale-Invariant Feature Transform) features.

Here's how to use SIFT in OpenCV:

```python
import cv2

# Create SIFT detector, detecting at most max_features features
sift = cv2.SIFT_create(nfeatures=max_features)

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img_uint8, None)

if descriptors is None or len(keypoints) == 0:
    print(f"WARNING: No SIFT features detected!")
    return np.array([[], []]), np.array([])

# Extract coordinates in (x, y) format from keypoints
coords_xy = np.array([kp.pt for kp in keypoints])  # (N, 2) in (x, y)
```

Note: SIFT has keypoint selection baked in according to the `max_features` best keypoints. As such, we don't need to perform ANMS here.


# Step 2: Feature matching

Now, as we did in HW2, we need to find correspondences between features in multiple images. We will use the same approach of visualizing the nearest neighbor distance ratio (NNDR) and setting a threshold on which matched descriptors we'll keep. 

**Note:** With SIFT, L2 is typically used to measure the similarity between descriptors, whereas in HW2 we used NCC.

# Step 3: RANSAC to estimate camera pose

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 1em 0;">
    <img src="/hws/hw3/assets/correspondences_to_pose.png" alt="Correspondences to pose diagram" style="max-width: 100%;">
    <figcaption style="margin-top: 0.5em; font-style: italic;">
      <strong>Figure 2:</strong> Step 3 aims to recover the motion of both cameras, which is another way of saying the rotation $R$ and translation $t$ between our two cameras.
    </figcaption>
  </figure>
</div>

Whereas in HW2 we used RANSAC to estimate the homography that relates two images, here we will use RANSAC to estimate the camera poses that relate two images. We do this via epipola geometry.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 1em 0;">
    <img src="/hws/hw3/assets/epipolar1.png" alt="Epipolar geometry setup" style="max-width: 100%;">
    <figcaption style="margin-top: 0.5em; font-style: italic;">
      <strong>Figure 3:</strong> Epipolar geometry relates two camera views of the same 3D point.
    </figcaption>
  </figure>
</div>

Suppose 2D points $$x$$ and $$x'$$ are the projections of a 3D point $$X$$ onto the image planes of two cameras. The epipolar geometry relates these two points via the essential matrix $$E$$:
$$
x'^T E x = 0
$$


<div style="text-align: center;">
  <figure style="display: inline-block; margin: 1em 0;">
    <img src="/hws/hw3/assets/epipolar2.png" alt="Epipolar geometry setup" style="max-width: 100%;">
    <figcaption style="margin-top: 0.5em; font-style: italic;">
      <strong>Figure 3:</strong> Constructing the epipolar constraint.
    </figcaption>
  </figure>
</div>

To see how we get here, we first observe that $x'$ is related to $x$ by a rotation and translation:

$$x' = R x + t$$

Where $t$ is the translation from camera 1's center of projection $$O$$ to camera 2's center of projection $$O'$$. $R$ is the rotation from camera 1's coordinate system to camera 2's coordinate system.

To relate the fact that $x$, $x'$, and $t$ are all coplanar, we define the normal vector to the plane as $n = t \times x'$ and substitute in the expression for $x'$:

$$
\begin{aligned}
n &= t \times x' \\
n &= t \times (R x + t) \\
n &= t \times R x + \cancel{t \times t} \quad (\text{since } t \times t = 0)\\
n &= t \times R x 
\end{aligned}
$$

Lastly, by definition of $n$, 

$$
\begin{aligned}
x' \cdot n &= 0 \\
x' \cdot (t \times (R x)) &= 0 \\
x'^T [t_{\leftrightarrow}] R x &= 0 \\
x'^T E x &= 0
\end{aligned}
$$

Where $$[t_{\leftrightarrow}]$$ is a [skew-symmetric matrix representing the cross product](https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product) of $t$ with $R x$ and $$E = [t_{\leftrightarrow}] R$$ is the essential matrix.

Thus, for every $(x, x')$ correspondence pair, we have an epipolar constraint relating the two by a rotation and translation: $x'^T E x = 0$.

We can construct a system of equations for every $(x, x')$ correspondence pair by unrolling E into a vector of variables:

$$x = (u, v, 1)^T, \quad x' = (u', v', 1)$$

$$
\begin{bmatrix} u' & v' & 1 \end{bmatrix}
\begin{bmatrix} 
e_{11} & e_{12} & e_{13} \\ 
e_{21} & e_{22} & e_{23} \\ 
e_{31} & e_{32} & e_{33} 
\end{bmatrix}
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = 0
\quad \Rightarrow \quad
\begin{bmatrix} u'u & u'v & u' & v'u & v'v & v' & u & v & 1 \end{bmatrix}
\begin{bmatrix} 
e_{11} \\ e_{12} \\ e_{13} \\ e_{21} \\ e_{22} \\ e_{23} \\ e_{31} \\ e_{32} \\ e_{33} 
\end{bmatrix} = 0
$$

Each of these is a constraint in the system of equations:

$$A e = 0$$

The rotation $$R$$ has 3 degrees of freedom (pitch, roll, yaw), and the translation $$t$$ has 3 degrees of freedom (x, y, z). Thus, we have 6 degrees of freedom in total. Note, however, that in figure 3 the translation between $$O$$ and $$O'$$ is ambiguous in scale. To see why, suppose we scale $$t$$ by some factor $$\lambda$$. Then the essential matrix becomes:

$$E' = [(\lambda t)_{\leftrightarrow}] R = \lambda [t_{\leftrightarrow}] R = \lambda E$$

Substituting this into the epipolar constraint:

$$
\begin{aligned}
x'^T E' x &= x'^T (\lambda E) x \\
&= \lambda (x'^T E x) \\
&= \lambda \cdot 0 \\
&= 0
\end{aligned}
$$

Since the constraint is satisfied regardless of $$\lambda$$, we cannot recover the magnitude of $$t$$ from the epipolar constraint alone. We can only recover the direction of $$t$$, reducing our degrees of freedom from 6 to 5.



### 8-point algorithm

Even though we only have 5 degrees of freedom, and therefore would typically only need 5 correspondences to solve for the essential matrix, [David Nistér 2004](https://www.scribd.com/document/471805325/Nister-5pt-pdf) demonstrates that if you write out these constraints strictly, you end up with a system of equations that reduces to a 10th-degree polynomial that encodes the geometric (rotation and translation) relationship between these 5 degrees of freedom. 

Instead, we can use the 8-point algorithm to solve for the essential matrix without factoring in the geometric constraints, after which we can enforce the geometric constraints to ensure it's a valid essential matrix and recover $$R$$ and $$t$$.

**Why 8 points if $$E$$ has 9 degrees of freedom?** While $$E = [t_{\leftrightarrow}] R$$ has only 5 true degrees of freedom (3 for rotation $$R$$, 2 for the direction of translation $$t$$), we solve for it as a 3×3 matrix with 9 elements. Since we can only recover $$E$$ up to a scale factor, we have 8 unknowns to solve for (9 elements - 1 scale factor = 8). Each correspondence pair $$(x, x')$$ gives us one linear constraint equation, so we need at least 8 correspondences to solve the system $$A \vec{e} = 0$$.

As with HW2, we use RANSAC to estimate the essential matrix by randomly sampling 8 correspondences and solving for the essential matrix expressed as a vector of variables $\vec{e}$:
$$A \vec{e} = 0$$

**Algorithm: 8-Point Essential Matrix Estimation**

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 1em 0;">
    <img src="/hws/hw3/assets/ransac_8_point.png" alt="RANSAC 8-point algorithm" style="max-width: 100%;">
    <figcaption style="margin-top: 0.5em; font-style: italic;">
      <strong>Algorithm 1:</strong> RANSAC 8-point algorithm to estimate essential matrix.
    </figcaption>
  </figure>
</div>

Let's walk through each major step of this algorithm:

### Step 1: Point Normalization (Line 3)

Before we begin estimating the essential matrix, we normalize the image coordinates. This step is crucial for numerical stability.

The normalization transforms 2D image coordinates $$x_i$$ into normalized camera coordinates $$\hat{x}_i$$ by applying the inverse camera matrix $$K^{-1}$$:

$$\hat{x}_i \leftarrow K^{-1}[x_i; 1]$$

This converts from pixel coordinates to normalized image coordinates where the camera's intrinsic parameters (focal length, principal point, etc.) are factored out. The essential matrix $$E$$ relates normalized coordinates, whereas the fundamental matrix $$F$$ (which we don't use here) relates pixel coordinates directly.

**Why normalize?** Without normalization, the elements of the constraint matrix $$A$$ (which we'll build next) can vary by several orders of magnitude, leading to poor numerical conditioning when solving via SVD. Normalized coordinates typically have values in the range $$[-1, 1]$$, making the system well-conditioned.

### Step 2: Constructing the Constraint Matrix (Lines 7-8)

Within each RANSAC iteration, we randomly sample 8 correspondence pairs. For each pair $$(\hat{x}_i, \hat{x}'_i)$$, we construct a row of the constraint matrix $$A$$ based on the epipolar constraint:

$$\hat{x}'^T_i E \hat{x}_i = 0$$

Writing $$\hat{x}_i = (u_i, v_i, 1)^T$$ and $$\hat{x}'_i = (u'_i, v'_i, 1)^T$$, we can expand this as we showed earlier. The vectorized form gives us:

$$\text{vec}(\hat{x}' \hat{x}^T)^T = \begin{bmatrix} u'u & u'v & u' & v'u & v'v & v' & u & v & 1 \end{bmatrix}$$

This becomes one row in our constraint matrix $$A \in \mathbb{R}^{8 \times 9}$$. With 8 correspondence pairs, we get 8 equations (rows).

### Step 3: Solving via SVD (Line 9)

We need to solve the homogeneous linear system:

$$A\vec{e} = 0$$

where $$\vec{e}$$ is the 9-element vector containing the entries of $$E$$ (flattened column-wise or row-wise).

**Why SVD?** For a homogeneous system $$A\vec{e} = 0$$, the solution lies in the null space of $$A$$. The Singular Value Decomposition gives us:

$$A = U \Sigma V^T$$

where the columns of $$V$$ corresponding to zero (or near-zero) singular values span the null space of $$A$$. Since we have 8 equations and 9 unknowns, the system is underdetermined by 1, meaning the null space is 1-dimensional. The solution $$\vec{e}$$ is the last column of $$V$$ (corresponding to the smallest singular value).

Once we have $$\vec{e}$$, we reshape it back into a $$3 \times 3$$ matrix to get $$E_{raw}$$.

### Step 4: Enforcing the Essential Matrix Constraint (Lines 10-12)

The matrix $$E_{raw}$$ we just computed might not satisfy the properties of a true essential matrix. Recall that $$E = [t_{\leftrightarrow}] R$$, where $$[t_{\leftrightarrow}]$$ is rank-2 and $$R$$ is a rotation matrix. This structure imposes two critical constraints on $$E$$:

1. **Rank 2**: $$E$$ must have rank 2 (one zero singular value)
2. **Two equal singular values**: The two non-zero singular values must be equal

To enforce these constraints, we "project" $$E_{raw}$$ onto the space of valid essential matrices:

$$E_{raw} = U_E \Sigma V_E^T = U_E \text{diag}(\sigma_1, \sigma_2, \sigma_3) V_E^T$$

We then construct the corrected essential matrix by:

$$E = U_E \text{diag}(1, 1, 0) V_E^T$$

This forces the two largest singular values to be equal (set to 1) and the smallest to be exactly zero, ensuring $$E$$ is rank-2 and has the proper structure.

**Why does this work?** This is the closest rank-2 matrix to $$E_{raw}$$ in the Frobenius norm sense, and setting the two non-zero singular values to be equal enforces the constraint that comes from $$E = [t_{\leftrightarrow}] R$$.

### Step 5: Computing Inliers (Line 13)

Now we evaluate how well our estimated $$E$$ fits all $$N$$ correspondence pairs (not just the 8 we sampled). For each normalized correspondence pair $$(\hat{x}_i, \hat{x}'_i)$$, we compute the **epipolar distance**:

$$d_i = (\hat{x}'_i)^T E \hat{x}_i$$

Ideally, if $$(\hat{x}_i, \hat{x}'_i)$$ are true correspondences of the same 3D point and $$E$$ is correct, this distance should be zero. In practice, due to noise and measurement errors, we accept points as inliers if:

$$|d_i| = |(\hat{x}'_i)^T E \hat{x}_i| < \tau$$

where $$\tau$$ is a threshold (a hyperparameter you tune based on expected noise levels).

The set of inliers:

$$S_{curr} = \{i \mid |(\hat{x}'_i)^T E \hat{x}_i| < \tau\}$$

represents the correspondences that are consistent with this hypothesis for $$E$$.

### Step 6: Recovering Camera Pose (Lines 19-20)

After RANSAC completes, we have the best essential matrix $$E_{best}$$. Now we need to extract the rotation $$R$$ and translation $$t$$ from it.

**The Ambiguity Problem**: Given an essential matrix $$E$$, there are **four possible** solutions for $$(R, t)$$:

$$
\begin{aligned}
&(R_1, t), \quad (R_1, -t)\\
&(R_2, t), \quad (R_2, -t)
\end{aligned}
$$

where $$R_1$$ and $$R_2$$ are two different rotation matrices and $$t$$ can point in either direction.

These four solutions can be extracted from the SVD of $$E$$. The decomposition procedure (which we won't derive here but can be found in Hartley & Zisserman's "Multiple View Geometry") gives us these four candidates.

**Cheirality Check**: Only one of these four solutions is physically valid. The **cheirality constraint** requires that the reconstructed 3D points lie **in front of both cameras** (positive depth). For each of the four solutions:

1. Triangulate the 3D positions of points in $$S_{best}$$ using the candidate $$(R, t)$$
2. Check if the reconstructed 3D points have positive depth in both camera coordinate systems
3. The solution where the most points (ideally all inliers) have positive depth in both cameras is the correct one

This eliminates the ambiguity and gives us the final $$(R, t)$$.

**Note on scale ambiguity**: As discussed earlier, we can only recover $$t$$ up to a scale factor. The translation vector we recover will have unit norm $$\|t\| = 1$$, representing only the direction of translation. The actual scale must be determined through additional information (e.g., known object sizes or camera baselines).


# Step 4: Triangulation to recover 3D structure

Now that we have recovered the camera poses (rotation $$R$$ and translation $$t$$ between cameras), we can use the 2D correspondences to reconstruct the 3D positions of the points in the scene. This process is called **triangulation**.

## The Triangulation Problem

Given:
- Two camera poses: Camera 1 at the origin with identity rotation, and Camera 2 with rotation $$R$$ and translation $$t$$
- A correspondence pair $$(\hat{x}, \hat{x}')$$ in normalized image coordinates
- Camera projection matrices $$P_1$$ and $$P_2$$

We want to find the 3D point $$X$$ that projects to $$\hat{x}$$ in camera 1 and $$\hat{x}'$$ in camera 2.

Without loss of generality, we can set the first camera's pose as the world coordinate frame:
$$P_1 = K[I | 0] = [I | 0] \text{ (since we work in normalized coordinates)}$$

The second camera's projection matrix incorporates the rotation and translation:
$$P_2 = [R | t]$$

## The Constraint

For a 3D point $$X = (X, Y, Z, 1)^T$$ in homogeneous coordinates, its projection onto camera $$i$$ should satisfy:
$$\hat{x}_i \sim P_i X$$

where $$\sim$$ denotes equality up to scale (since we're in homogeneous coordinates). This means that $$\hat{x}_i$$ and $$P_i X$$ should be parallel, or equivalently, their cross product should be zero:
$$\hat{x}_i \times (P_i X) = 0$$

## Linear Triangulation via DLT

The cross product constraint gives us a system of linear equations. For each camera, writing $$\hat{x} = (u, v, 1)^T$$ and $$P_i = [p_i^1; p_i^2; p_i^3]$$ (where $$p_i^j$$ is the $$j$$-th row of $$P_i$$), the cross product expands to:

$$
\begin{bmatrix} 
u \\ v \\ 1 
\end{bmatrix} 
\times 
\begin{bmatrix} 
p_1^T X \\ p_2^T X \\ p_3^T X 
\end{bmatrix} 
= 
\begin{bmatrix} 
v(p_3^T X) - (p_2^T X) \\ 
(p_1^T X) - u(p_3^T X) \\ 
u(p_2^T X) - v(p_1^T X) 
\end{bmatrix}
= 0
$$

This gives us three equations, but only two are linearly independent (the third is a linear combination of the first two). Rearranging:

$$
\begin{aligned}
u(p_3^T X) - (p_1^T X) &= 0 \quad \Rightarrow \quad (u p_3^T - p_1^T) X = 0\\
v(p_3^T X) - (p_2^T X) &= 0 \quad \Rightarrow \quad (v p_3^T - p_2^T) X = 0
\end{aligned}
$$

Each camera gives us 2 equations. With 2 cameras, we get 4 equations for the 3D point $$X$$ (which has 4 components in homogeneous coordinates, but only 3 degrees of freedom due to scale). Stacking these equations:

$$
A X = 
\begin{bmatrix}
u_1 p_{1,3}^T - p_{1,1}^T \\
v_1 p_{1,3}^T - p_{1,2}^T \\
u_2 p_{2,3}^T - p_{2,1}^T \\
v_2 p_{2,3}^T - p_{2,2}^T
\end{bmatrix}
X = 0
$$

where subscripts 1 and 2 denote camera 1 and camera 2, respectively.

**Solving via SVD**: Just as we did for the essential matrix, we solve this homogeneous system $$AX = 0$$ using SVD. The solution is the last column of $$V$$ (corresponding to the smallest singular value) from the decomposition $$A = U\Sigma V^T$$.

The resulting 4D homogeneous point must be converted to 3D Euclidean coordinates by dividing by the last component:
$$X_{euclidean} = \begin{bmatrix} X/W \\ Y/W \\ Z/W \end{bmatrix}$$
where $$X = (X, Y, Z, W)^T$$ in homogeneous coordinates.

## Triangulation for All Inliers

We triangulate all correspondences in $$S_{best}$$ (the inlier set from RANSAC) to reconstruct the 3D structure of the scene. Each correspondence gives us one 3D point.

**Quality Check**: After triangulation, it's good practice to:
1. Verify that triangulated points have positive depth in both cameras (this should be satisfied if the cheirality check was done correctly)
2. Check the reprojection error: project the 3D point back onto both images and measure the distance to the original 2D points
3. Discard points with large reprojection errors, as they likely correspond to outliers or poorly conditioned triangulation

**Why can triangulation fail?** Even with correct correspondences, triangulation can be poorly conditioned when:
- The two camera centers are too close (small baseline)
- The viewing directions are nearly parallel
- The point is very far from both cameras

These scenarios lead to large uncertainty in the depth estimate. A good rule of thumb is that the angle between the two viewing rays should be at least 2-5 degrees for reliable triangulation.


# Step 5: Refinement with bundle adjustment

At this point, we have estimates for:
- The camera poses: $$R$$ and $$t$$ (from Step 3)
- The 3D structure: positions of all inlier points $$\{X_j\}$$ (from Step 4)

However, these estimates are not optimal. Why?

1. **RANSAC gives a good but not optimal solution**: RANSAC finds a robust estimate by maximizing inliers, but it doesn't minimize reprojection error
2. **Errors accumulate**: Small errors in pose estimation affect triangulation, and vice versa
3. **We solved components separately**: The pose and structure were estimated in separate steps, not jointly

**Bundle adjustment** is a joint optimization that refines both camera poses and 3D structure simultaneously by minimizing the reprojection error across all observations.

## The Bundle Adjustment Problem

Given:
- Initial camera poses (rotations and translations)
- Initial 3D point positions
- 2D observations (correspondences) of 3D points in multiple images

We want to minimize the **reprojection error** - the sum of squared distances between observed 2D points and the projections of the estimated 3D points:

$$
\min_{R, t, \{X_j\}} \sum_{i,j} \left\| x_{ij} - \pi(P_i X_j) \right\|^2
$$

where:
- $$i$$ indexes cameras (views)
- $$j$$ indexes 3D points
- $$x_{ij}$$ is the observed 2D position of point $$j$$ in camera $$i$$
- $$P_i$$ is the projection matrix for camera $$i$$ (function of $$R_i, t_i$$)
- $$\pi()$$ is the projection function that maps 3D points to 2D
- $$X_j$$ is the 3D position of point $$j$$

The term "bundle adjustment" comes from the idea of adjusting the "bundle" of light rays (projections) from each 3D point to each camera so they all meet consistently at the 3D point positions.

## Why is this a non-linear optimization?

The projection function $$\pi(P_i X_j)$$ involves:
1. **Rotation**: Rotating 3D points is non-linear (rotation matrices involve trigonometric functions)
2. **Division by depth**: Converting from 3D to 2D via $$\pi([X,Y,Z]^T) = [X/Z, Y/Z]^T$$ introduces non-linearity

Therefore, we cannot solve this with simple linear least squares. Instead, we use **non-linear least squares optimization**, typically with the **Levenberg-Marquardt algorithm** (a combination of gradient descent and Gauss-Newton method).

## The Structure of Bundle Adjustment

Bundle adjustment optimizes many parameters simultaneously:
- Camera parameters: For $$n$$ cameras, we have rotation (3 DOF per camera) and translation (3 DOF per camera) = $$6n$$ parameters
- 3D point positions: For $$m$$ 3D points, we have $$3m$$ parameters

Total: $$6n + 3m$$ parameters to optimize!

For a typical SfM problem with, say, 100 images and 10,000 points, that's 30,600 parameters.

**How is this tractable?** The key insight is that the problem has **sparse structure**:
- Each 3D point is only observed in a subset of cameras
- The reprojection error for point $$j$$ in camera $$i$$ only depends on $$X_j$$, $$R_i$$, and $$t_i$$

This means the Jacobian matrix (matrix of partial derivatives) is extremely sparse, which can be exploited by specialized sparse optimization algorithms. Libraries like `scipy.optimize.least_squares` with the `method='lm'` option or specialized SfM libraries leverage this sparsity.

## Implementing Bundle Adjustment

In practice, you would:

1. **Parameterize the problem**: 
   - Represent rotations using a minimal 3-parameter representation (e.g., axis-angle or Euler angles)
   - Pack all parameters into a single vector: $$\theta = [r_1, t_1, r_2, t_2, ..., X_1, X_2, ...]$$

2. **Define the residual function**:
   ```python
   def residuals(params, observations, n_cameras, n_points):
       """
       params: packed vector of camera and point parameters
       observations: list of (camera_idx, point_idx, observed_2d_point)
       returns: vector of residuals (difference between observed and projected)
       """
       camera_params = params[:n_cameras * 6]  # 6 per camera
       points_3d = params[n_cameras * 6:].reshape(-1, 3)  # rest are 3D points
       
       residuals = []
       for cam_idx, point_idx, observed in observations:
           R, t = extract_camera_params(camera_params, cam_idx)
           X = points_3d[point_idx]
           projected = project(R, t, X)
           residuals.append(observed - projected)
       return np.concatenate(residuals)
   ```

3. **Run the optimizer**:
   ```python
   from scipy.optimize import least_squares
   
   result = least_squares(
       residuals, 
       initial_params,
       method='lm',  # Levenberg-Marquardt
       args=(observations, n_cameras, n_points)
   )
   optimized_params = result.x
   ```

## Practical Considerations

**Convergence**: Bundle adjustment is an iterative algorithm. It typically converges in 10-50 iterations if initialized well (which is why we do RANSAC + triangulation first).

**Local minima**: Like all non-linear optimization, bundle adjustment can get stuck in local minima. Good initialization (from RANSAC and triangulation) is critical.

**Outliers**: Bundle adjustment assumes all observations are correct. A single bad correspondence can significantly corrupt the solution. It's important to:
- Run bundle adjustment only on verified inliers from RANSAC
- Optionally use robust cost functions (e.g., Huber loss) that downweight large errors

**Degeneracies**: Certain camera configurations (e.g., all cameras looking in the same direction) can make the problem ill-conditioned. In practice, having diverse viewpoints improves stability.

## The Impact of Bundle Adjustment

Bundle adjustment typically reduces reprojection error by 50-90% compared to the initial RANSAC + triangulation solution. This refinement is essential for:
- High-quality 3D reconstructions
- Accurate camera pose estimation for AR/VR applications  
- Multi-view stereo and dense reconstruction pipelines

Think of bundle adjustment as the "polish" step that takes a good initial estimate and makes it great by ensuring global consistency across all cameras and points.