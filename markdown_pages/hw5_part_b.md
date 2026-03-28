---
title: "Part B: Flow Matching from Scratch"
layout: default
permalink: /hw5/part-b/
parent: Homework 5
nav_order: 2
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
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.0.3/styles/default.min.css" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.0.3/highlight.min.js"></script>
<script>hljs.initHighlightingOnLoad();</script>

<style>

code {
background-color: #f4f4f4;
padding: 5px;
border-radius: 5px;
}
.image-container {
display: flex;
flex-wrap: wrap;
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container {
display: flex;
flex-wrap: wrap;
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container > div {
display: flex;
flex-direction: column;
align-items: center;
width: 14%;
min-width: 75px;
max-width: 100px;
position: relative;
padding: 20px;
overflow: visible;
}

.image-container img {
width: 100%;
height: auto;
transform-origin: center center;
}

.image-container p {
text-align: center;
margin-top: 10px;
}
/* Two image containers */
.column {
float: left;
width: 45%;
padding: 5px;
}

/* Clear floats after image containers */
.row::after {
content: "";
clear: both;
display: table;
}

@keyframes rotate180 {
from {
transform: rotate(0deg);
}
to {
transform: rotate(180deg);
}
}
      
.rotating-image {
transition: transform 1.5s;
transform: rotate(0deg);
}
      
.rotating-image:hover {
transform: rotate(180deg);
}

.zoom-animation {
transition: transform 1s ease-in-out;
transform: scale(1);
}

.zoom-animation:hover,
.zoom-animation.active {
transform: scale(0.25);
}

.rotating-image {
transition: transform 1.5s;
transform: rotate(0deg);
}

.rotating-image:hover,
.rotating-image.active {
transform: rotate(180deg);
}

.caption-container {
position: relative;
height: auto;
min-height: 2em;
text-align: center;
width: 100%;
padding: 5px 0;
}

.caption-default, .caption-transform {
position: absolute;
width: 100%;
transition: opacity 1.5s;
white-space: normal;
left: 0;
}

.caption-transform {
opacity: 0;
}

.rotating-image:hover + .caption-container .caption-default,
.active + .caption-container .caption-default {
opacity: 0;
}

.rotating-image:hover + .caption-container .caption-transform,
.active + .caption-container .caption-transform {
opacity: 1;
}

.zoom-animation:hover + .caption-container .caption-default {
opacity: 0;
}

.zoom-animation:hover + .caption-container .caption-transform {
opacity: 1;
}

.image-container > div:hover .zoom-animation {
transform: scale(0.25);
}

.image-container > div:hover .caption-default {
opacity: 0;
}

.image-container > div:hover .caption-transform {
opacity: 1;
}

.caption-container .caption-default {
opacity: 1;
transition: opacity 1.5s;
}

.caption-container .caption-transform {
opacity: 0;
transition: opacity 1.5s;
}

.active + .caption-container .caption-default {
opacity: 0;
}

.active + .caption-container .caption-transform {
opacity: 1;
}

.dissolve-container {
position: relative;
width: 100%;
height: 0;
padding-bottom: 100%; /* Creates a square aspect ratio */
margin-bottom: 5px; /* Reduced from 10px to match other captions */
}

.dissolve-image {
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
transition: opacity 1s ease-in-out;
}

.dissolve-image.original {
opacity: 1;
}

.dissolve-image.edited {
opacity: 0;
}

.dissolve-image.original.active {
opacity: 0;
}

.dissolve-image.edited.active {
opacity: 1;
}

/* Ensure consistent caption styling */
.image-container > div p {
text-align: center;
margin-top: 5px;  /* Reduced from 10px to align with other captions */
margin-bottom: 0;
}

/* Hover state */
.dissolve-container:hover .dissolve-image.original {
opacity: 0;
}

.dissolve-container:hover .dissolve-image.edited {
opacity: 1;
}

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
    

code {
background-color: #f4f4f4;
padding: 5px;
border-radius: 5px;
}
.image-container {
display: flex;
flex-wrap: wrap;
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container > div {
display: flex;
flex-direction: column;
align-items: center;
width: 30%;
max-width: 100px;
}

.image-container img {
width: 100%;
height: auto;
}

.image-container p {
text-align: center;
margin-top: 10px;
}
/* Two image containers */
.column {
float: left;
width: 45%;
padding: 5px;
}

/* Clear floats after image containers */
.row::after {
content: "";
clear: both;
display: table;
}
code {
background-color: #f4f4f4;
padding: 2.5px;
border-radius: 5px;
}
.image-container {
display: flex;
flex-wrap: wrap;
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container {
display: flex;
flex-wrap: wrap;
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container > div {
display: flex;
flex-direction: column;
align-items: center;
width: 14%;
min-width: 75px;
max-width: 100px;
position: relative;
padding: 20px;
overflow: visible;
}

.image-container img {
width: 100%;
height: auto;
transform-origin: center center;
}

.image-container p {
text-align: center;
margin-top: 10px;
}
/* Two image containers */
.column {
float: left;
width: 45%;
padding: 5px;
}

/* Clear floats after image containers */
.row::after {
content: "";
clear: both;
display: table;
}

@keyframes rotate180 {
from {
transform: rotate(0deg);
}
to {
transform: rotate(180deg);
}
}
      
.rotating-image {
transition: transform 1.5s;
transform: rotate(0deg);
}
      
.rotating-image:hover {
transform: rotate(180deg);
}

.zoom-animation {
transition: transform 1s ease-in-out;
transform: scale(1);
}

.zoom-animation:hover,
.zoom-animation.active {
transform: scale(0.25);
}

.rotating-image {
transition: transform 1.5s;
transform: rotate(0deg);
}

.rotating-image:hover,
.rotating-image.active {
transform: rotate(180deg);
}

.caption-container {
position: relative;
height: auto;
min-height: 2em;
text-align: center;
width: 100%;
padding: 5px 0;
}

.caption-default, .caption-transform {
position: absolute;
width: 100%;
transition: opacity 1.5s;
white-space: normal;
left: 0;
}

.caption-transform {
opacity: 0;
}

.rotating-image:hover + .caption-container .caption-default,
.active + .caption-container .caption-default {
opacity: 0;
}

.rotating-image:hover + .caption-container .caption-transform,
.active + .caption-container .caption-transform {
opacity: 1;
}

.zoom-animation:hover + .caption-container .caption-default {
opacity: 0;
}

.zoom-animation:hover + .caption-container .caption-transform {
opacity: 1;
}

.image-container > div:hover .zoom-animation {
transform: scale(0.25);
}

.image-container > div:hover .caption-default {
opacity: 0;
}

.image-container > div:hover .caption-transform {
opacity: 1;
}

.caption-container .caption-default {
opacity: 1;
transition: opacity 1.5s;
}

.caption-container .caption-transform {
opacity: 0;
transition: opacity 1.5s;
}

.active + .caption-container .caption-default {
opacity: 0;
}

.active + .caption-container .caption-transform {
opacity: 1;
}

.dissolve-container {
position: relative;
width: 100%;
height: 0;
padding-bottom: 100%; /* Creates a square aspect ratio */
margin-bottom: 5px; /* Reduced from 10px to match other captions */
}

.dissolve-image {
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
transition: opacity 1s ease-in-out;
}

.dissolve-image.original {
opacity: 1;
}

.dissolve-image.edited {
opacity: 0;
}

.dissolve-image.original.active {
opacity: 0;
}

.dissolve-image.edited.active {
opacity: 1;
}

/* Ensure consistent caption styling */
.image-container > div p {
text-align: center;
margin-top: 5px;  /* Reduced from 10px to align with other captions */
margin-bottom: 0;
}

/* Hover state */
.dissolve-container:hover .dissolve-image.original {
opacity: 0;
}

.dissolve-container:hover .dissolve-image.edited {
opacity: 1;
}

/* Add styling for code comments */
code .hljs-comment {
color: #666666;  /* A dark grey color */
}

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

.responsive-code {
  width: 60%;
  margin: 0 auto;
}
.responsive-algo {
  width: 75%;
  display: block; 
  margin-left: auto; 
  margin-right: auto;
}
@media only screen and (max-width: 800px) {
  .responsive-code {
    width: 90%;
  }
  .responsive-algo {
    width: 95%;
  }
}


/* Collapsible section styles */
details.section {
  margin: 0.5em 0;
}

details.section > summary {
  font-size: x-large;
  text-align: left;
  font-weight: bold;
  cursor: pointer;
  padding: 0.25em 0;
  list-style: revert;
}

details.section > summary::-webkit-details-marker {
  display: initial;
}

</style>

<div>

<video id="hero-video" width="640" height="320" muted autoplay playsinline
style="display: block; margin-left: auto; margin-right: auto;">
<source type="video/mp4" src="/hws/hw5/assets/new_c_20_fm.mp4" />
</video>
<p style="text-align: center; font-size: 0.8em; color: #666; margin-top: 4px;">(refresh page to rewatch animations)</p>
</div>

# HW5 Part B: Flow Matching from Scratch!
<a href="../../">COMS4732: Computer Vision 2</a>

[Part A](/hw5/part-a/)

<h2 style="text-align: center">
<b style="color: red;">Due: Friday, April 17 11:59pm PT</b>
</h2>
<h4 style="text-align: center">
<b>We recommend using GPUs from <a
href="https://colab.research.google.com/">Colab</a> to finish this
project! <br>(students get Colab Pro for free)!</b>
</h4>

<br>

<span style="color: darkgreen;">**Starter code can be found [here](/hw5_part_b_startercode/).**</span>

<p style="font-size: 0.85em; margin: 0.5em 1em;">
  <a href="javascript:void(0)" id="toggle-all" onclick="(function(){var d=document.querySelectorAll('details.section'),open=d[0]&&!d[0].open;d.forEach(function(el){el.open=open});document.getElementById('toggle-all').textContent=open?'Collapse all':'Expand all'})()">Collapse all</a>
</p>

<details open class="section" markdown="1">
<summary>Overview</summary>
You will train your own <a href="https://arxiv.org/abs/2210.02747">flow matching</a> model on MNIST. Starter code can
be found in the <a
href="https://colab.research.google.com/drive/1GqpAzvLuPwYiwJaY0xLEqdx5IkBNqk1B?usp=drive_link">provided
notebook</a>.
<br>
<br>

## Neural Network Resources
<p>
In this part, you will build and train a 
<a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a>, 
which is more complex than the MLP you implemented in the NeRF project. 
We provide all class definitions you may need in the notebook (but feel free to add or modify them as necessary).  
</p>

<p>
Instead of asking ChatGPT to write everything for you, please consult the following resources when you get stuck — 
they will help you understand how and why things work under the hood.
</p>

<ul>
<li>
PyTorch Documentation — 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html" target="_blank"><code>Conv2d</code></a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html" target="_blank"><code>ConvTranspose2d</code></a>, and
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html" target="_blank"><code>AvgPool2d</code></a>.
</li>
<li>
PyTorch Documentation — 
<a href="https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html" target="_blank"><code>torchvision.datasets.MNIST</code></a>, 
the dataset we’re going to use, and 
<a href="https://docs.pytorch.org/docs/stable/data.html" target="_blank"><code>torch.utils.data.DataLoader</code></a>, 
the off-the-shelf dataloader we can directly use.
</li>
<li>
PyTorch 
<a href="https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html" target="_blank">tutorial</a> 
on how to train a classifier on the CIFAR10 dataset. 
The structure of your training code will be very similar to this one.
</li>
</ul>
<br>

<!-- <p>Note: this is an updated version of <a href="https://cal-cs180.github.io/fa24/hw/proj5/">CS180's Project 5</a> part B with flow matching instead of DDPM diffusion. For the DDPM version, please see <a href="https://cal-cs180.github.io/fa24/hw/proj5/partb.html">here</a>.</p> -->

</details>

<details open class="section" markdown="1">
<summary>Part 1: Training a Single-Step Denoising UNet</summary>
<p class="text">
Let's warmup by building a simple one-step denoiser. Given a noisy image
$z$, we
aim to train a denoiser $D_\theta$ such that it maps $z$ to a clean
image $x$. To do so, we can optimize over an L2 loss:
$$L = \mathbb{E}_{z,x} \|D_{\theta}(z) - x\|^2 \tag{B.1}$$
</p>

##  1.1 Implementing the UNet
In this project, we implement the denoiser as a <a
href="https://arxiv.org/abs/1505.04597"> UNet</a>. It consists of a
few downsampling and upsampling blocks with skip connections. **Note: you should implement the exact UNet below. As bells & whistles you can modify it.**
<br>
<br>
<div style="text-align: center;">
<img src="/hws/hw5/assets/unconditional_arch.png" alt="UNet Architecture" height="500"
style="display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 1: Unconditional UNet</p>
</div>

<p>The diagram above uses a number of standard tensor operations defined as
follows:</p>
<div style="text-align: center;">
<img src="/hws/hw5/assets/atomic_ops_new.png" alt="UNet Operations" height="400"
style="display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 2: Standard UNet Operations</p>
</div>

<!-- <br><br> -->

where:
<ul>
<li><b><tt>Conv2d(kernel_size, stride, padding)</tt></b> is
<code>nn.Conv2d()</code></li>
<li><b><tt>BN</tt></b> is <code>nn.BatchNorm2d()</code></li>
<li><b><tt>GELU</tt></b> is <code>nn.GELU()</code></li>
<li><b><tt>ConvTranspose2d(kernel_size, stride, padding)</tt></b> is
<code>nn.ConvTranspose2d()</code></li>
<li><b><tt>AvgPool(kernel_size)</tt></b> is
<code>nn.AvgPool2d()</code></li>
<li><code>D</code> is the number of hidden channels and is a
hyperparameter that we will set ourselves.</li>
</ul>

At a high level, the blocks do the following:
<ul>
<li><b><tt>(1) Conv</tt></b> is a convolutional layer that doesn't
change the image resolution, only the channel dimension.</li>
<li><b><tt>(2) DownConv</tt></b> is a convolutional layer that
downsamples the tensor by 2.</li>
<li><b><tt>(3) UpConv</tt></b> is a convolutional layer that upsamples
the tensor by 2.</li>
<li><b><tt>(4) Flatten</tt></b> is an average pooling layer that
flattens a 7x7 tensor into a 1x1 tensor. 7 is the resulting height and
width after the downsampling operations.</li>
<li><b><tt>(5) Unflatten</tt></b> is a convolutional layer that
unflattens/upsamples a 1x1 tensor into a 7x7 tensor.</li>
<li><b><tt>(6) Concat</tt></b> is a channel-wise concatenation between
tensors with the same 2D shape. This is simply
<code>torch.cat()</code>.</li>
</ul>

<p class="text">
We define composed operations using our simple operations in order to
make our network deeper. This doesn't change the tensor's height, width,
or number of channels, but simply adds more learnable parameters.
<!-- <ul>
<li><b><tt>(7) ConvBlock</tt></b>, is similar to <b><tt>Conv</tt></b>
but includes an additional <b><tt>Conv</tt></b>. Note that it has
the same input and output shape as <b><tt> (1) Conv</tt></b>.</li>
<li><b><tt>(8) DownBlock</tt></b>, is similar to
<b><tt>DownConv</tt></b> but includes an additional
<b><tt>ConvBlock</tt></b>. Note that it has the same input and
output shape as <b><tt> (2) DownConv</tt></b>.</li>
<li><b><tt>(9) UpBlock</tt></b>, is similar to <b><tt>UpConv</tt></b>
but includes an additional <b><tt>ConvBlock</tt></b>. Note that it
has the same input and output shape as <b><tt> (3)
UpConv</tt></b>.</li>
</ul> -->
</p>

##  1.2 Using the UNet to Train a Denoiser
Recall from equation 1 that we aim to solve the following denoising
problem:

Given a noisy image $z$, we
aim to train a denoiser $D_\theta$ such that it maps $z$ to a clean
image $x$. To do so, we can optimize over an L2 loss
$$
L = \mathbb{E}_{z,x} \|D_{\theta}(z) - x\|^2.
$$

To train our denoiser, we need to generate training data pairs of ($z$,
$x$), where each $x$ is a clean MNIST digit. For each training batch, we
can generate noisy $z$ from clean $x$. One such way is the following noising process:
$$
z = x + \sigma \epsilon,\quad \text{where }\epsilon \sim N(0, I). \tag{B.2}
$$

Visualize the different noising processes over $\sigma = [0.0, 0.2, 0.4,
0.5, 0.6, 0.8, 1.0]$, assuming normalized $x \in [0, 1]$.



### <span style="color: red;">Deliverable</span>
<ul>
<li>A visualization of the noising process using $\sigma = [0.0,
0.2, 0.4, 0.5, 0.6, 0.8, 1.0]$.</li>
</ul>

### <span style="color: green;">Hint</span>
<ul>
<li> You should see noisier images as $\sigma$ increases.
</li>
</ul>
<!-- <div style="text-align: center;">
<img src="/hws/hw5/assets/varying_sigma.png" alt="Varying Sigmas" height="600"
style="display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 3: Varying levels of noise on MNIST digits</p>
</div> -->

### 1.2.1 Training
<p class="text">
Now, we will train the model to perform denoising.
</p>
<ul>
<li><b>Objective:</b> Train a denoiser to denoise noisy image $z$ with
$\sigma = 0.5$ applied to a clean image $x$.</li>

<li><b>Dataset and dataloader:</b> Use the MNIST dataset via
<code>torchvision.datasets.MNIST</code>。 Train only on the training set. 
Shuffle the dataset before creating the dataloader. Recommended batch 
size: 256. We'll train over our dataset for 5 epochs.

<ul>
<li>You should only noise the image batches when fetched from the
dataloader so that in every epoch the network will see new noised
images due to a random $\epsilon$, improving generalization.</li>
</ul>

</li>

<li><b>Model:</b> Use the UNet architecture defined in section 1.1 with
recommended hidden dimension <code>D = 128</code>.</li>

<li><b>Optimizer:</b> Use Adam optimizer with learning rate of
1e-4.</li>
</ul>
<!-- <div style="text-align: center;">
<img src="/hws/hw5/assets/training_losses_uncond.png" alt="Training Loss Curve"
height="400"
style="display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 4: Training Loss Curve</p>
</div> -->

<p class="text">You should visualize denoised results on the test set at the end of
training. Display sample results after the 1st and 5th epoch.</p>

<p class="text">
After 5 epoch training, they should look something like these:
</p>
<!-- <div style="text-align: center;">
<img src="/hws/hw5/assets/unet_sample_epoch0.png" alt="After the first epoch"
height="400"
style="display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 5: Results on digits from the test set after 1
epoch of training</p>
</div> -->
<div style="text-align: center;">
<img src="/hws/hw5/assets/unet_sample_epoch5.png" alt="After the 5-th epoch"
height="400"
style="display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 3: Results on digits from the test set after 5
epochs of training</p>
</div>

### <span style="color: red;">Deliverables</span>
<ul>
<li>A training loss curve plot every few iterations during the whole
training process of $\sigma = 0.5$.</li>
<li>Sample results on the test set with noise level 0.5 after the first and the 5-th epoch</li>
</ul>

### 1.2.2 Out-of-Distribution Testing

<p class="text">
Our denoiser was trained on MNIST digits noised with $\sigma = 0.5$. Let's
see how the denoiser performs on different $\sigma$'s that it wasn't
trained for.
</p>
<p class="text">
Visualize the denoiser results on test set digits with varying levels of
noise $\sigma = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]$.
</p>
<!-- <div style="text-align: center;">
<img src="/hws/hw5/assets/out_of_distribution2.png" alt="Varying Sigmas"
style="max-width: 90%; height: auto; display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 7: Results on digits from the test set with varying
noise levels.</p>
</div> -->
### <span style="color: red;">Deliverable</span>
<ul>
<li>Sample results on the test set with out-of-distribution noise levels
after the model is trained. Keep the same image and
vary $\sigma = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]$.</li>
</ul>

### 1.2.3 Denoising Pure Noise
<p>To make denoising a generative task, we'd like to be able to denoise pure, random Gaussian noise. We can think of this as starting with a blank canvas $z = \epsilon$ where $\epsilon \sim N(0, I)$ and denoising it to get a clean image $x$.</p>

<p>Repeat the same training process as in part 1.2.1, but input pure noise $\epsilon \sim N(0, I)$ and denoise it for 5 epochs. Display your results after 1 and 5 epochs.</p>

<p>Sample from the denoiser that was trained to denoise pure noise. What patterns do you observe in the generated outputs? What relationship, if any, do these outputs have with the training images (e.g., digits 0–9)? Why might this be happening?</p>

### <span style="color: red;">Deliverables</span>
<ul>
<li>A training loss curve plot every few iterations during the whole
training process that denoises pure noise.</li>
<li>Sample results on pure noise after the first and the 5-th epoch.</li>
<li>A brief description of the patterns observed in the generated outputs and explanations for why they may exist.</li>
</ul>

### <span style="color: green;">Hints</span>
<ul>
<li>
For the last question, recall that with an MSE loss, the model learns to predict the point that
minimizes the sum of squared distances to all training examples. This is 
closely related to the idea of a centroid in clustering. What does it 
represent in the context of the training images?
</li>
<li>Since training can take a while, <b>we strongly recommend that you
checkpoint your model</b> every epoch onto your personal Google
Drive.
This is because Colab notebooks aren't persistent such that if you are
idle for a while, you will lose connection and your training progress.
This consists of: <ul>
<li>Google Drive mounting.</li>
<li>Epoch-wise model & optimizer checkpointing.</li>
<li>Model & optimizer resuming from checkpoints.</li>
</ul>
</li>
</ul>

</details>

<details open class="section" markdown="1">
<summary>Part 2: Training a Flow Matching Model</summary>
We just saw that one-step denoising does not work well for generative tasks. Instead, we need to iteratively denoise the image, and we will do so with <a href="https://arxiv.org/abs/2210.02747">flow matching</a>. 
Here, we will iteratively denoise an image by training a UNet model to predict the `flow' from our noisy data to clean data.

In our flow matching setup, we sample a pure noise image $x_0 \sim \mathcal{N}(0, I)$ and generate a realistic image $x_1$. 

<p>For iterative denoising, we need to define how intermediate noisy samples are constructed. The simplest approach would be a linear interpolation between noisy $x_0$ and clean $x_1$ for some $x_1$ in our training data:</p>

\begin{equation}
x_t = (1-t)x_0 + tx_1 \quad \text{where } x_0 \sim \mathcal{N}(0, 1), t \in [0, 1]. \tag{B.3}
\end{equation}

This is a vector field describing the position of a point $x_t$ at time $t$ relative to the clean data distribution $p_1(x_1)$ and the noisy data distribution $p_0(x_0)$. Intuitively, we see that for small $t$, we remain close to noise, while for larger $t$, we approach the clean distribution.

<p>Flow can be thought of as the velocity (change in posiiton w.r.t. time) of this vector field, describing how to move from $x_0$ to $x_1$:

\begin{equation} u(x_t, t) = \frac{d}{dt} x_t = x_1 - x_0. \tag{B.4}\end{equation}</p>

<p>Our aim is to learn a UNet $u_\theta(x_t,t)$ which approximates this flow $u(x_t, t) = x_1 - x_0$, giving us our learning objective:

\begin{equation}
L = \mathbb{E}_{x_0 \sim p_0(x_0), x_1 \sim p_1(x_1), t \sim U[0, 1]} \|(x_1-x_0) - u_\theta(x_t, t)\|^2. \tag{B.5}
\end{equation}</p>

## 2.1 Adding Time Conditioning to UNet
We need a way to inject scalar $t$ into our UNet model to condition it. There are many ways to do this. Here is what we suggest:

<div style="text-align: center;">
<div style="text-align: center;">
<img src="/hws/hw5/assets/conditional_arch_fm.png" alt="UNet Highlighted" height="500" />
<p class="text">Figure 4: Conditioned UNet</p>
</div>
</div>

<p><b>Note:</b> It may look like we're predicting the original image in the figure above, but we are not. We're predicting the flow from the noisy $x_0$ to clean $x_1$, which will contain both parts of the original image as well as the noise to remove.</p>

<p class="text">This uses a new operator called
<b><tt>FCBlock</tt></b> (fully-connected block) which we use to inject the conditioning signal into the UNet:</p>
<div style="text-align: center;">
<img src="/hws/hw5/assets/fc_long.png" alt="FCBlock" height="200"
style="display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 5: FCBlock for conditioning</p>
</div>
Here <b><tt>Linear(F_in, F_out)</tt></b> is a linear layer with
<b><tt>F_in</tt></b> input features and <b><tt>F_out</tt></b> output
features. You can implement it using <code>nn.Linear</code>.

<p class="text">Since our conditioning signal $t$ is a scalar, <b><tt>F_in</tt></b> should be of size 1.</p>
    

<p class="text">
You can embed $t$ by following this pseudo code:
</p>
<div class="responsive-code">
<pre><code class="language-python">
fc1_t = FCBlock(...)
fc2_t = FCBlock(...)

# the t passed in here should be normalized
# to be in the range [0, 1]
t1 = fc1_t(t)
t2 = fc2_t(t)

# Follow diagram to get unflatten.
# Replace the original unflatten with modulated unflatten.
unflatten = unflatten * t1
# Follow diagram to get up1.
...
# Replace the original up1 with modulated up1.
up1 = up1 * t2
# Follow diagram to get the output.
...
</code></pre>
</div>

## 2.2 Training the UNet
Training our time-conditioned UNet $u_\theta(x_t, t)$ is now pretty easy. Basically, we pick a random image $x_1$
from the training set, a random timestep $t$, add noise to $x_1$ to get $x_t$, and train the denoiser to predict the flow at $x_t$. We repeat this for different images and different timesteps until the model converges and we are happy.

<br>
<br>

<div style="text-align: center;">
<img src="/hws/hw5/assets/algo1_t_only_fm.png" alt="Algorithm Diagram"
class="responsive-algo" />
<p class="text">Algorithm B.1. Training time-conditioned UNet</p>
</div>

<ul>
<li><b>Objective:</b> Train a time-conditioned UNet $u_\theta(x_t, t)$ to predict the flow at $x_t$ given a noisy image $x_t$ and a timestep $t$.</li>

<li><b>Dataset and dataloader:</b> Use the MNIST dataset via
<code>torchvision.datasets.MNIST</code>. Train only on the training set. Shuffle the dataset
before creating the dataloader. Recommended batch size: 64.
<ul>
<li>As shown in algorithm B.1, You should only noise the image batches when fetched from the
dataloader.</li>
</ul>

</li>

<li><b>Model:</b> Use the time-conditioned UNet architecture defined in section 2.1 with
recommended hidden dimension <code>D = 64</code>. Follow the diagram and pseudocode for how to inject the conditioning signal $t$ into the UNet. Remember to normalize $t$ before embedding it.</li>

<li><b>Optimizer:</b> Use Adam optimizer with an initial learning rate of
1e-2. We will be using an exponential learning rate decay scheduler with a gamma of $0.1^{(1.0 / \text{num_epochs})}$. This can be implemented using <code>scheduler = torch.optim.lr_scheduler.ExponentialLR(...)</code>. You should call <code>scheduler.step()</code> after every epoch.</li>
</ul>
<!-- <div style="text-align: center;">
<img src="/hws/hw5/assets/t_cond_training_fm.png" alt="Loss Curve" height="300" />
<p class="text">Figure 10: Time-Conditioned UNet training loss curve</p>
</div> -->

### <span style="color: red;">Deliverable</span>
<ul>
<li>A training loss curve plot for the time-conditioned UNet over the whole training process. </li>
</ul>


## 2.3 Sampling from the UNet
We can now use our UNet for iterative denoising using the algorithm below! The results would not be perfect, but legible digits should emerge
<br>
<br>
<div style="text-align: center;">
<img src="/hws/hw5/assets/algo2_t_only_fm.png" alt="Algorithm Diagram"
class="responsive-algo" />
<p class="text">Algorithm B.2. Sampling from time-conditioned UNet</p>
</div>

<div class="image-container"
style="justify-content: center; max-width: 1200px; margin: 0 auto;">
<div style="width: 100%; max-width: 600px;">
        
<video id="video1" width="100%" muted autoplay playsinline
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/t_only_e1_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 1</p>
</div>
<div style="width: 100%; max-width: 600px;">
        
<video id="video2" width="100%" muted autoplay playsinline
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/t_only_e10_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 10</p>
</div>
</div>

<!-- Second row with 2 videos
<div class="image-container"
style="justify-content: center; max-width: 1200px; margin: 20px auto 0;">
<div style="width: 100%; max-width: 600px;">
          
<video id="video3" width="100%" muted loop playbackRate="0.75"
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/t_only_e10_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 10</p>
</div>
<div style="width: 100%; max-width: 600px;">
          
<video id="video4" width="100%" muted loop playbackRate="0.75"
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/t_only_e15_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 15</p>
</div>
</div> -->

<!-- Third row with 1 centered video
<div class="image-container"
style="justify-content: center; max-width: 1200px; margin: 20px auto 0;">
<div style="width: 100%; max-width: 600px;">
          
<video id="video5" width="100%" muted loop playbackRate="0.75"
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/t_only_e20_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 20</p>
</div>
</div> -->

### <span style="color: red;">Deliverables</span>
<ul>
<li>Sampling results from the time-conditioned UNet for 1, 5, and 10 epochs. The results should not be perfect, but reasonably good.</li>
<li>(Optional) Check the Bells and Whistles if you want to make it better!</li>
</ul>


## 2.4 Adding Class-Conditioning to UNet
To make the results better and give us more control for image generation, we can also optionally condition our UNet on the class of the digit 0-9. This will require adding 2 more <b><tt>FCBlock</tt></b>s to our UNet but, we suggest that for class-conditioning vector $c$, you make it a one-hot vector instead of a single scalar. 

Because we still want our UNet to work without it being conditioned on the class (recall the classifer-free guidance you implemented in part a), we implement dropout where 10% of the time ($p_{\text{uncond}}= 0.1$) we drop the class conditioning vector $c$ by setting it to 0.

Here is one way to condition our UNet $u_\theta(x_t, t, c)$ on both time $t$ and class $c$:

<div class="responsive-code">
<pre><code class="language-python">
fc1_t = FCBlock(...)
fc1_c = FCBlock(...)
fc2_t = FCBlock(...)
fc2_c = FCBlock(...)

t1 = fc1_t(t)
c1 = fc1_c(c)
t2 = fc2_t(t)
c2 = fc2_c(c)

# Follow diagram to get unflatten.
# Replace the original unflatten with modulated unflatten.
unflatten = c1 * unflatten + t1
# Follow diagram to get up1.
...
# Replace the original up1 with modulated up1.
up1 = c2 * up1 + t2
# Follow diagram to get the output.
...
</code></pre>
</div>

## 2.5 Training the UNet
<p>Training for this section will be the same as time-only, with the only difference being the conditioning vector $c$ and doing unconditional generation periodically.</p>
<br>
        

<div style="text-align: center;">
<img src="/hws/hw5/assets/algo3_c_fm.png" alt="Algorithm Diagram"
class="responsive-algo" />
<p class="text">Algorithm B.3. Training class-conditioned UNet</p>
</div>
    
### <span style="color: red;">Deliverable</span>
<ul>
<li>A training loss curve plot for the class-conditioned UNet over the whole training process. </li>
</ul>

<!-- <div style="text-align: center;"></div>
<div style="text-align: center;">
          
<img src="/hws/hw5/assets/correct_c_losses_fm.png" alt="Training Loss Curve"
style="width: 500px; height: auto; display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 11: Class-conditioned UNet training loss curve</p>
</div>
</div> -->

## 2.6 Sampling from the UNet 
<!-- The sampling process is the same as part A, where we saw that conditional results aren't good unless we use classifier-free guidance. Use classifier-free guidance with $\gamma = 5.0$ for this part. -->
Now we will sample with class-conditioning and will use classifier-free guidance with $\gamma = 5.0$.
<br>
<br>
<div style="text-align: center;">
        
<img src="/hws/hw5/assets/algo4_c_fm.png" alt="Algorithm Diagram"
class="responsive-algo" />
<p class="text">Algorithm B.4. Sampling from class-conditioned UNet</p>
        
</div>

<!-- First row with 2 videos -->
<div class="image-container"
style="justify-content: center; max-width: 1200px; margin: 0 auto;">
<div style="width: 100%; max-width: 600px;">
          
<video id="video1" width="100%" muted autoplay playsinline
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/new_c_1_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 1</p>
</div>
<div style="width: 100%; max-width: 600px;">
          
<video id="video2" width="100%" muted autoplay playsinline
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/new_c_10_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 10</p>
</div>
</div>

<!-- Second row with 2 videos
<div class="image-container"
style="justify-content: center; max-width: 1200px; margin: 20px auto 0;">
<div style="width: 100%; max-width: 600px;">
          
<video id="video3" width="100%" muted loop playbackRate="0.75"
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/new_c_10_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 10</p>
</div>
<div style="width: 100%; max-width: 600px;">
          
<video id="video4" width="100%" muted loop playbackRate="0.75"
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/new_c_15_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 15</p>
</div>
</div> -->

<!-- Third row with 1 centered video
<div class="image-container"
style="justify-content: center; max-width: 1200px; margin: 20px auto 0;">
<div style="width: 100%; max-width: 600px;">
          
<video id="video5" width="100%" muted loop playbackRate="0.75"
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/new_c_20_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 20</p>
</div>
</div> -->


### <span style="color: red;">Deliverables</span>
<ul>
<li>Sampling results from the class-conditioned UNet for 1, 5, and 10 epochs. Class-conditioning lets us converge faster, hence why we only train for 10 epochs. Generate 4 instances of each digit as shown above.
</li>
<li><b>Can we get rid of the annoying learning rate scheduler?</b> Simplicity is the best. Please try to maintain the same performance after removing the exponential 
learning rate scheduler. Show your visualization after training without the scheduler and provide a description of what you did to compensate for the loss of the scheduler.</li>
</ul>

</details>

<details open class="section" markdown="1">
<summary>Bells &amp; Whistles (Optional)</summary>

<!-- <b>Required for CS280A students only:</b> -->
<ul>
<li><b>A better time-conditioned only UNet: </b> Our time-conditioning only UNet in part 2.3 is actually far from perfect. Its result is way worse than the UNet conditioned by both time and class.
We can definitively make it better! Show a better visualization image for the time-conditioning only network. Possible approaches include extending the training schedule or making the architecture more expressive. </li>
</ul>
<ul>
<li><b>Your own ideas</b>: Be creative! This UNet can generate images more than digits! You can try it on <a href="http://ufldl.stanford.edu/housenumbers/">SVHN</a> (still digits, but more fancy!), <a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST</a> (not digits, but still grayscale!), or <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR10</a>!</li>
</ul>

</details>

<details open class="section" markdown="1">
<summary><span style="color: red;">Deliverable Checklist</span></summary>

<ul>
<li>Make sure that your website and submission include <b>all the deliverables</b> in each section above.</li>
<li>As with all past assignments, submit your <b>PDF</b>, <b>webpage</b>, and <b>code</b> to corresponding assignments on Gradescope.</li>
<!-- <li>
<b>The Google Form is required for Part B.</b> Once you have finished both parts A and B, submit the link to your webpage (containing both parts) using this
<a href="https://forms.gle/gLQhNCyBUaCACt7W6">Google Form</a>.
</li> -->
</ul>

</details>

<details open class="section" markdown="1">
<summary>Acknowledgements</summary>
<p>This project was a joint effort by <a
href="https://ryantabrizi.com/">Ryan Tabrizi</a>, <a
href="https://dangeng.github.io/">Daniel Geng</a>, <a
href="https://hangg7.com/">Hang Gao</a>, and <a 
href="https://jingfeng0705.github.io/">Jingfeng Yang</a>, advised by <a
href="https://liyueshen.engin.umich.edu/">Liyue Shen</a>, <a
href="https://andrewowens.com/">Andrew Owens</a>, 
<a href="https://people.eecs.berkeley.edu/~kanazawa/">Angjoo Kanazawa</a>,
and <a
href="https://people.eecs.berkeley.edu/~efros/">Alexei
Efros</a>. We also thank <a href="https://mcallisterdavid.com/">David McAllister</a> and <a href="https://songweige.github.io/">Songwei Ge</a> for their helpful feedback and suggestions.</p>

</details>
    