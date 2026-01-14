---
title: Homework 5
layout: default
permalink: /hw5/
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
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container {
display: flex;
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container > div {
display: flex;
flex-direction: column;
align-items: center;
width: 40%;
max-width: 200px;
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
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container > div {
display: flex;
flex-direction: column;
align-items: center;
width: 30%;
max-width: 200px;
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
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container {
display: flex;
justify-content: center;
align-items: flex-start;
gap: 20px;
}

.image-container > div {
display: flex;
flex-direction: column;
align-items: center;
width: 40%;
max-width: 200px;
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
    
</style>

<div class="image-container">
<div>
<div class="dissolve-container">
<img src="/hws/hw5/assets/hole_filling3.png" alt="Original Campanile"
class="dissolve-image original">
<img src="/hws/hw5/assets/hole_filling.png" alt="Hole Filled"
class="dissolve-image edited">
</div>
<p><b>Hole Filling</b></p>
</div>
<div>
<!-- <div class="dissolve-container">
<img src="/hws/hw5/assets/dog.png" alt="Original Dog"
class="dissolve-image original">
<img src="/hws/hw5/assets/sdedit_dog.png" alt="Edited Dog"
class="dissolve-image edited">
</div> -->
<div class="dissolve-container">
<img src="/hws/hw5/assets/pixel_bear.png" alt="Original Dog"
class="dissolve-image original">
<img src="/hws/hw5/assets/sdedit_bear2.png" alt="Edited Dog"
class="dissolve-image edited">
</div>
<p><b>"Make it Real"</b></p>
</div>
<div>
<img src="/hws/hw5/assets/skull2.png" alt="Man Wearing Hat"
class="zoom-animation">
<div class="caption-container">
<p class="caption-default"><b>A Lithograph of a Waterfall</b></p>
<p class="caption-transform"><b>A Lithograph of a Skull</b></p>
</div>
</div>
<div>
<img src="/hws/hw5/assets/old_man.png" alt="Bear Dancing" class="rotating-image">
<div class="caption-container">
<p class="caption-default"><b>An Oil Painting of an Old Man</b></p>
<p class="caption-transform"><b>An Oil Painting of People Around a
Fire</b></p>
</div>
        
</div>

</div>

<br />

# HW5 Part A: The Power of Diffusion Models!
<a href="../../">COMS4732: Computer Vision 2</a>

<h2 style="text-align: center">
<b style="color: red;">Due: TBD</b>
</h2>
<h4 style="text-align: center">
<b>We recommend using GPUs from <a
href="https://colab.research.google.com/">Colab</a> to finish this
project!</b>
</h4>

## Overview
<p>In part A you will play around with diffusion models, implement diffusion
sampling loops, and use them for other tasks such as inpainting and
creating optical illusions.
Instructions can be found below and in the <a
href="https://colab.research.google.com/drive/19mp-ssAv3CQuVvFsUu2VvWEwnqLds9gx?usp=sharing">provided
notebook</a>.</p>

<p>Because part A is simply to get your feet wet with pre-trained diffusion
models, all deliverables should be completed in the notebook. You will
still submit a webpage with your results.</p>

<p style="color: #CC0000; display: inline;"><b>START EARLY!</b></p><span style="margin-left: 5px;"> This project, in many ways, will be the most difficult project this semester.</span>

#  Part 0: Setup
### Gaining Access to DeepFloyd
<p class="text">
We are going to use the <a
href="https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if">DeepFloyd
IF</a> diffusion model.
DeepFloyd is a two stage model trained by Stability AI.
The first stage produces images of size $64 \times 64$ and the second
stage takes the outputs of the first stage and generates images of size
$256 \times 256$. We provide upsampling code at the very end of the
notebook, though this is not required in your submission.
Before using DeepFloyd, you must accept its usage conditions. To do so:
</p>
<ol>
<li>Make a <a href="https://huggingface.co/join">Hugging Face account</a>
and log in.</li>
<li>Accept the license on the model card of <a
href="https://huggingface.co/DeepFloyd/IF-I-XL-v1.0">DeepFloyd/IF-I-XL-v1.0</a>.
For affiliation, you can fill in "The University of California, Berkeley."
Accepting the license on the stage I model card will auto accept for the
other IF models.</li>
<li>Log in locally by entering your <a
href="https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens">Hugging
Face Hub access token</a> below. You should be able to find and create
tokens <a href="https://huggingface.co/settings/tokens">here</a>. A read token is enough for this project.</li>
</ol>
### Play with the Model using Your Own Text Prompts!
<p>
DeepFloyd was trained as a text-to-image model, which takes text prompts as input and outputs images that are aligned with the text. 
However, a raw text string cannot be directly used as the model's input — we first need to convert it into a high-dimensional vector (of 4096 dimensions in our case) that the model can understand, a.k.a. prompt embeddings.
</p>
<p>
Since prompt encoders are always very big and hard to run in your notebook, we provide two Huggingface clusters <a href="https://huggingface.co/spaces/jamesoncrate/CS180-T5-Encoder">A</a> and 
<a href="https://huggingface.co/spaces/konpat/CS180-T5-Encoder">B</a> for generating your own prompt embeddings! Both are the same and feel free to use either of them.
Please follow their instructions to create a dictionary of embeddings for your prompts, download the resulting <code>.pth</code> file, and load it in Google Colab. 
</p>
<p>
Please note that both clusters have daily usage limits, so if you're unable to use one, please try another or try again tomorrow. Alternatively, <b>START EARLY</b> and download the <code>.pth</code> file in advance
— you only need to generate it once, and you can reuse the downloaded file afterward. If the official site experiences issues or runs out of computation, you can download one of our precomputed embeddings,
but this is a predefined set of prompts and lacks flexibility. We want to see your creativity!
</p>

<b>Deliverables </b>

<ul>
<li>Come up with some interesting text prompts and generate their embeddings.</li>
<li>Choose 3 of your prompts to generate images and display 
the caption and the output of the model. Reflect on the quality of the outputs and their
relationships to the text prompts. Make sure to try at least 2
different <code>num_inference_steps</code> values.</li>
<li>Report the random seed that you're using here. You should use the
same seed all subsequent parts.</li>
</ul>

**Hints**

* Since we ask you to generate [visual anagrams](https://dangeng.github.io/visual_anagrams/) and [hybrid images](https://dangeng.github.io/factorized_diffusion/), you may want to include several text pairs prompting them beforehand.

# Part 1: Sampling Loops
In this part of the problem set, you will write your own "sampling loops"
that use the pretrained DeepFloyd denoisers. These should produce high
quality images such as the ones generated above.

You will then modify these sampling loops to solve different tasks such
as inpainting or producing optical illusions.

### Diffusion Models Primer
<!-- <div style="text-align: center;">
<img src="/hws/hw5/assets/ddpm_markov.png" alt="DDPM Markov Chain"
style="width: 30vw; display: block; margin-left: auto; margin-right: auto" />
</div>

<p>(<a href="https://arxiv.org/abs/2006.11239">Image Source</a>)</p> -->

<p class="text">
Starting with a clean image, $x_0$, we can iteratively add noise to an
image, obtaining progressively more and more noisy versions of the
image, $x_t$, until we're left with basically pure noise at timestep
$t=T$. When $t=0$, we have a clean image, and for larger $t$ more noise
is in the image.
</p>
<p class="text">
A diffusion model tries to reverse this process by denoising the image.
By giving a diffusion model a noisy $x_t$ and the timestep $t$, the
model predicts the noise in the image. With the predicted noise, we can
either completely remove the noise from the image, to obtain an estimate
of $x_0$, or we can remove just a portion of the noise, obtaining an
estimate of $x_{t-1}$, with slightly less noise.
</p>
<p class="text">
To generate images from the diffusion model (sampling), we start with
pure noise at timestep $T$ sampled from a gaussian distribution, which
we denote $x_T$. We can then predict and remove part of the noise,
giving us $x_{T-1}$. Repeating this process until we arrive at $x_0$
gives us a clean image.
</p>
<p>For the DeepFloyd models, $T = 1000$.</p>

The exact amount of noise added at each step is dictated by noise
coefficients, $\bar\alpha_t$, which were chosen by the people who
trained DeepFloyd.

### 1.1 Implementing the Forward Process

<p class="text">
A key part of diffusion is the forward process, which takes a clean
image and adds noise to it. In this part, we will write a function to
implement this. The forward process is defined by:
</p>
<p class="text">
$$q(x_t | x_0) = N(x_t ; \sqrt{\bar\alpha} x_0, (1 -
\bar\alpha_t)\mathbf{I})\tag{A.1}$$
</p>
<p class="text">
which is equivalent to computing
$$ x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon
\quad \text{where}~ \epsilon \sim N(0, 1) \tag{A.2}$$
That is, given a clean image $x_0$, we get a noisy image $ x_t $ at
timestep $t$ by sampling from a Gaussian with mean $
\sqrt{\bar\alpha_t}
x_0 $ and variance $ (1 - \bar\alpha_t) $.
Note that the forward process is not just adding noise -- we also
scale
the image.
</p>
<p class="text">
You will need to use the <code>alphas_cumprod</code> variable, which
contains the $\bar\alpha_t$ for all $t \in [0, 999]$.
Remember that $t=0$ corresponds to a clean image, and larger $t$
corresponds to more noise.
Thus, $\bar\alpha_t$ is close to 1 for small $t$, and close to 0 for
large $t$. The test image of the Campanile can be downloaded at <a
href="/hws/hw5/assets/campanile.jpg" download>here</a>, which you should then
resize to 64x64.
Run the forward process on the test image with $t \in [250, 500, 750]$
and display the results. You should get progressively more noisy
images.
</p>
<p class="text">
<b>Deliverables</b>
</p>
<ul>
<li>Implement the <code>noisy_im = forward(im, t)</code> function</li>
<li> Show the Campanile at noise level [250, 500, 750].</li>
</ul>
<p class="text">
<b>Hints</b>
</p>
<ul>
<li>The <code>torch.randn_like</code> function is helpful for
computing
$\epsilon$.</li>
<li>Use the <code>alphas_cumprod</code> variable, which contains an
array of the hyperparameters, with <code>alphas_cumprod[t]</code>
corresponding to $\bar\alpha_t$.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/campanile_resized.png" alt="Berkeley Campanile">
<p>Berkeley Campanile</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_noisy_250.png" alt="Noisy Campanile at t=250">
<p>Noisy Campanile at t=250</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_noisy_500.png" alt="Noisy Campanile at t=500">
<p>Noisy Campanile at t=500</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_noisy_750.png" alt="Noisy Campanile at t=750">
<p>Noisy Campanile at t=750</p>
</div>
</div>

### 1.2 Classical Denoising
<p class="text">
Let's try to denoise these images using classical methods.
Again, take noisy images for timesteps [250, 500, 750], but use
<b>Gaussian blur filtering</b> to try to remove the noise.
Getting good results should be quite difficult, if not impossible.
</p>

<b> Deliverables </b>
<ul>
<li>For each of the 3 noisy Campanile images from the previous part, show
your best Gaussian-denoised version side by side.</li>
</ul>

<b>Hint:</b>
<ul>
<li> <code>torchvision.transforms.functional.gaussian_blur</code> is
useful. Here is the <a
href="https://pytorch.org/vision/0.16/generated/torchvision.transforms.functional.gaussian_blur.html">documentation</a>.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.2_noisy_250.png" alt="Noisy Campanile at t=250">
<p>Noisy Campanile at t=250</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_noisy_500.png" alt="Noisy Campanile at t=500">
<p>Noisy Campanile at t=500</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_noisy_750.png" alt="Noisy Campanile at t=750">
<p>Noisy Campanile at t=750</p>
</div>
</div>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.2_gaussianblur_250.png"
alt="Gaussian Blur at t=250">
<p>Gaussian Blur Denoising at t=250</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_gaussianblur_500.png"
alt="Gaussian Blur Denoising at t=500">
<p>Gaussian Blur Denoising at t=500</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_gaussianblur_750.png"
alt="Gaussian Blur Denoising at t=750">
<p>Gaussian Blur Denoising at t=750</p>
</div>
</div>

### 1.3 One-Step Denoising

<p class="text">
Now, we'll use a pretrained diffusion model to denoise. The actual
denoiser can be found at <code>stage_1.unet</code>.
This is a UNet that has already been trained on a <i>very, very</i>
large dataset of $(x_0, x_t)$ pairs of images.
We can use it to recover Gaussian noise from the image. Then, we can
remove this noise to recover (something close to) the original image.
Note: this UNet is conditioned on the amount of Gaussian noise by
taking
timestep $t$ as additional input.
</p>
<p class="text">
Because this diffusion model was trained with text conditioning, we
also need a text prompt embedding. We provide the embedding for the
prompt <code>"a high quality photo"</code> for you to use. Later on, you can
use your own text prompts.
</p>
<p class="text">
<b>Deliverables</b>
</p>
<ul>
<li>For the 3 noisy images from 1.2 (t = [250, 500, 750]):
<ul><li>Use your <code>forward</code> function to add noise to your Campanile.</li>
<li>Estimate the noise in the new noisy image, by passing it
through
<code>stage_1.unet</code></li>
<li>Remove the noise from the noisy image to obtain an estimate of
the original image.</li>
<li>Visualize the original image, the noisy image, and the
estimate
of the original image</li></ul>

</li>
</ul>

<p class="text">
<b>Hints</b>
</p>
<ul>
<li>When removing the noise, you can't simply subtract the noise
estimate. Recall that in equation A.2 we need to scale the noise. Look
at equation A.2 to figure out how we predict $x_0$ from $x_t$ and
$t$.</li>
<li>You will probably have to wrangle tensors to the correct device
and
into the correct data types. The functions <code>.to(device)</code>
and <code>.half()</code> will be useful. The denoiser is loaded on
the device <code>cuda</code> as <code>half</code> precision (to save memory), so inputs
to the denoiser need to match them.</li>
<li>The signature for the unet is <code>stage_1.unet(im_noisy, t,
encoder_hidden_states=prompt_embeds, return_dict=False)</code>.
You
need to pass in the noisy image, the timestep, and the prompt
embeddings. The <code>return_dict</code> argument just makes the
output nicer.</li>
<li>The unet will output a tensor of shape (1, 6, 64, 64). This is
because DeepFloyd was trained to predict the noise as well as
variance
of the noise. The first 3 channels is the noise estimate, which you
will use. The second 3 channels is the variance estimate which you
may
ignore.</li>
<li>To save GPU memory, you should wrap all of your code in a
<code>with
torch.no_grad():</code> context. This tells torch not to do
automatic differentiation, and saves a considerable amount of
memory.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.2_noisy_250.png" alt="Noisy Campanile at t=250">
<p>Noisy Campanile at t=250</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_noisy_500.png" alt="Noisy Campanile at t=500">
<p>Noisy Campanile at t=500</p>
</div>
<div>
<img src="/hws/hw5/assets/1.2_noisy_750.png" alt="Noisy Campanile at t=750">
<p>Noisy Campanile at t=750</p>
</div>
</div>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.3_estimate_250.png"
alt="Estimated Campanile at t=250">
<p>One-Step Denoised Campanile at t=250</p>
</div>
<div>
<img src="/hws/hw5/assets/1.3_estimate_500.png"
alt="Denoised Campanile at t=500">
<p>One-Step Denoised Campanile at t=500</p>
</div>
<div>
<img src="/hws/hw5/assets/1.3_estimate_750.png"
alt="Denoised Campanile at t=750">
<p>One-Step Denoised Campanile at t=750</p>
</div>
</div>

### 1.4 Iterative Denoising
<p class="text">
In part 1.3, you should see that the denoising UNet does a much better
job of projecting the image onto the natural image manifold, but it
does
get worse as you add more noise. This makes sense, as the problem is
much harder with more noise!
</p>

<p class="text">
But diffusion models are designed to denoise iteratively.
In this part we will implement this.
</p>
<p class="text">
In theory, we could start with noise $x_{1000}$ at timestep $T=1000$,
denoise for one step to get an estimate of $x_{999}$, and carry on
until
we get $x_0$. But this would require running the diffusion model 1000
times, which is quite slow (and costs $$$).
</p>
<p class="text">
It turns out, we can actually speed things up by skipping steps. The
rationale for why this is possible is due to a connection with
differential equations. It's a tad complicated, and not within scope
for
this course, but if you're interested you can check out <a
href="https://yang-song.net/blog/2021/score/">this excellent
article</a>.
</p>
<p class="text">
To skip steps we can create a new list of timesteps that we'll call
<code>strided_timesteps</code>, which does just this.
<code>strided_timesteps[0]</code> will correspond to the the largest $t$
(and thus the noisiest image) and
<code>strided_timesteps[-1]</code> will correspond to $t = 0$ (and thus a clean image).
One
simple way of constructing this list is by introducing a regular
stride
step (e.g. stride of 30 works well).
</p>

<p class="text">
On the <code>i</code>th denoising step we are at $ t = $
<code>strided_timesteps[i]</code>, and want to get to $ t' =$
<code>strided_timesteps[i+1]</code> (from more noisy to less noisy).
To
actually do this, we have the following formula:
</p>

$$ x_{t'} = \frac{\sqrt{\bar\alpha_{t'}}\beta_t}{1 - \bar\alpha_t} x_0 +
\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t'})}{1 - \bar\alpha_t} x_t +
v_\sigma\tag{A.3}$$

<p class="text">
where:
</p>
<ul>
<li>$x_t$ is your image at timestep $t$</li>
<li>$x_{t'}$ is your noisy image at timestep $t'$ where $t' < t$ (less
noisy)</li>
<li>$\bar\alpha_t$ is defined by <code>alphas_cumprod</code>, as
explained above.</li>
<li>$\alpha_t = \frac{\bar\alpha_t}{\bar\alpha_{t'}}$</li>
<li>$\beta_t = 1 - \alpha_t$</li>
<li>$x_0$ is our current estimate of the clean image using one-step denoising</li>
</ul>

<p class="text"></p>
The $v_\sigma$ is random noise, which in the case of DeepFloyd is also
predicted.
The process to compute this is not very important, so we supply a
function, <code>add_variance</code>, to do this for you.
</p>

You can think of this as a linear interpolation between the signal and
noise:

<div style="text-align: center;">
<img src="/hws/hw5/assets/interpolation.png" alt="Interpolation Example"
style="width: 20vw; display: block; margin-left: auto; margin-right: auto" />
<p class="text">Interpolation</p>
</div>

See equations 6 and 7 of the <a
href="https://arxiv.org/pdf/2006.11239">DDPM paper</a> for more
information (Denoising Diffusion Probabilistic Models, the paper 
that introduces the diffusion model, which comes from Cal!). 
Be careful about bars above the alpha! Some have them and some do not.

<p>
First, create the list <code>strided_timesteps</code>. You should
start at timestep 990, and take step sizes of size 30 until you arrive at
0. After completing the problem set, feel free to try different
"schedules" of timesteps.
</p>

<p>
Also implement the function <code>iterative_denoise(im_noisy,
i_start)</code>, which takes a noisy image <code>image</code>, as well
as a starting index <code>i_start</code>. The function should denoise
an image starting at timestep <code>timestep[i_start]</code>, applying
the above formula to obtain an image at timestep <code>t' =
timestep[i_start + 1]</code>, and repeat iteratively until we arrive at
a clean image.
</p>

<p>
Add noise to the test image <code>im</code> to timestep
<code>timestep[10]</code> and display this image. Then run the
<code>iterative_denoise</code> function on the noisy image, with
<code>i_start = 10</code>, to obtain a clean image and display it. Please
display every 5th image of the denoising loop. Compare this to the
"one-step" denoising method from the previous section, and to gaussian
blurring.
</p>

<p class="text">
<b>Deliverables</b>
</p>
Using <code>i_start = 10</code>:
<ul>
<li>Create <code>strided_timesteps</code>: a list of monotonically
decreasing timesteps, starting at 990, with a stride of 30, eventually
reaching 0. Also initialize the timesteps using the function
<code>stage_1.scheduler.set_timesteps(timesteps=strided_timesteps)</code></li>
<li>Complete the <code>iterative_denoise</code> function</li>
<li>Show the noisy Campanile every 5th loop of denoising (it should
gradually
become less noisy)</li>
<li>Show the final predicted clean image, using iterative denoising</li>
<li>Show the predicted clean image using only a single denoising step,
as
was done in the previous part. This should look much worse.</li>
<li>Show the predicted clean image using gaussian blurring, as was done
in
part 1.2.</li>
</ul>

<b>Hints</b>
<ul>
<li>Remember, the unet will output a tensor of shape (1, 6, 64, 64).
This is because DeepFloyd was trained to predict the noise as well as
variance of the noise. The first 3 channels is the noise estimate,
which you will use here.
The second 3 channels is the variance estimate which you will pass to
the <code>add_variance</code> function</li>
<li>Read the documentation for the <code>add_variance</code> function to
figure out how to use it to add the $v_\sigma$ to the image.</li>
<li>Depending on if your final images are torch tensors or numpy arrays,
you may need to modify the `show_images` call a bit.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.4_noisy_90.png" alt="Noisy Campanile at t=90">
<p>Noisy Campanile at t=90</p>
</div>
<div>
<img src="/hws/hw5/assets/1.4_noisy_240.png" alt="Noisy Campanile at t=240">
<p>Noisy Campanile at t=240</p>
</div>
<div>
<img src="/hws/hw5/assets/1.4_noisy_390.png" alt="Noisy Campanile at t=390">
<p>Noisy Campanile at t=390</p>
</div>
<div>
<img src="/hws/hw5/assets/1.4_noisy_540.png" alt="Noisy Campanile at t=540">
<p>Noisy Campanile at t=540</p>
</div>
<div>
<img src="/hws/hw5/assets/1.4_noisy_690.png" alt="Noisy Campanile at t=690">
<p>Noisy Campanile at t=690</p>
</div>
</div>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/campanile_resized.png"
alt="Original Campanile">
<p>Original</p>
</div>
<div>
<img src="/hws/hw5/assets/1.4_clean_iterative.png"
alt="Iteratively Denoised Campanile">
<p>Iteratively Denoised Campanile</p>
</div>
<div>
<img src="/hws/hw5/assets/1.4_clean_onestep.png"
alt="One-Step Denoised Campanile">
<p>One-Step Denoised Campanile</p>
</div>
<div>
<img src="/hws/hw5/assets/1.4_gaussianblur.png"
alt="Gaussian Blurred Campanile">
<p>Gaussian Blurred Campanile</p>
</div>
</div>

### 1.5 Diffusion Model Sampling
<p class="text">
In part 1.4, we use the diffusion model to denoise an image. Another
thing
we can do with the <code>iterative_denoise</code> function is to
generate
images from scratch. We can do this by setting <code>i_start = 0</code>
and passing <code>im_noisy</code> as random noise. This effectively denoises pure noise.
Please
do this, and show 5 results of the prompt<code>"a high quality photo"</code>.
</p>
<p class="text">
<b>Deliverables</b>
<ul>
<li>Show 5 sampled images.</li>
</ul>
</p>
<p class="text">
<b>Hints</b>
</p>
<ul>
<li>Use <code>torch.randn</code> to make the noise.</li>
<li>Make sure you move the tensor to the correct device and correct data
type by calling <code>.half()</code> and
<code>.to(device)</code>.</li>
<li>The quality of the images will not be spectacular, but should be
reasonable images.
We will fix this in the next section with CFG.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.5_1.png" alt="Noisy Campanile at t=90">
<p>Sample 1</p>
</div>
<div>
<img src="/hws/hw5/assets/1.5_2.png" alt="Sample 2">
<p>Sample 2</p>
</div>
<div>
<img src="/hws/hw5/assets/1.5_3.png" alt="Sample 3">
<p>Sample 3</p>
</div>
<div>
<img src="/hws/hw5/assets/1.5_4.png" alt="Sample 4">
<p>Sample 4</p>
</div>
<div>
<img src="/hws/hw5/assets/1.5_5.png" alt="Sample 5">
<p>Sample 5</p>
</div>
</div>

### 1.6 Classifier-Free Guidance (CFG)
<p class="text">
You may have noticed that the generated images in the prior section are
not very good, and some are completely non-sensical.
In order to greatly improve image quality (at the expense of image
diversity), we can use a technicque called <a
href="https://arxiv.org/abs/2207.12598">Classifier-Free Guidance</a>.
</p>
<p class="text">
In CFG, we compute both a conditional and an unconditional noise
estimate. We denote these $\epsilon_c$ and $\epsilon_u$.
Then, we let our new noise estimate be: $$\epsilon = \epsilon_u + \gamma
(\epsilon_c - \epsilon_u) \tag{A.4}$$
where $\gamma$ controls the strength of CFG. Notice that for $\gamma=0$,
we get an unconditional noise estimate, and for $\gamma=1$ we get the
conditional noise estimate.
The magic happens when $\gamma > 1$. In this case, we get much higher
quality images. Why this happens is still up to vigorous debate.
For more information on CFG, you can check out <a
href="https://sander.ai/2022/05/26/guidance.html">this blog post</a>.
</p>
<p class="text">
Please implement the <code>iterative_denoise_cfg</code> function,
identical to the <code>iterative_denoise</code> function but using
classifier-free guidance.
To get an unconditional noise estimate, we can just pass an empty prompt
embedding to the diffusion model (the model was trained to predict an
unconditional noise estimate when given an empty text prompt).
</p>
<p class="text">
<b>Disclaimer</b>
Disclaimer
Before, we used <code>"a high quality photo"</code> as a "null"
condition.
Now, we will use the actual <code>""</code> null prompt for
unconditional
guidance for CFG. In the later part, you should always use
<code>""</code>
null prompt for unconditional guidance.
</p>
<p class="text">
<b>Deliverables</b>
<ul>
<li>Implement the <code>iterative_denoise_cfg</code> function</li>
<li>Show 5 images of <code>"a high quality photo"</code> with a CFG
scale of $\gamma=7$. Now this prompt becomes a <b>condition</b> (but fairly weak)
to generate <b>conditional</b> noise! You will use your customized prompts as
stronger conditions in part 1.7 - part 1.9.</li>
</ul>
</p>
<p class="text">
<b>Hints</b>
</p>
<ul>
<li>You will need to run the UNet twice, once for the conditional prompt
embedding, and once for the unconditional</li>
<li>The UNet will predict both a conditional and an unconditional
variance. Just use the conditional variance with the
<code>add_variance</code> function.</li>
<li>The resulting images should be much better than those in the prior
section.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.6_1.png" alt="Noisy Campanile at t=90">
<p>Sample 1 with CFG</p>
</div>
<div>
<img src="/hws/hw5/assets/1.6_2.png" alt="Sample 2">
<p>Sample 2 with CFG</p>
</div>
<div>
<img src="/hws/hw5/assets/1.6_3.png" alt="Sample 3">
<p>Sample 3 with CFG</p>
</div>
<div>
<img src="/hws/hw5/assets/1.6_4.png" alt="Sample 4">
<p>Sample 4 with CFG</p>
</div>
<div>
<img src="/hws/hw5/assets/1.6_5.png" alt="Sample 5">
<p>Sample 5 with CFG</p>
</div>
</div>

### 1.7 Image-to-image Translation
<b style="color: #CC0000;">Note: You should use CFG from this point forward.</b> 
<p class="text">
In part 1.4, we take a real image, add noise to it, and then denoise.
This
effectively allows us to make edits to existing images. The more noise
we
add, the larger the edit will be. This works because in order to denoise
an image, the diffusion model must to some extent "hallucinate" new
things
-- the model has to be "creative." Another way to think about it is that
the denoising process "forces" a noisy image back onto the manifold of
natural images.
</p>
<p class="text">
Here, we're going to take the original Campanile image, noise it a little,
and
force it back onto the image manifold without any conditioning.
Effectively, we're going to get an image that is similar to the Campanile
(with a low-enough noise level). This follows the <a
href="https://sde-image-editing.github.io/">SDEdit</a> algorithm.
</p>
<p>To start, please run the forward process to get a noisy Campanile, and
then run the <code>iterative_denoise_cfg</code> function using a
starting
index of [1, 3, 5, 7, 10, 20] steps and show the results, labeled with
the
starting index. You should see a series of "edits" to the original
image,
gradually matching the original image closer and closer.</p>
<b> Deliverables </b>
<ul>
<li>Edits of the Campanile image, using the given prompt at noise levels [1,
3, 5, 7, 10, 20] with the conditional text prompt
<code>"a high quality photo"</code></li>
<li>Edits of 2 of your own test images, using the same procedure.</li>
</ul>
<p class="text">
Hints
<ul>
<li>You should have a range of images, gradually looking more like the
original image</li>
</ul>
</p>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.7_start_1.png" alt="Sample 5">
<p>SDEdit with <code>i_start=1</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_start_3.png" alt="Sample 4">
<p>SDEdit with <code>i_start=3</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_start_5.png" alt="Sample 3">
<p>SDEdit with <code>i_start=5</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_start_7.png" alt="Sample 2">
<p>SDEdit with <code>i_start=7</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_start_10.png" alt="Noisy Campanile at t=90">
<p>SDEdit with <code>i_start=10</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_start_20.png" alt="Noisy Campanile at t=90">
<p>SDEdit with <code>i_start=20</code></p>
</div>
<div>
<img src="/hws/hw5/assets/campanile_resized.png" alt="Original Campanile">
<p>Campanile</p>
</div>
</div>

#### 1.7.1 Editing Hand-Drawn and Web Images
<p class="text">
This procedure works particularly well if we start with a nonrealistic
image (e.g. painting, a sketch, some scribbles) and project it onto the
natural image manifold.
</p>
<p class="text">
Please experiment by starting with hand-drawn or other non-realistic
images and see how you can get them onto the natural image manifold in
fun ways.
</p>
<p class="text">
We provide you with 2 ways to provide inputs to the model:
</p>
<ol>
<li>Download images from the web</li>
<li>Draw your own images</li>
</ol>
<p class="text">
Please find an image from the internet and apply edits exactly as above.
And also draw your own images, and apply edits exactly as above. Feel
free to copy the prior cell here. For drawing inspiration, you can check
out the examples on <a href="https://sde-image-editing.github.io/">this
project page</a>.
</p>
<p>
<b>Deliverables</b>
<ul>
<li>1 image from the web of your choice, edited using the above method
for noise levels [1, 3, 5, 7, 10, 20] (and whatever additional noise
levels you want)</li>
<li>2 hand drawn images, edited using the above method for noise
levels [1, 3, 5, 7, 10, 20] (and whatever additional noise levels
you want)</li>
</ul>

<b>Hints</b>
<ul>
<li>We provide you with preprocessing code to convert web images to the format expected by DeepFloyd</li>
<li>Unfortunately, the drawing interface is hardcoded to be 300x600
pixels, but we need a square image. The code will center crop, so
just draw in the middle of the canvas.</li>
</ul>
</p>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.7_avo_1.png" alt="Avocado at noise level 1">
<p>Avocado at <code>i_start=1</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_avo_3.png" alt="Avocado at noise level 3">
<p>Avocado at <code>i_start=3</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_avo_5.png" alt="Avocado at noise level 5">
<p>Avocado at <code>i_start=5</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_avo_7.png" alt="Avocado at noise level 7">
<p>Avocado at <code>i_start=7</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_avo_10.png" alt="Avocado at noise level 10">
<p>Avocado at <code>i_start=10</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7_avo_20.png" alt="Avocado at noise level 20">
<p>Avocado at <code>i_start=20</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.1_avo_original.png" alt="Original Avocado">
<p>Avocado</p>
</div>
</div>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.7.1_house_1.png" alt="House at noise level 1">
<p>House at <code>i_start=1</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.1_house_3.png" alt="House at noise level 3">
<p>House at <code>i_start=3</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.1_house_5.png" alt="House at noise level 5">
<p>House at <code>i_start=5</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.1_house_7.png" alt="House at noise level 7">
<p>House at <code>i_start=7</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.1_house_10.png" alt="House at noise level 10">
<p>House at <code>i_start=10</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.1_house_20.png" alt="House at noise level 20">
<p>House at <code>i_start=20</code></p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.1_house_sketch_resized.png"
alt="House at noise level 20">
<p>Original House Sketch</p>
</div>
</div>

### 1.7.2 Inpainting
<p class="text">
We can use the same procedure to implement inpainting (following the <a
href="https://arxiv.org/abs/2201.09865">RePaint</a> paper). That is,
given an image $x_{orig}$, and a binary mask $\bf m$, we can create a
new image that has the same content where $\bf m$ is 0, but new content
wherever $\bf m$ is 1.
</p>
<p class="text">
To do this, we can run the diffusion denoising loop. But at every step,
after obtaining $x_t$, we "force" $x_t$ to have the same pixels as
$x_{orig}$ where $\bf m$ is 0, i.e.:
</p>
<p class="text">
$$ x_t \leftarrow \textbf{m} x_t + (1 - \textbf{m})
\text{forward}(x_{orig}, t) \tag{A.5}$$
</p>
<p class="text">
Essentially, we leave everything inside the edit mask alone, but we
replace everything outside the edit mask with our original image -- with
the correct amount of noise added for timestep $t$.
</p>
<p class="text">
Please implement this below, and edit the picture to inpaint the top of
the Campanile.
</p>
<p class="text">
<b>Deliverables</b>

<ul>
<li>A properly implemented <code>inpaint</code> function</li>
<li>The Campanile inpainted (feel free to use your own mask)</li>
<li>2 of your own images edited (come up with your own mask)
<ul>
<li>look at the results from <a
href="http://graphics.cs.cmu.edu/projects/scene-completion/">this
paper</a> for inspiration</li>
</ul>
</li>
</ul>
</p>
<p class="text">
<b>Hints</b>
</p>
<ul>
<li>Reuse the <code>forward</code> function you implemented earlier to
implement inpainting</li>
<li>Because we are using the diffusion model for tasks it was not
trained for, you may have to run the sampling process a few times
before you get a nice result.</li>
<li>You can copy and paste your iterative_denoise_cfg function. To get
inpainting to work should only require (roughly) 1-2 additional lines
and a few small changes.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/campanile_resized.png" alt="Resized Campanile">
<p>Campanile</p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.2_mask.png" alt="Mask">
<p>Mask</p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.2_to_replace.png" alt="To Replace">
<p>Hole to Fill</p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.2_campanile_inpainted.png"
alt="Campanile Inpainted">
<p>Campanile Inpainted</p>
</div>
</div>

### 1.7.3 Text-Conditional Image-to-image Translation

<p>Now, we will do the same thing as SDEdit, but guide the
projection with a text prompt. This is no longer pure
"projection to the natural image manifold" but also adds control using
language. This is simply a matter of changing the prompt from
<code>"a high quality photo"</code> to any of your prompt!</p>

<b>Deliverables</b>
<ul>
<li>Edits of the Campanile, using the given prompt at noise levels [1,
3, 5, 7, 10, 20]</li>
<li>Edits of 2 of your own test images, using the same procedure</li>
</ul>

<b></b>Hints</b>
<ul>
<li>The images should gradually look more like original image, but also
look like the text prompt.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/1.7.3_rocket_1.png" alt="Rocket Ship at noise level 1">
<p>Rocket Ship at noise level 1</p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.3_rocket_3.png" alt="Rocket Ship at noise level 3">
<p>Rocket Ship at noise level 3</p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.3_rocket_5.png" alt="Rocket Ship at noise level 5">
<p>Rocket Ship at noise level 5</p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.3_rocket_7.png" alt="Rocket Ship at noise level 7">
<p>Rocket Ship at noise level 7</p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.3_rocket_10.png"
alt="Rocket Ship at noise level 10">
<p>Rocket Ship at noise level 10</p>
</div>
<div>
<img src="/hws/hw5/assets/1.7.3_rocket_20.png"
alt="Rocket Ship at noise level 20">
<p>Rocket Ship at noise level 20</p>
</div>
<div>
<img src="/hws/hw5/assets/campanile_resized.png"
alt="Rocket Ship at noise level 20">
<p>Campanile</p>
</div>
</div>

### 1.8 Visual Anagrams
<p class="text">
In this part, we are finally ready to implement <a
href="https://dangeng.github.io/visual_anagrams/">Visual
Anagrams</a> and create optical illusions with diffusion models. In
this part, we will create an image that looks like <code>"an oil
painting of people around a campfire"</code>, but when flipped upside
down will reveal <code>"an oil painting of an old man"</code>.
</p>
<p class="text">
To do this, we will denoise an image $x_t$ at step $t$ normally with the
prompt
$p_1$, to obtain noise estimate
$\epsilon_1$. But at the same time, we will flip $x_t$ upside down, and
denoise with the prompt
$p_2$, to get noise estimate $\epsilon_2$. We can flip $\epsilon_2$ back, and average the two noise estimates. We can then perform a
reverse/denoising diffusion step with the averaged noise estimate.
</p>
<p class="text">
The full algorithm will be:
</p>
<p class="text">
$$ \epsilon_1 = \text{CFG of UNet}(x_t, t, p_1) $$
</p>
<p class="text">
$$ \epsilon_2 = \text{flip}(\text{CFG of UNet}(\text{flip}(x_t), t, p_2)) $$
</p>
<p class="text">
$$ \epsilon = (\epsilon_1 + \epsilon_2) / 2 $$
</p>
<p class="text">
where UNet is the diffusion model UNet from before, $\text{flip}(\cdot)$
is a function that flips the image, and $p_1$ and $p_2$ are two different
text prompt embeddings. And our final noise estimate is $\epsilon$. Please
implement the above algorithm and show example of an illusion.
</p>

<b>Deliverables</b>
<ul>
<li>Correctly implemented <code>visual_anagrams</code> function</li>
<li>2 illusions of your choice that change appearance when you flip
it upside down (feel free to take inspirations from this <a href="https://dangeng.github.io/visual_anagrams/">page</a>).</li>
</ul>

<b>Hints</b>
<ul>
<li>You may have to run multiple times to get a really good result for
the same reasons as above.</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/old_man.png" alt="Old Man">
<p>An Oil Painting of an Old Man</p>
</div>
<div>
<img src="/hws/hw5/assets/old_man_flipped.png" alt="Old Man Flipped">
<p>An Oil Painting of People around a Campfire</p>
</div>
</div>

### 1.9 Hybrid Images
<p class="text">
In this part we'll implement <a
href="https://arxiv.org/abs/2404.11615">Factorized
Diffusion</a> and create hybrid images just like in project 2.
</p>
<p class="text">
In order to create hybrid images with a diffusion model we can use a
similar technique as above. We will create a composite noise estimate
$\epsilon$, by estimating the noise with two different text prompts, and
then combining low frequencies from one noise estimate with high
frequencies of the other. The algorithm is:
</p>
<p class="text">
$ \epsilon_1 = \text{CFG of UNet}(x_t, t, p_1) $
</p>
<p class="text">
$ \epsilon_2 = \text{CFG of UNet}(x_t, t, p_2) $
</p>
<p class="text">
$ \epsilon = f_\text{lowpass}(\epsilon_1) + f_\text{highpass}(\epsilon_2)$
</p>
<p class="text">
where UNet is the diffusion model UNet, $f_\text{lowpass}$ is a low pass
function, $f_\text{highpass}$ is a high pass function, and $p_1$ and $p_2$
are two different text prompt embeddings. Our final noise estimate is
$\epsilon$. Please show an example of a hybrid image using this technique
(you may have to run multiple times to get a really good result for the
same reasons as above). We recommend that you use a gaussian blur of
kernel size 33 and sigma 2.
</p>

<b>Deliverables</b>
<ul>
<li>Correctly implemented <code>make_hybrids</code> function</li>
<li>2 hybrid images of your choosing (feel free to take inspirations from this <a href="https://dangeng.github.io/factorized_diffusion/">page</a>).</li>
</ul>

<b>Hints</b>
<ul>
<li>use <code>torchvision.transforms.functional.gaussian_blur</code> </li>
<li>You may have to run multiple times to get a really good result for the
same reasons as above</li>
</ul>

<div class="image-container">
<div>
<img src="/hws/hw5/assets/skull2.png"
alt="Hybrid image of a skull and a waterfall">
<p>Hybrid image of a skull and a waterfall</p>
</div>
</div>

#  Part 2: Bells & Whistles 
<b>Required for CS280A students only:</b>
<ul>
<li><b>More visual anagrams!</b> Visual anagrams in part 1.8 are created by flipping images
upside down. However, there are much more transformations that also create visual
anagrams! Refer to this <a href="https://arxiv.org/pdf/2311.17919">paper</a> and select two more
transformations to create visual anagrams.</li>
<li><b>Design a course logo</b>! Doing text-conditioned image-to-image translation on UCB's
logo or your drawing may be a good idea.</li>
</ul>
<b>Optional for all students:</b>
<ul>
<li><b>Your own ideas</b>: Be creative!</li>
</ul>

# Deliverable Checklist
<ul>
<li>Make sure that your website and submission include <b>all the deliverables</b> in each section above.</li>
<li>Submit your <b>PDF</b> and <b>code</b> to corresponding assignments on Gradescope.</li>
<li><b>The Google Form is not required for Part A</b>; you only need to complete the Google Form after both parts are finished.</li>
</ul>

<script>
window.addEventListener('load', function() {
const zoomEls = document.querySelectorAll('.zoom-animation');
const rotateEls = document.querySelectorAll('.rotating-image');
const dissolveOriginals = document.querySelectorAll('.dissolve-image.original');
const dissolveEditeds = document.querySelectorAll('.dissolve-image.edited');
        
zoomEls.forEach(el => el.classList.add('active'));
rotateEls.forEach(el => el.classList.add('active'));
dissolveOriginals.forEach(el => el.classList.add('active'));
dissolveEditeds.forEach(el => el.classList.add('active'));
        
setTimeout(() => {
zoomEls.forEach(el => el.classList.remove('active'));
rotateEls.forEach(el => el.classList.remove('active'));
dissolveOriginals.forEach(el => el.classList.remove('active'));
dissolveEditeds.forEach(el => el.classList.remove('active'));
}, 2000);
});
</script>

<br><br>
<hr>
<br><br>

<div>

<video id="video5" width="640" height="320" muted
style="display: block; margin-left: auto; margin-right: auto;"
onmouseover="handleMouseOver(this)"
onmouseout="handleMouseOut(this)">
<source type="video/mp4" src="/hws/hw5/assets/new_c_20_fm.mp4" />
</video>
</div>

# HW5 Part B: Flow Matching from Scratch!
<a href="../../">COMS4732: Computer Vision 2</a>
### For this part, you need to submit your code and website PDF, and also your web url to class gallery via this <a href="https://forms.gle/gLQhNCyBUaCACt7W6">Google Form</a>.
<h2 style="text-align: center">
<b style="color: red;">Due: TBD</b>
</h2>
<h4 style="text-align: center">
<b>We recommend using GPUs from <a
href="https://colab.research.google.com/">Colab</a> to finish this
project!</b>
</h4>

## Overview
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

# Part 1: Training a Single-Step Denoising UNet
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
few downsampling and upsampling blocks with skip connections.
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

<br\>

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
can generate $z$ from $x$ using the the following noising process:
$$
z = x + \sigma \epsilon,\quad \text{where }\epsilon \sim N(0, I). \tag{B.2}
$$

Visualize the different noising processes over $\sigma = [0.0, 0.2, 0.4,
0.5, 0.6, 0.8, 1.0]$, assuming normalized $x \in [0, 1]$.

You should see noisier images as $\sigma$ increases.

### Deliverable
<ul>
<li>A visualization of the noising process using $\sigma = [0.0,
0.2, 0.4, 0.5, 0.6, 0.8, 1.0]$.</li>
</ul>
<!-- <div style="text-align: center;">
<img src="/hws/hw5/assets/varying_sigma.png" alt="Varying Sigmas" height="600"
style="display: block; margin-left: auto; margin-right: auto" />
<p class="text">Figure 3: Varying levels of noise on MNIST digits</p>
</div> -->

## 1.2.1 Training
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

<p class="text"></p>
You should visualize denoised results on the test set at the end of
training. Display sample results after the 1st and 5th epoch.
</p>
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

### Deliverables
<ul>
<li>A training loss curve plot every few iterations during the whole
training process of $\sigma = 0.5$.</li>
<li>Sample results on the test set with noise level 0.5 after the first and the 5-th epoch
(staff solution takes ~3 minutes for 5 epochs on a Colab T4 GPU).</li>
</ul>

## 1.2.2 Out-of-Distribution Testing

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
### Deliverables
<ul>
<li>Sample results on the test set with out-of-distribution noise levels
after the model is trained. Keep the same image and
vary $\sigma = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]$.</li>
</ul>

## 1.2.3 Denoising Pure Noise
<p>To make denoising a generative task, we'd like to be able to denoise pure, random Gaussian noise. We can think of this as starting with a blank canvas $z = \epsilon$ where $\epsilon \sim N(0, I)$ and denoising it to get a clean image $x$.</p>

<p>Repeat the same training process as in part 1.2.1, but input pure noise $\epsilon \sim N(0, I)$ and denoise it for 5 epochs. Display your results after 1 and 5 epochs.</p>

<p>Sample from the denoiser that was trained to denoise pure noise. What patterns do you observe in the generated outputs? What relationship, if any, do these outputs have with the training images (e.g., digits 0–9)? Why might this be happening?</p>

### Deliverables
<ul>
<li>A training loss curve plot every few iterations during the whole
training process that denoises pure noise.</li>
<li>Sample results on pure noise after the first and the 5-th epoch.</li>
<li>A brief description of the patterns observed in the generated outputs and explanations for why they may exist.</li>
</ul>

<b>Hint</b>
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

# Part 2: Training a Flow Matching Model
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

# the t passed in here should be normalized to be in the range [0, 1]
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

### Deliverable
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
        
<video id="video1" width="100%" muted loop playbackRate="0.75"
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/t_only_e1_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 1</p>
</div>
<div style="width: 100%; max-width: 600px;">
        
<video id="video2" width="100%" muted loop playbackRate="0.75"
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

### Deliverables
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
    
### Deliverable
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
          
<video id="video1" width="100%" muted loop playbackRate="0.75"
style="display: block; margin-left: 0;">
<source type="video/mp4" src="/hws/hw5/assets/new_c_1_fm.mp4" />
</video>
<p style="text-align: left;">Epoch 1</p>
</div>
<div style="width: 100%; max-width: 600px;">
          
<video id="video2" width="100%" muted loop playbackRate="0.75"
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


### Deliverables
<ul>
<li>Sampling results from the class-conditioned UNet for 1, 5, and 10 epochs. Class-conditioning lets us converge faster, hence why we only train for 10 epochs. Generate 4 instances of each digit as shown above.
</li>
<li><b>Can we get rid of the annoying learning rate scheduler?</b> Simplicity is the best. Please try to maintain the same performance after removing the exponential 
learning rate scheduler. Show your visualization after training without the scheduler and provide a description of what you did to compensate for the loss of the scheduler.</li>
</ul>

#  Part 3: Bells & Whistles (Optional)
<!-- <b>Required for CS280A students only:</b> -->
<ul>
<li><b>A better time-conditioned only UNet: </b> Our time-conditioning only UNet in part 2.3 is actually far from perfect. Its result is way worse than the UNet conditioned by both time and class.
We can definitively make it better! Show a better visualization image for the time-conditioning only network. Possible approaches include extending the training schedule or making the architecture more expressive. </li>
</ul>
<ul>
<li><b>Your own ideas</b>: Be creative! This UNet can generate images more than digits! You can try it on <a href="http://ufldl.stanford.edu/housenumbers/">SVHN</a> (still digits, but more fancy!), <a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST</a> (not digits, but still grayscale!), or <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR10</a>!</li>
</ul>

# Deliverable Checklist
<ul>
<li>Make sure that your website and submission include <b>all the deliverables</b> in each section above.</li>
<li>Submit your <b>PDF</b> and <b>code</b> to corresponding assignments on Gradescope.</li>
<li>
<b>The Google Form is required for Part B.</b> Once you have finished both parts A and B, submit the link to your webpage (containing both parts) using this 
<a href="https://forms.gle/gLQhNCyBUaCACt7W6">Google Form</a>.
</li>
</ul>

<script>
window.addEventListener('load', function() {
// Handle video autoplay and playback speed for all videos
var videos = document.querySelectorAll('video');
videos.forEach(function(video) {
video.playbackRate = 1;
video.loop = false;
video.play();

// Add hover behavior to each video
video.addEventListener('mouseover', function() {
// Pause normal playback
video.pause();
        
// Play in reverse by decreasing currentTime
const rewindInterval = setInterval(() => {
if (video.currentTime <= 0) {
clearInterval(rewindInterval);
} else {
video.currentTime -= 0.05; // Slow rewind speed
}
}, 40); // Smooth interval

// Store the interval ID so we can clear it on mouseout
video.rewindInterval = rewindInterval;
});

video.addEventListener('mouseout', function() {
// Clear the rewind interval if it exists
if (video.rewindInterval) {
clearInterval(video.rewindInterval);
video.rewindInterval = null;
}
// Play forward normally
video.play();
});
});
});

// These handlers are no longer needed since we're handling everything in the load event
function handleMouseOver(video) {
// Empty or can be removed
}

function handleMouseOut(video) {
// Empty or can be removed
}
</script>
### Acknowledgements
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
    