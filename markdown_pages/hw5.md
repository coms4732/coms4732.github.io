---
title: Homework 5
layout: default
permalink: /hw5/
has_children: true
nav_order: 6
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

/* Hero images at the top - fit all 4 on one line at max size */
.hero-images {
gap: 10px;
}

.hero-images > div {
flex: 1 1 22%;
width: 22%;
min-width: 0;
max-width: 280px;
padding: 5px;
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

/* Hero images at the top - fit all 4 on one line at max size */
.hero-images {
gap: 10px;
}

.hero-images > div {
flex: 1 1 22%;
width: 22%;
min-width: 0;
max-width: 280px;
padding: 5px;
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

/* Hero images at the top - fit all 4 on one line at max size */
.hero-images {
gap: 10px;
}

.hero-images > div {
flex: 1 1 22%;
width: 22%;
min-width: 0;
max-width: 280px;
padding: 5px;
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

<div class="image-container hero-images">
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

# Homework 5: Diffusion Models and Flow Matching

This assignment is split into two parts:

- **[Part A: The Power of Diffusion Models]({{ '/hw5/part-a/' | relative_url }})** - Play with pre-trained diffusion models, implement sampling loops, and create optical illusions
- **[Part B: Flow Matching from Scratch]({{ '/hw5/part-b/' | relative_url }})** - Train your own flow matching model on MNIST

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
