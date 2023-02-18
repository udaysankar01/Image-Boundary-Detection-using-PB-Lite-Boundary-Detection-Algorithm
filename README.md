<a name="readme-top"></a>

# Image Boundary Detection using PB-Lite Boundary Detection Algorithm

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>   
      <li><a href="#installation">Installation</a></li>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

In this project, I implement a simplified version of [the new PB (Probability of Boundary) Boundary Detection algorithm](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf) which produce much better boundary detection results compared to the conventional Sobel and Canny algorithms. Specifically, PB algorithm is capable of ruling out a lot of false positives that these classical techniques produce in textured regions. This simplified method can be called <b>pb-lite boundary</b> detection method. The piepline of this method is shown below.

![image](https://user-images.githubusercontent.com/22807879/219878983-bf382879-46a6-4069-b3a4-0ed1f0feadb5.png)

The first step in pb-lite method is the construction of the filter banks. The purpose of these filter banks is to obtain the texture information from the image by filtering the image and clustering the filter responses. In this project, I have implemented three main filter banks. They are:

<ol>
  <li>Oriented Derivative of Gaussian (DoG) Filter Bank</li>
  https://github.com/udaysankar01/Image-Boundary-Detection-using-PB-Lite-Boundary-Detection-Algorithm/blob/main/Filter%20Banks/DoG.png
  <li>Leung Malik (LM) Filter Bank</li>
  <li>Gabor Filter Bank</li>
</ol>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

To run the pb-lite edge detection algorithm:

`python Code/Wrapper.py`

All the results will be stored in the folder shown below:

`.Code/Results/`


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Uday Sankar - usankar@wpi.edu

Project Link: [https://github.com/udaysankar01/Image-Boundary-Detection-using-PB-Lite-Boundary-Detection-Algorithm](https://github.com/udaysankar01/Image-Boundary-Detection-using-PB-Lite-Boundary-Detection-Algorithm)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
