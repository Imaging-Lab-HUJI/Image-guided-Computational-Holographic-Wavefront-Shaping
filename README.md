# Image-guided Computational Holographic Wavefront Shaping

### Introduction

This repository contains the Python code implementation for the following publication:

> Omri Haim*, Jeremy Boger-Lombard*, and Ori Katz. Image-guided computational holographic wavefront shaping. _Nature Photonics_, 2024. [![DOI](https://img.shields.io/badge/DOI-10.1038/s41566--024--01544--6-blue)](https://doi.org/10.1038/s41566-024-01544-6)

**Abstract**: Optical imaging through scattering media is important in a variety of fields ranging from microscopy to autonomous vehicles. While advanced wavefront shaping techniques have offered several breakthroughs in the past decade, current techniques still require a known guide star and a high-resolution spatial light modulator (SLM) or a very large number of measurements and are limited in their correction field-of-view. Here, we introduce a guide star-free, non-invasive approach that can correct more than 190,000 scattered modes using only 25 incoherently-compounded holographically measured, scattered light fields, obtained under unknown random illuminations. This is achieved by computationally emulating an image-guided wavefront shaping experiment, where several ’virtual SLMs’ are simultaneously optimized to maximize the reconstructed image quality. Our method shifts the burden from the physical hardware to a digital, naturally-parallelizable computational optimization, leveraging state-of-the-art automatic differentiation tools. We demonstrate the flexibility and generality of this framework by applying it to imaging through various complex samples and imaging modalities, including epi-illumination, anisoplanatic multi-conjugate correction of highly scattering layers, lensless endoscopy in multicore fibers, and acousto-optic tomography. The presented approach offers high versatility, effectiveness, and generality for fast, non-invasive imaging in diverse applications.

### Prerequisites
- Python 3.8 or later
- PyTorch 1.12.0 or later
- numpy
- matplotlib (optional)
- xmltodict

### Project Structure
- `main.py`: The main script that runs the optimization.
- `data.py`: Contains the `load_data` function to load the measurement data from the dataset.
- `optimization.py`: Contains functions for the gradient ascent optimization.
- `utility.py`: Contains utility functions for the optimization.
- A dataset containing the measurements used to generate figure 3 in the main text of the article (`target_measurements.npy`), an invasive reference measurement (`reference_measurement.npy`), and an XML with measurement parameters (`measurement_data.xml`), along with additional experimental data supporting this publication, is available on Figshare: [![Figshare DOI](https://img.shields.io/badge/Figshare-10.6084/m9.figshare.23790264-blue)](https://doi.org/10.6084/m9.figshare.23790264).


### Running the Code
To run the `main` script, download the repository, change the optimization parameters in `main.py`, and use the following command: `python main.py`

### Results
The code produces wavefront shaping reconstructions, similar to those presented in figure 3 of the publication.
<div align="center">
  <img src="https://github.com/user-attachments/assets/4eb0fa47-a430-4690-ad45-44ecac20f0e0" alt="figure_for_github" width="600">
</div>



### License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Citation
Please cite the publication if you use this code or data in your research.
