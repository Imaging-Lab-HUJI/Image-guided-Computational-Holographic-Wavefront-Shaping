# Image-guided Computational Holographic Wavefront Shaping
This repository contains the Python code implementation as described and presented in the article "*Image-guided Computational Holographic Wavefront Shaping*" by Omri Haim, Jeremy Boger-Lombard and Ori Katz.

## Prerequisites
- Python 3.8 or later
- PyTorch 1.12.0 or later
- numpy
- matplotlib (optional)
- xmltodict

## Project Structure
- main.py: The main script that runs the optimization.
- data.py: Contains the load_data function to load the measurement data from the dataset.
- optimization.py: Contains functions for the gradient ascent optimization.
- utility.py: Contains utility functions for the optimization.
- A dataset containing the measurements used to generate figure 2 in the main text of the article (target_measurements.npy), an invasive reference measurement (reference_measurement.npy) and an xml with measurement parameters (measurement_data.xml).

## Running the Code
To run the main script, download the repository, change the optimization parameters in ```main.py``` according to the documentation and use the following command: ```python main.py```.
