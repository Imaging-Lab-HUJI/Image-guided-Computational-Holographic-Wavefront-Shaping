import numpy as np
import torch
import xmltodict


def load_data(optim_params):
    """
    This function loads the measurement data from the dataset.
    It reads the XML file containing the measurement data and loads the target and
    reference images from the .npy files.
    :param optim_params: A dictionary containing the optimization parameters.
    :return: The target and reference measurements and updated optimization parameters.
    """
    # Open and read the XML file containing measurement data
    with open('measurement_data.xml', 'r', encoding='utf-8') as file:
        xml_data = file.read()

    # Parse the XML data into a dictionary
    measurement_data = xmltodict.parse(xml_data)['struct']

    # Extract necessary parameters from the measurement data
    optim_params['num_imgs'] = int(measurement_data['num_imgs'])
    optim_params['wl'] = float(measurement_data['wl'])

    if optim_params["z_prop"] is None:
        optim_params["z_prop"] = float(measurement_data['Z_prop'])

    # Convert crop coordinates from string to integer
    for i in range(4):
        measurement_data['crop_coordinates'][i] = \
            int(measurement_data['crop_coordinates'][i])
    optim_params["crop_coordinates"] = measurement_data['crop_coordinates']

    # Add FFT center coordinates to the crop coordinates
    for i in range(2):
        measurement_data['crop_coordinates'].append(int(measurement_data[
                                                             'fft_center'][i]))

    # Extract the 4f magnification factor
    optim_params['magnif_4f'] = float(measurement_data['magnif_4f'])

    # Load target and reference images from numpy files
    target_imgs = torch.tensor(np.load('target_measurements.npy'))
    ref_imgs = torch.tensor(np.load('reference_measurement.npy'))

    # Return the images and updated optimization parameters
    return target_imgs, ref_imgs, optim_params
