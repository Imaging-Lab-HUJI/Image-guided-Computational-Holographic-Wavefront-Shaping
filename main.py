import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from enums import PhaseInit, FieldInit, QualityMetric
import optimization as optim
import data


def optim_main():

    optim_params = optim_params_initialization()

    # Load dataset to workspace: I_cam is the camera image of target measurements, and
    # I_cam_ref is the invasive reference camera image
    I_cam, I_cam_ref, optim_params = data.load_data(optim_params)

    # Perform phase gradient ascent optimization:
    # - phase is the optimized phase,
    # - I_final is the optimized object image,
    # - metric_values are the values of the image quality metric during optimization
    phase, I_final, metric_values = optim.phase_optim(I_cam, optim_params)

    # Display the final object image
    plt.imshow(I_final, cmap="gray")
    plt.colorbar()
    plt.show(block=True)



def optim_params_initialization():
    optim_params = dict()
    # ================================================================================== #
    # ============================== User Initializations ============================== #
    # ================================================================================== #

    # Initial phase for the optimization process (see `enums.py` for options)
    optim_params["initial_phase"] = PhaseInit.ZEROS

    # Desired number of sqrt(DoF) for optimization
    optim_params["DOF"] = 300

    # Number of fields to optimize. Set to `np.inf` to use all fields available.
    optim_params["num_fields"] = np.inf

    # If number of fields is less than total number of fields, select fields according to
    # the following criteria (see `enums.py` for options).
    optim_params["init_fields_mode"] = FieldInit.ALL

    # Maximum number of allowed iterations
    optim_params["num_iters"] = 500

    # Stop condition for the optimization process in radians
    optim_params["stop_condition"] = 0

    # Image quality metric for the optimization process (see `enums.py` for options)
    optim_params["image_quality_metric"] = QualityMetric.VARIANCE

    # The propagation distances between the planes in single phase optimization.
    # Leave as None for pre-programmed value
    optim_params["z_prop"] = None

    # Ratio for zero padding. Final propagated lateral field Size = (2 * zero_pad_ratio + 1) * DOF
    optim_params["zero_pad_ratio"] = 0.8
    # ================================================================================== #


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Don't Touch These! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set the device to GPU if available, else CPU
    optim_params["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Set the data type for torch tensors
    optim_params["dtype"] = torch.float32
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return optim_params


def set_random_seed():
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # # Use deterministic algorithms only
    # torch.use_deterministic_algorithms(True)
    # # Use deterministic convolution algorithm in CUDA
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_random_seed()
    optim_main()




