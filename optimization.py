import gc
import torch
from torch import fft
from torch import pi

import utility
from enums import QualityMetric, PhaseInit, FieldInit

optim_globals = {"Q0": None, "stop_flag": False}



def phase_optim(I_cam: torch.Tensor, optim_params: dict):
	"""
	This main function that performs gradient ascent to optimize the virtual SLM phase.
	:param I_cam: The input image in the camera plane.
	:param optim_params: A dictionary containing the optimization parameters.
						 It must include the degrees of freedom, the maximum number of
						 iterations, the stopping condition, the image quality metric,
						 and the device and data type for calculations.
	:return: The optimized virtual SLM phase,
			 the intensity of the final field in the object plane,
			 the values of the image quality metric at each iteration,
			 the updated optimization parameters
	"""

	E_n, phase, metric_values = initialize_optim(I_cam, optim_params)

	phase, metric_values = optim_loop(E_n, phase, metric_values, optim_params)
	print("Finished!")

	# Calculate the final object field and intensity
	O_final = physical_model(E_n, phase, optim_params)
	I_final = torch.mean(torch.abs(O_final) ** 2, dim=2).detach().cpu().numpy()
	# Detach the phase from the computation graph and convert to numpy
	phase = phase.detach().cpu().numpy()

	return phase, I_final, metric_values


def initialize_optim(I_cam: torch.Tensor, optim_params: dict,
                     I_cam_ref: torch.Tensor = None):

	# Extract the off-axis field from the camera image
	E_n = utility.off_axis_extract(I_cam, optim_params)

	# Calculate the padding size for the propagation of the fields
	optim_params["pad_size"] = (int(E_n.shape[0] *
	                                optim_params["zero_pad_ratio"]),) * 4

	# Choose Initial Phase
	# ----------------------- Zero Phase -----------------------
	if optim_params["initial_phase"] == PhaseInit.ZEROS:
		phase = torch.zeros(E_n.shape[0:2], requires_grad=True,
		                    device=optim_params["device"],
		                    dtype=optim_params["dtype"])

	# ---------------------- Random Phase ----------------------
	elif optim_params["initial_phase"] == PhaseInit.RANDOM:
		phase = 2 * torch.pi * torch.rand(E_n.shape[0:2],
		                                  requires_grad=True,
		                                  device=optim_params["device"],
		                                  dtype=optim_params["dtype"])
	# ------------------ Default: Zero Phase -------------------
	else:
		phase = torch.zeros(E_n.shape[0:2], requires_grad=True,
		                    device=optim_params["device"],
		                    dtype=optim_params["dtype"])
	# ----------------------------------------------------------

	# Choose initial fields
	if optim_params["num_fields"] < E_n.size(dim=2):
		# --------------------------- Random Initialization ------------------------------
		if optim_params["init_fields_mode"] == FieldInit.RANDOM:
			optim_params["picked_fields"] = torch.randint(0, E_n.size(dim=2),
			                                              (optim_params["num_fields"],))
			E_n = E_n[:, :, optim_params["picked_fields"]]
		# --------------------------- Linear Initialization ------------------------------
		elif optim_params["init_fields_mode"] == FieldInit.LINEAR:
			E_n = E_n[:, :, 0:optim_params["num_fields"]]
		# --------------------- Choose the least correlative fields ----------------------
		elif optim_params["init_fields_mode"] == FieldInit.MIN_CORRELATION:
			E_n, optim_params["picked_fields"] \
				= utility.pick_least_correlative(E_n, optim_params["num_fields"])
	else:
		optim_params["num_fields"] = E_n.size(dim=2)


	# Initialize a list to store the values of the image quality metric
	metric_values = []
	# Reset global variables
	reset_globals()

	return E_n, phase, metric_values



def optim_loop(E_n: torch.Tensor, phase: torch.Tensor, metric_values: list,
               optim_params: dict):
	"""
	This function performs the main loop of the gradient ascent optimization. It
	calculates the propagated field and the image quality metric at each iteration, and
	updates the phase using the gradient of the metric.
	:param E_n: The input field in the proximity plane.
	:param phase: The initial virtual SLM phase.
	:param metric_values: A list to store the values of the image quality metric at each
						  iteration.
	:param optim_params: A dictionary containing the optimization parameters.
						 It should include the maximum number of iterations and the
						 stopping condition.
	:return: The optimized phase and the updated list of image quality metric values.
	"""
	# Loop over the pre-determined number of iterations
	for t in range(1, optim_params["num_iters"] + 1):

		# Calculate the propagated object fields
		O_n = physical_model(E_n, phase, optim_params).to(optim_params["device"])

		# Calculate the current image quality metric
		Q_t = calculate_image_quality(O_n, optim_params)

		# Update the phase using the gradient ascent step
		phase = step(Q_t, phase, optim_params)

		# Append the current metric value to the list (for convenience)
		metric_values.append(Q_t.cpu().detach().clone().item())

		# If the stopping condition is reached, break the loop
		if optim_globals["stop_flag"] is True:
			print("Stopping condition reached!")
			break

	# If the loop finished without reaching the stopping condition, print a message
	if optim_globals["stop_flag"] is False:
		print("Reached last iteration!")

	return phase, metric_values



def physical_model(E_n: torch.Tensor, phase: torch.Tensor, optim_params: dict):
	"""
	This function calculates the propagated fields in the object plane.
	The propagation is done using the angular spectrum propagation method.
	:param E_n: The input field in the proximal plane.
	:param phase: Virtual SLM phase
	:param optim_params: A dictionary containing the optimization parameters. It should
						 include the pixel size, the wavelength, and the propagation
						 distance.
	:return: The propagated fields in the object plane.
	"""

	# Calculate the exponential of the phase: the virtual SLM
	exp_phase = torch.exp(-1j * phase)

	# If the proximal field has more than 2 dimensions, repeat the exponential phase
	if E_n.ndim > 2:
		NUM_FIELDS = E_n.size(dim=2)
		exp_phase = torch.unsqueeze(exp_phase, dim=-1).repeat(1, 1, NUM_FIELDS)

	# Multiply the proximal field by the virtual SLM
	U_n = E_n * exp_phase

	# If the proximal field has more than 2 dimensions, pad the back surface fields

	if E_n.ndim > 2:
		U_n = torch.nn.functional.pad(U_n, (0, 0, *optim_params["pad_size"]))
	else:
		U_n = torch.nn.functional.pad(U_n, *optim_params["pad_size"])

	# Return the angular spectrum propagation of the distal field
	return utility.ang_spec_prop(U_n, optim_params["pixel_size"],
	                             optim_params["wl"], optim_params["z_prop"])


def calculate_image_quality(O_n, optim_params):
	"""
	This function calculates the image quality metric for the given field.

	:param O_n: The fields in the object plane.
	:param optim_params: A dictionary containing the optimization parameters.
						 It should include the image quality metric indicator.
	:return: The calculated image quality metric.
	"""
	# Calculate the object intensity
	I_t = torch.mean(torch.abs(O_n) ** 2, dim=2)

	# Calculate the image quality metric based on the user's choice
	if optim_params["image_quality_metric"] == QualityMetric.VARIANCE:
		optim_params["direction"] = 1
		Q = torch.var(I_t)

	elif optim_params["image_quality_metric"] == QualityMetric.ENTROPY:
		optim_params["direction"] = -1
		I_t_nrm = I_t / torch.sum(I_t) + 1e-19
		Q = -torch.sum(I_t_nrm * torch.log10(I_t_nrm))

	elif optim_params["image_quality_metric"] == QualityMetric.FOURIER_VARIANCE:
		optim_params["direction"] = 1
		Q = torch.var(torch.abs(fft.fftshift(fft.fft2(I_t), dim=(0, 1))))

	else:
		optim_params["direction"] = 1
		Q = torch.var(I_t)

	# If this is the first iteration, store the initial quality metric value
	if optim_globals["Q0"] is None:
		with torch.no_grad():
			optim_globals["Q0"] = Q.detach().clone()

	Q_t = Q / optim_globals["Q0"]
	if torch.isnan(Q_t):
		print("Q_t is NaN")

	return Q_t


def step(curr_metric: torch.Tensor, phase: torch.Tensor, optim_params):
	"""
	This function performs one step of the gradient ascent optimization.
	It calculates the gradient of the image quality metric with respect to the phase,
	and updates the phase using the calculated step size.
	:param curr_metric: The current value of the image quality metric.
	:param phase: The current virtual SLM phase.
	:param optim_params: A dictionary containing the optimization parameters.
	                     It should include the stopping condition.
	:return: The updated phase.
	"""

	# Retain the gradient of the phase for backpropagation
	phase.retain_grad()

	# Perform backpropagation to calculate the gradient of the current metric
	curr_metric.backward(retain_graph=True)

	# Get the gradient of the phase
	dQ_dphi = phase.grad
	direction = optim_params["direction"]

	with (torch.no_grad()):
		# Calculate the step size using the two-point step size method
		step_size = utility.two_point_step_size(dQ_dphi.detach(), phase.detach())

		# Update the phase using the calculated step size and gradient
		phase = phase + direction * step_size * dQ_dphi

		# Modulate the phase by 2*pi to keep it within the range [0, 2*pi]
		phase = phase % (2 * pi)

		# If the maximum change in phase is below the stopping condition,
		# set the stop flag
		if torch.max(dQ_dphi) != 0 and \
		   step_size * torch.max(dQ_dphi) <= optim_params["stop_condition"]:
			utility.utility_globals["do_stop"] = True

	# Set the phase to require gradient for the next iteration
	phase.requires_grad = True

	return phase


def reset_globals():
	"""
	This function resets the global variables.
	"""
	del optim_globals["Q0"], optim_globals["stop_flag"]
	optim_globals["Q0"] = None
	optim_globals["stop_flag"] = False
	utility.reset_globals()
	gc.collect()
	torch.cuda.empty_cache()
