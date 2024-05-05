import torch
from torch import pi
from torch import fft

utility_globals = {"circle_filter": None,
                   "rect_filter": None,
                   'do_stop': False,
                   'prev_phase': None,
                   'prev_grad': None,
                   'argH': None}


def reset_globals():
	global utility_globals
	del utility_globals["circle_filter"], utility_globals["rect_filter"], \
		utility_globals["do_stop"], utility_globals["prev_phase"], utility_globals["prev_grad"], \
		utility_globals["argH"]

	utility_globals["circle_filter"] = None
	utility_globals["rect_filter"] = None
	utility_globals["do_stop"] = False
	utility_globals["prev_phase"] = None
	utility_globals["prev_grad"] = None
	utility_globals["argH"] = None


def calc_pixel_size(init_img: torch.Tensor, cropped_img: torch.Tensor, params: dict):
	global utility_globals
	magnif = params["magnif_4f"]
	cam_pixel_size = 6.5e-6
	cropped_img_size = cropped_img.shape
	init_img_size = init_img.shape
	return cam_pixel_size * init_img_size[0] / cropped_img_size[0] / magnif


def ang_spec_prop(Ein: torch.Tensor, pixel_size: float, wl: float, z: float,
                  use_globals=True):
	"""
	This function calculates the angular spectrum propagation of a field.
	The angular spectrum propagation is a method used in wave optics to calculate the
	field distribution of a wave at a certain distance from the source, given the field
	distribution at the source.
	:param Ein: The input field(s).
	:param pixel_size: The size of a pixel in the input field.
	:param wl: The wavelength of the light.
	:param z: The propagation distance.
	:param use_globals: A flag to determine whether to use the stored angular spectrum transfer
						function.
	:return: The propagated field(s).
	"""
	if use_globals:
		# If ARGH is not defined or its dimensions do not match the input field,
		# calculate the angular spectrum transfer function
		if utility_globals["argH"] is None \
				or (isinstance(utility_globals["argH"], torch.Tensor) and
				    utility_globals["argH"].shape[2] < Ein.shape[2]):
			# Calculate the angular spectrum transfer function argument
			utility_globals["argH"] = ang_spec_trans_func_arg(Ein, pixel_size, wl)

		# Multiply the Fourier transform of the input field by the transfer function
		# then calculate the inverse Fourier transform to get the propagated field
		return fft.ifft2(fft.ifftshift(
			fft.fftshift(fft.fft2(Ein, dim=(0, 1)), dim=(0, 1))
			* torch.exp(1j * (utility_globals["argH"][:, :, 0:Ein.shape[2]] * z)),
			dim=(0, 1)), dim=(0, 1))
	else:
		H_local = ang_spec_transfer_func(Ein, pixel_size, wl, z)
		return fft.ifft2(fft.ifftshift(
			fft.fftshift(fft.fft2(Ein, dim=(0, 1)), dim=(0, 1)) * H_local,
			dim=(0, 1)), dim=(0, 1))


def ang_spec_transfer_func(Ein: torch.Tensor, pixel_size: float, wl: float, z: float):
	"""
	This function calculates the angular spectrum transfer function of a field.
	"""
	H_arg = ang_spec_trans_func_arg(Ein, pixel_size, wl)
	return torch.exp(1j * z * H_arg).to(device=Ein.device).detach()


def ang_spec_trans_func_arg(Ein: torch.Tensor, pixel_size: float, wl: float):
	"""
	This function calculates the angular spectrum transfer function angle (without z).
	"""
	diffraction_Ny = Ein.size(dim=0)
	diffraction_Nx = Ein.size(dim=1)
	kx_1d = torch.linspace(-pi, pi, diffraction_Nx) / pixel_size
	ky_1d = torch.linspace(-pi, pi, diffraction_Ny) / pixel_size
	Kx, Ky = torch.meshgrid(kx_1d, ky_1d, indexing='ij')
	if Ein.dim() == 3:
		NUM_FIELDS = Ein.size(dim=2)
		Kx = torch.unsqueeze(Kx, dim=-1).repeat(1, 1, NUM_FIELDS)
		Ky = torch.unsqueeze(Ky, dim=-1).repeat(1, 1, NUM_FIELDS)

	return torch.sqrt((2 * pi / wl) ** 2 - (Kx ** 2 + Ky ** 2)).to(device=Ein.device).detach()


def off_axis_extract(imgs: torch.Tensor, optim_params):
	"""
	This function extracts the off-axis field from an image. The off-axis field is a
	portion of the Fourier transform of the image that corresponds to a specific spatial
	frequency range.

	:param imgs: The input image.
	:param optim_params: A dictionary containing the optimization parameters. It should
	include the coordinates for cropping the image and the DoFs for the optimization.
	:return: The extracted off-axis field.
	"""

	# Extract the crop coordinates and the degree of freedom from the optimization
	# parameters
	X_min, X_max, Y_min, Y_max, Xfft_center, Yfft_center = \
		optim_params["crop_coordinates"]
	half_size = optim_params["DOF"] // 2

	# If the image has two dimensions, crop the image and calculate the Fourier transform
	if imgs.ndim == 2:
		imgs = imgs[Y_min:Y_max, X_min: X_max]
		imgs_F = torch.fft.fftshift(torch.fft.fft2(imgs))
		# Extract the off-axis field from the Fourier transform
		imgs_F = imgs_F[Yfft_center - half_size:Yfft_center + half_size,
		                Xfft_center - half_size:Xfft_center + half_size]
		# Calculate the inverse Fourier transform to get the field in the spatial domain
		E = torch.fft.ifft2(torch.fft.ifftshift(imgs_F))
	else:
		# If the image has more than two dimensions, repeat the above steps for each
		# dimension
		imgs = imgs[Y_min:Y_max, X_min: X_max, :]
		imgs_F = torch.fft.fftshift(torch.fft.fft2(imgs, dim=(0, 1)), dim=(0, 1))
		imgs_F = imgs_F[Yfft_center - half_size:Yfft_center + half_size,
		                Xfft_center - half_size:Xfft_center + half_size, :]
		E = torch.fft.ifft2(torch.fft.ifftshift(imgs_F, dim=(1, 0)), dim=(0, 1))

	# Move the field to the device specified in the optimization parameters
	E = E.to(device=optim_params["device"])

	# Calculate the pixel size of the field
	optim_params["pixel_size"] = calc_pixel_size(imgs, E, optim_params)

	return E


def two_point_step_size(grad_val: torch.Tensor, phase_val: torch.Tensor):
	"""
	This function calculates the step size for the two-point step size method.
	The two-point step size method is a way to adaptively choose the step size in a
	gradient ascent algorithm. It uses the gradient values at the current and previous
	steps to calculate the step size.
	:param grad_val: The gradient value at the current step.
	:param phase_val: The phase value at the current step.
	:return step_size: The calculated step size.
	"""
	grad_val = torch.flatten(grad_val)
	phase_val = torch.flatten(phase_val)

	# If this is the first iteration or the phase or gradient have not changed,
	# calculate an initial step size
	if (utility_globals["prev_phase"] is None or utility_globals["prev_grad"] is None) or \
			((utility_globals["prev_phase"] is not None) and
			 torch.equal(phase_val, utility_globals["prev_phase"])) or \
			((utility_globals["prev_grad"] is not None) and
			 torch.equal(grad_val, utility_globals["prev_grad"])):
		utility_globals["prev_phase"] = phase_val.detach().clone()
		utility_globals["prev_grad"] = grad_val.detach().clone()
		alpha = torch.max(utility_globals["prev_grad"])
		alpha = torch.sign(alpha) * alpha
		return 10 ** (-torch.floor(torch.log10(alpha)) - 1)

	# If this is not the first iteration and the phase and gradient have changed,
	# calculate the step size
	step_size = torch.abs((phase_val - utility_globals["prev_phase"]).t() @
	                      (grad_val - utility_globals["prev_grad"])) / \
	            (torch.linalg.norm(grad_val - utility_globals["prev_grad"]) ** 2)

	# Store the current phase and gradient for the next iteration
	utility_globals["prev_phase"] = phase_val.detach().clone()
	utility_globals["prev_grad"] = grad_val.detach().clone()

	return step_size


def pick_least_correlative(E_n: torch.Tensor, k: int):
	"""
	This function picks the k-least correlative fields (approximately), from a set
	of input fields.
	:param E_n: The set of fields.
	:param k: The number of fields to pick.
	:return:
	"""
	I_n = torch.abs(E_n) ** 2
	nrows = I_n.size(dim=0)
	ncols = I_n.size(dim=1)
	N = E_n.size(dim=2)
	L = nrows * ncols
	I_n = torch.reshape(I_n, (L, N))
	# l'th row = one sample of all pixels / one picture
	# n'th column = all samples corresponding to the (i,j) pixel
	I_nT = I_n.t().conj()
	R = torch.corrcoef(I_nT)
	R.fill_diagonal_(0)
	V = torch.mean(R, dim=0)
	idxs = torch.argsort(V)
	idxs = idxs[0:k]
	E_n_min_corr = E_n[..., idxs]
	return E_n_min_corr, idxs
