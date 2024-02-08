"""
General functions for generating and finding trends in large data sets obtained through py_multislice
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


def store_result(func):
	# stores return of func
	def f(*args, **kwargs):
		import pickle
		x = func(*args, **kwargs)
		f = open(func.__name__+'.pkl', 'wb')
		# f = open('Hn0s.pkl', 'wb')
		pickle.dump(x, f)
		f.close()

		return x
	return f

def probe_parameter_space_multiprocessing(func, *args, **kwargs):
	""" 
	Collects samples from func based on args and returns grid on which evaluations occured.
	func: function to be sampled.
	*args: arrays containing parameter keyword as the first arguement and points to sample as the second arguement
	e.g. ['Z', np.linspace(1,20,20,dtype=int)] will draw samples for Z=1 to Z=20.
	**kwargs: inputs to func
	"""
	try:
		function_return_type = func.__annotations__["return"]
	except:
		function_return_type = float

	shape = list(sum([arg[1].shape for arg in args], ()))
	parameters_grid = np.empty(shape, dtype=np.ndarray)
	parameters_grid_eval = np.empty(shape, dtype=function_return_type)

	"""
	The following loops iterates the parameters_grid_eval positions,
	modifies the kwargs to the corresponding combination of parameters
	and passes it to func, the result is then stored in the corresponding 
	position in parameters_grid.
	"""
	for ind, _ in np.ndenumerate(parameters_grid):	# loop over slots in parameters_grid_eval
		for i, var in enumerate(ind):				# loop over indices
			kwargs[args[i][0]] = args[i][1][var]	# change kwargs depending on slot in parameters_grid_eval we are interested in
		parameters_grid[ind] = list(kwargs.values())

	parameters_grid = parameters_grid.flatten()
	parameters_grid_eval = parameters_grid_eval.flatten()

	with Pool() as pool:
		parameters_grid_eval = pool.starmap(func, parameters_grid)

	parameters_grid = parameters_grid.reshape(shape)
	parameters_grid_eval = np.asarray(parameters_grid_eval).reshape(shape)


	return parameters_grid,parameters_grid_eval

# @store_result
def probe_parameter_space(func, *args, **kwargs):
	""" 
	Collects samples from func based on args and returns grid on which evaluations occured.
	func: function to be sampled.
	*args: arrays containing parameter keyword as the first arguement and points to sample as the second arguement
	e.g. ['Z', np.linspace(1,20,20,dtype=int)] will draw samples for Z=1 to Z=20.
	**kwargs: inputs to func
	"""
	try:
		function_return_type = func.__annotations__["return"]
	except:
		function_return_type = float

	parameters_grid_eval = np.empty(
			list(sum([arg[1].shape for arg in args], ()))
		, dtype=function_return_type)
	parameters_grid = np.empty(parameters_grid_eval.shape, dtype=np.ndarray)

	"""
	The following loops iterates the parameters_grid_eval positions,
	modifies the kwargs to the corresponding combination of parameters
	and passes it to func, the result is then stored in the corresponding 
	position in parameters_grid.
	"""
	for ind, _ in np.ndenumerate(parameters_grid_eval):	# loop over slots in parameters_grid_eval
		params = []
		for i, var in enumerate(ind):				# loop over indices
			params.append(args[i][1][var])
			kwargs[args[i][0]] = args[i][1][var]	# change kwargs depending on slot in parameters_grid_eval we are interested in
		parameters_grid_eval[ind] = func(**kwargs)
		parameters_grid[ind] = params

	return parameters_grid,parameters_grid_eval

# @store_result
def parameter_space_best_fit(parameters_grid, parameters_grid_eval, f_model, x0):
	# obtain fit for data using least squares fit, returns fit parameters according to specified model

	def e(model_params):
		"""
		populates parameters_grid_eval using f_model and model parameters and 
		finds the absolute squared error with parameters_grid.
		"""
		parameters_grid_model = np.empty(parameters_grid.shape)
		for ind, _ in np.ndenumerate(parameters_grid):
			parameters_grid_model[ind] = f_model(*model_params, *parameters_grid[ind])

		return np.sum(
				np.abs(parameters_grid_eval - parameters_grid_model)**2
			)

	from scipy.optimize import minimize
	return minimize(e, x0)

def plot_parameter_space(parameters_grid, parameters_grid_eval,**kwargs):
	"""
	Provide parameters which will remain fixed and plot with every compination which 
	will vary is plotted. If one parameter is varied 1d plots are returned, if two parameters 
	varied then 2d plots returned.
	"""
	...
	# for kwargs





def mosaic(func,*args, **kwargs):
	"""
	creates mosaic based on two parameters with each tile of the mosaic corresponding to a
	number of unit cells. 
	"""
	if len(args) != 2:
		raise AttributeError('Two parameters must be specified to build a mosiac')

	# specify return type explicitly
	def tile_sample(**kwargs) -> np.ndarray:
		return func(**kwargs)

	_, parameters_grid_eval = probe_parameter_space(tile_sample, *args, **kwargs)
	parameters_grid_eval = np.array(parameters_grid_eval)
	gx, gy = parameters_grid_eval.shape
	px, py = parameters_grid_eval[0,0].shape

	mosaic_img = np.empty([px*gx, py*gy], dtype=float)
	for ind, _ in np.ndenumerate(parameters_grid_eval):
		sx, sy = np.asarray(ind)*np.asarray([px, py])
		ex, ey = sx+px,sy+py
		mosaic_img[sx:ex, sy:ey] = parameters_grid_eval[ind]

	# plot mosaic
	fig, ax = plt.subplots()
	im = ax.imshow(mosaic_img, extent=(0, gy*py, 0, gx*px))

	ax.set_ylabel(args[0][0])
	ax.set_xlabel(args[1][0])

	ax.set_xticks(np.asarray(range(gy))*py + py/2)
	ax.set_xticklabels(args[1][1])

	ax.set_yticks(ax.get_ylim()[1] - np.flip(range(gx))*px - px/2)
	ax.set_yticklabels(np.flip(args[0][1]))

	return fig, ax, im

