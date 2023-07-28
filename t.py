import pyms
import re
import matplotlib.pyplot as plt
import numpy as np

Z = 6
n = 0
ell = 0
epsilon = 100
target_orbital_string='1s'

s = pyms.orbital(
	Z=Z, 
	config=pyms.full_orbital_filling(Z), 
	n=n, 
	ell=ell, 
	epsilon=epsilon, 
	target_orbital_string=target_orbital_string
)

r = np.linspace(0,10,1000)
plt.plot(r,s(r))
plt.show()