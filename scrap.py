import numpy as np
import matplotlib.pyplot as plt

def sin_func(x):
    return np.sin(x)



x = np.linspace(0,10,400)

plt.plot(x,sin_func(x))
plt.show()