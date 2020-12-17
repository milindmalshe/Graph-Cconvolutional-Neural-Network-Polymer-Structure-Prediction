import numpy as np
from  scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x, a, b, c):

    return a*np.log(1 + np.exp(b*x + c))


xdata = np.linspace(0.0, 1.0, 101)
y = func(xdata, 1,  -10, 10)


plt.plot(xdata, y, 'ko')
plt.show()

popt, pconv = curve_fit(func, xdata=xdata, ydata=y)
print popt