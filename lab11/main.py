import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

df = pd.read_csv("data.csv", header=None)
df = df.values

time, pos = df[:, 0], df[:, 1]

equilibrium = np.mean(pos)
pos -= equilibrium

amplitude = np.max(pos)

peaks, _ = find_peaks(pos)
peaks = peaks[np.where(pos[peaks] > .0015)]
period = np.mean(np.diff(time[peaks]))

def fitting_function(t, alpha):
    return amplitude * np.exp(-alpha * t)

popt, pcov = curve_fit(fitting_function, time[peaks], pos[peaks])
alpha = popt[0]

print("Amplitude:", amplitude)
print("Equilibrium:", equilibrium)
print("Period:", period)
print("Alpha:", alpha)

plt.title(f"Amplitude: {amplitude:.2f}, Equilibrium: {equilibrium:.2f}, Period: {period:.2f}, Alpha: {alpha:.2f}")
plt.plot(time, pos, color="red")
plt.scatter(time[peaks], pos[peaks], color="orange")
plt.plot(time, amplitude * np.exp(-alpha * time), color="blue")

plt.show()
