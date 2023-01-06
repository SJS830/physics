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

def fitting_function(t, alpha, alphaDecay):
    return amplitude * np.exp(-alpha * np.exp(-alphaDecay * t) * t)

popt, pcov = curve_fit(fitting_function, time[peaks], pos[peaks])
alpha, alphaDecay = popt

print("Amplitude:", amplitude)
print("Equilibrium:", equilibrium)
print("Period:", period)
print("Alpha:", alpha)
print("Alpha Decay:", alphaDecay)

plt.title(f"Amplitude: {amplitude:.4f}, Equilibrium: {equilibrium:.4f}, Period: {period:.4f}, Alpha: {alpha:.4f}, Alpha Decay: {alphaDecay:.4f}")
plt.xlabel("Time [s]")
plt.ylabel("Distance from equilibrium [m]")
plt.plot(time, pos, color="red")
plt.scatter(time[peaks], pos[peaks], color="orange")
plt.plot(time, fitting_function(time, *popt), color="blue")

plt.show()
