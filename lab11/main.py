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

def amplitudeFunction(t, alpha, alphaDecay):
    return amplitude * np.exp(-alpha * np.exp(-alphaDecay * t) * t)

popt, pcov = curve_fit(amplitudeFunction, time[peaks], pos[peaks])
alpha, alphaDecay = popt

def positionFunction(t, phaseShift):
    return amplitudeFunction(t, alpha, alphaDecay) * np.cos(2 * np.pi * (t - phaseShift) / period)

popt, pcov = curve_fit(positionFunction, time, pos)
phaseShift, = popt

print(f"Amplitude: {amplitude:.4f}m")
print(f"Equilibrium: {equilibrium:.4f}m")
print(f"Period: {period:.4f}s")
print(f"Phase Shift: {phaseShift:.4f}s")
print(f"Alpha: {alpha:.4f}")
print(f"Alpha Decay: {alphaDecay:.4f}")

plt.title(f"Amplitude: {amplitude:.4f}m, Equilibrium: {equilibrium:.4f}m, Period: {period:.4f}s, Phase Shift: {phaseShift:.4f}s, Alpha: {alpha:.4f}, Alpha Decay: {alphaDecay:.4f}")
plt.xlabel("Time [s]")
plt.ylabel("Distance from equilibrium [m]")

plt.scatter(time[peaks], pos[peaks], color="orange", label="Peaks")

lines = [
    plt.plot(time, pos, color="red", label="Actual position")[0],
    plt.plot(time, amplitudeFunction(time, alpha, alphaDecay), color="blue", label="Amplitude")[0],
    plt.plot(time, positionFunction(time, phaseShift), color="green", label="Fit")[0]
]

plt.legend(handles=lines)
plt.show()
