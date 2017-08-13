import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from scipy.io.wavfile import write

# 16 kHz (16000 samples per second), 16 bits (2^16 different amplitude values)
spf = wave.open("raw_data/helloworld.wav", 'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, "Int16")
print ("shape: ", signal.shape)

plt.plot(signal)
plt.title("Hellow world without echo")
plt.show()

delta = np.array([1., 0., 0.])
noecho = np.convolve(signal, delta)
print ("noecho shape: ", noecho.shape)
assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.0000001)

noecho = noecho.astype(np.int16)
write("output/noecho.wav", 16000, noecho)

# One second
filt = np.zeros(16000)
# Repeat the signal itself
filt[0] = 1
# Every quarter, decrease by 0.6
filt[4000] = 0.6
filt[8000] = 0.3
filt[12000] = 0.2
filt[15999] = 0.1
print ("filt shape: ", filt.shape)
out = np.convolve(signal, filt)
out = out.astype(np.int16)
print ("echo shape: ", out.shape)
write("output/echo.wav", 16000, out)
