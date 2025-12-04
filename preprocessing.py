import scipy 
import pandas as pd 
import numpy as np 
from skimage.util.shape import view_as_windows

def signal_clean(signal, sampling_rate=125):
  # highpass_filter 
  sos = scipy.signal.butter(5, 0.5, btype='highpass', output="sos", fs=sampling_rate)
  filtered = scipy.signal.sosfiltfilt(sos, signal)

  #powerline filter
  b = np.ones(int(sampling_rate / 50))
  a = [len(b)]
  y = scipy.signal.filtfilt(b, a, filtered, method="pad")

  return y 


def stationarySignalCheck(signal):
  if signal.shape[0] == 0:
    return np.empty((0,1250))
    
  filter = (signal.shape[0],25)
  windows = view_as_windows(signal, window_shape = filter, step=5)
  window_min = windows.min(axis=3)
  window_max = windows.max(axis=3)
  filteredBool = (window_min != window_max).all(axis=1).reshape(-1,)
  filteredSignal = signal[filteredBool, :]
  return filteredSignal

class Pan_Tompkins_QRS():
  
  def band_pass_filter(self,signal):
    '''
    Band Pass Filter
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    Bandpass filter is used to attenuate the noise in the input signal.
    To acheive a passband of 5-15 Hz, the input signal is first passed 
    through a low pass filter having a cutoff frequency of 11 Hz and then
    through a high pass filter with a cutoff frequency of 5 Hz, thus
    achieving the required thresholds. 

    The low pass filter has the recursive equation:
      y(nT) = 2y(nT - T) - y(nT - 2T) + x(nT) - 2x(nT - 6T) + x(nT - 12T)

    The high pass filter has the recursive equation:
      y(nT) = 32x(nT - 16T) - y(nT - T) - x(nT) + x(nT - 32T)
    '''

    # Initialize result
    result = None

    # Create a copy of the input signal
    sig = signal.copy()
	
    for index in range(signal.shape[1]):
      sig[:, index] = signal[:, index]

      if (index >= 1):
        sig[:, index] += 2*sig[:, index-1]

      if (index >= 2):
        sig[:,index] -= sig[:,index-2]

      if (index >= 6):
        sig[:,index] -= 2*signal[:,index-6]

      if (index >= 12):
        sig[:,index] += signal[:,index-12] 

    # Copy the result of the low pass filter
    result = sig.copy()

    # Apply the high pass filter using the equation given
    for index in range(signal.shape[1]):
      result[:,index] = -1*sig[:,index]

      if (index >= 1):
        result[:,index] -= result[:,index-1]

      if (index >= 16):
        result[:,index] += 32*sig[:,index-16]

      if (index >= 32):
        result[:,index] += sig[:,index-32]

    # Normalize the result from the high pass filter
    max_val = np.stack([result.max(axis=1),-result.min(axis=1)]).max(axis=0)
    result = result/max_val[:, None]

    return result

  def derivative(self,signal):
    '''
    Derivative Filter 
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The derivative of the input signal is taken to obtain the
    information of the slope of the signal. Thus, the rate of change
    of input is obtain in this step of the algorithm.

    The derivative filter has the recursive equation:
      y(nT) = [-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]/(8T)
    '''

    # Initialize result
    result = signal.copy()

    # Apply the derivative filter using the equation given
    for index in range(signal.shape[1]):
      result[:,index] = 0

      if (index >= 1):
        result[:,index] -= 2*signal[:,index-1]

      if (index >= 2):
        result[:,index] -= signal[:,index-2]

      if (index >= 2 and index <= signal.shape[1]-2):
        result[:,index] += 2*signal[:,index+1]

      if (index >= 2 and index <= signal.shape[1]-3):
        result[:,index] += signal[:,index+2]

      result[:,index] = (result[:,index]*125)/8

    return result

  def squaring(self,signal):
    '''
    Squaring the Signal
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The squaring process is used to intensify the slope of the
    frequency response curve obtained in the derivative step. This
    step helps in restricting false positives which may be caused
    by T waves in the input signal.

    The squaring filter has the recursive equation:
      y(nT) = [x(nT)]^2
    '''

    # Initialize result
    result = signal.copy()

    # Apply the squaring using the equation given
    for index in range(len(signal)):
      result[index] = signal[index]**2

    return result    

  def moving_window_integration(self,signal):
    '''
    Moving Window Integrator
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The moving window integration process is done to obtain
    information about both the slope and width of the QRS complex.
    A window size of 0.15*(sample frequency) is used for more
    accurate results.

    The moving window integration has the recursive equation:
      y(nT) = [y(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N

      where N is the number of samples in the width of integration
      window.
    '''

    # Initialize result and window size for integration
    result = signal.copy()
    win_size = round(0.150 * 125)
    sum = 0

    # Calculate the sum for the first N terms
    for j in range(win_size):
      sum += signal[:, j]/win_size
      result[:,j] = sum

    # Apply the moving window integration using the equation given
    for index in range(win_size,signal.shape[1]):  
      sum += signal[:,index]/win_size
      sum -= signal[:,index-win_size]/win_size
      result[:,index] = sum

    return result

  def solve(self,signal):
    '''
    Solver, Combines all the above functions
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The peak detection algorithm works on the moving window and bandpass
    filtered signal. So the input signal is first bandpassed, then the
    output of the bandpass filter is given to the derivative function and
    the result is squared. Finally the output of the squaring function
    is given to the moving window integration function and returned. 
    '''

    # Convert the input signal into numpy array
    input_signal = signal

    # Bandpass Filter
    global bpass
    bpass = self.band_pass_filter(input_signal.copy())

    # Derivative Function
    global der
    der = self.derivative(bpass.copy())

    # Squaring Function
    global sqr
    sqr = self.squaring(der.copy())

    # Moving Window Integration Function
    global mwin
    mwin = self.moving_window_integration(sqr.copy())

    return mwin

def heartRateCheck(signal):
  if signal.shape[0] == 0:
    return np.empty((0,1250))
    
  QRS_detector = Pan_Tompkins_QRS()
  processedSignal = QRS_detector.solve(signal)

  index_list = []
  for index in range(processedSignal.shape[0]):
    peaks, _  = scipy.signal.find_peaks(processedSignal[index,:], height=np.mean(processedSignal[index,:]), distance=round(125*0.200))
    HR = len(peaks) * 6
    if (HR >= 24) & (HR <= 300):
      index_list.append(index)

  output_signal = signal[index_list,:]
  return output_signal

def SNRcheck(signal, sampling_rate=125):
  if signal.shape[0] == 0:
    return np.empty((0,1250))
    
  f, den = scipy.signal.periodogram(signal, fs=sampling_rate, axis=1)

  PSD_signal = den[:, (f >= 2) & (f<=40)].sum(axis=1)
  PSD_noise1 = den[:, (f >= 0) & (f<=2)].sum(axis=1)
  PSD_noise2 = den[:, (f >= 40)].sum(axis=1)

  PSD_signal/(PSD_noise1+PSD_noise2)

  filteredSignal = signal[PSD_signal/(PSD_noise1+PSD_noise2) >= 0.5, :]

  ##### AF challenge 데이터셋으로 threshold 에 대한 검증이 필요함

  return filteredSignal
