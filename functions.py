import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def get_experimental_spectra(molecule_file, wavelengthMIN, wavelengthMAX, wavelengthNUM):
    """
    Obtain experimental spectral data
    :param wavelengthMIN: Minimum of interpolated wavelength range
    :param wavelengthMAX: Maximum of interpolated wavelength range
    :param wavelengthNUM: Number of points in interpolation
    :param molecule_file: file with experimental spectra of a given molecule
    :return: wavelength, normalized absorption of molecule
    """
    with open(molecule_file) as f:
        data = pd.read_csv(f, delimiter=',')

    wavelength = data.values[:, 0]
    absorption = data.values[:, 1]

    func = interp1d(wavelength, absorption, kind='quadratic')
    wavelength = np.linspace(wavelengthMIN, wavelengthMAX, wavelengthNUM)
    absorption = func(wavelength)
    absorption -= absorption.min()
    absorption *= 100. / absorption.max()
    absorption = savgol_filter(absorption, 5, 3)

    return wavelength, absorption


# ==================================================================================================================== #
#                                                                                                                      #
#                                                  FUNCTIONS FOR PLOTTING                                              #
#   ---------------------------------------------------------------------------------------------------------------    #
# ==================================================================================================================== #


def render_axis(axis, labelSIZE='x-large', labelCOLOR='k', gridCOLOR='r', gridLINE='--'):
    """
    Style plots for better representation
    :param axis: axis to be rendered
    :param labelSIZE: size of ticks in plot
    :param labelCOLOR: color code for labels
    :param gridCOLOR: color code for grid lines
    :param gridLINE: line style for grid lines
    :return:
    """
    """
    Style plots for better representation
    :param axis: axes class of plot
    """
    plt.rc('font', weight='bold')
    axis.tick_params(bottom=True, top=True, left=True, right=True)
    axis.get_xaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelrotation=0, labelsize=labelSIZE)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelcolor=labelCOLOR, labelsize=labelSIZE)
    axis.grid(color=gridCOLOR, linestyle=gridLINE, linewidth=0.5, alpha=0.5, b=None, which='both', axis='both')
    return


def plot_dynamics(axis, x, y, xlabel, ylabel):
    """
    Plot y vs. x with labels ylabel and xlabel on axis.
    :param axis: axis for plotting dynamics
    :param x: x-axis variable
    :param y: y-axis variable
    :param xlabel: label for x-axis
    :param ylabel: label for y-axis
    :return:
    """
    return


def nonuniform_frequency_range_3(params):
    """
    Generation of nonuniform frequency range taylored to the 3d order optical effects
    :param params:
    :return:
    """

    pointsFREQpolarization = np.linspace(-params.combNUM * params.freqDEL, 0.25 * params.combNUM * params.freqDEL, params.combNUM + 1)[:, np.newaxis]
    pointsFREQcomb = np.linspace(-params.combNUM * params.freqDEL, 0.25 * params.combNUM * params.freqDEL, 2 * params.combNUM + 1)[:, np.newaxis]
    resolution = np.linspace(-0.1 * params.freqDEL, 0.1 * params.freqDEL, params.resolutionNUM)

    frequency_12 = 2 * params.omegaM2 - params.omegaM1 + pointsFREQpolarization + resolution
    frequency_21 = 2 * params.omegaM1 - params.omegaM2 + pointsFREQpolarization + resolution

    field1FREQ = params.omegaM1 + pointsFREQcomb + resolution
    field2FREQ = params.omegaM2 + pointsFREQcomb + resolution

    frequency = np.sort(np.hstack([frequency_12.flatten(), frequency_21.flatten(), field1FREQ.flatten(), field2FREQ.flatten()]))
    field1FREQ = np.ascontiguousarray(field1FREQ.flatten())
    field2FREQ = np.ascontiguousarray(field2FREQ.flatten())
    
    return frequency, field1FREQ, field2FREQ