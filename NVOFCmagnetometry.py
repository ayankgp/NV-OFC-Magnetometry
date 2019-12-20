#!/usr/bin/env python

"""
CalculateSpectra.py:

Class containing C calls for spectra calculation and discriminating OFC-pulse generation.
Plots results obtained from C calls.
"""

__author__ = "Ayan Chattopadhyay"
__affiliation__ = "Princeton University"


# ---------------------------------------------------------------------------- #
#                      LOADING PYTHON LIBRARIES AND FILES                      #
# ---------------------------------------------------------------------------- #

from multiprocessing import cpu_count
from types import MethodType, FunctionType
import time
from itertools import product
from functions import *
from wrapper import *


class ADict(dict):
    """
    Appended Dictionary: where keys can be accessed as attributes: A['*'] --> A.*
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class Spectra:
    """
    Calculates the linear absorption spectra and fits molecular parameters in the process
    """

    def __init__(self, spectra_variables, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.rho_0 = np.ascontiguousarray(spectra_variables.rho_0)
        self.rho = np.ascontiguousarray([spectra_variables.rho_0.copy() for _ in range(spectra_variables.molNUM)])
        self.spectra_time = np.linspace(0, spectra_variables.spectra_timeAMP, spectra_variables.spectra_timeDIM)
        self.spectra_field = np.zeros(spectra_variables.spectra_timeDIM, dtype=np.complex)
        self.gammaMATRIXpopd = np.ascontiguousarray(self.gammaMATRIXpopd)
        self.gammaMATRIXdephasing = np.ascontiguousarray(self.gammaMATRIXdephasing)
        self.muMATRIX = np.ascontiguousarray(self.muMATRIX)
        self.energies = np.ascontiguousarray(self.energies)
        self.levelsNUM = spectra_variables.levelsNUM
        self.spectra_absTOTAL = np.ascontiguousarray(np.zeros((spectra_variables.molNUM, len(self.spectra_frequencies[0]))))
        self.spectra_absDIST = np.ascontiguousarray(np.empty((spectra_variables.molNUM, spectra_variables.ensembleNUM, len(self.spectra_frequencies[0]))))
        # ------------------------------------------------------------------------------------------------------------ #
        #                       DECLARE NEW SET OF VARIABLES FOR ALL N MOLECULES IN ENSEMBLE                           #
        # ------------------------------------------------------------------------------------------------------------ #

    def create_molecule(self, spectra_molecule, indices):
        spectra_molecule.levelsNUM = self.levelsNUM
        spectra_molecule.energies = self.energies[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.gammaMATRIXpopd = self.gammaMATRIXpopd[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.gammaMATRIXdephasing = self.gammaMATRIXdephasing[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_frequencies = self.spectra_frequencies[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_freqDIM = len(self.spectra_frequencies[indices])
        spectra_molecule.muMATRIX = self.muMATRIX[indices].ctypes.data_as(POINTER(c_complex))
        spectra_molecule.spectra_field = self.spectra_field.ctypes.data_as(POINTER(c_complex))
        spectra_molecule.rho = self.rho[indices].ctypes.data_as(POINTER(c_complex))
        spectra_molecule.spectra_absTOTAL = self.spectra_absTOTAL[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_absDIST = self.spectra_absDIST[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_absREF = self.spectra_absREF[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.levelsVIBR = self.levelsVIBR[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.levels = self.levels[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.probabilities = self.probabilities[indices].ctypes.data_as(POINTER(c_double))
        return

    def create_parameters(self, spectra_parameters, variables):
        spectra_parameters.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        spectra_parameters.levelsNUM = self.levelsNUM
        spectra_parameters.excitedNUM = variables.excitedNUM
        spectra_parameters.spectra_time = self.spectra_time.ctypes.data_as(POINTER(c_double))
        spectra_parameters.spectra_timeAMP = variables.spectra_timeAMP
        spectra_parameters.spectra_timeDIM = len(self.spectra_time)
        spectra_parameters.spectra_fieldAMP = variables.spectra_fieldAMP
        spectra_parameters.threadNUM = variables.threadNUM
        spectra_parameters.ensembleNUM = variables.ensembleNUM
        spectra_parameters.guessLOWER = variables.guessLOWER.ctypes.data_as(POINTER(c_double))
        spectra_parameters.guessUPPER = variables.guessUPPER.ctypes.data_as(POINTER(c_double))
        spectra_parameters.iterMAX = variables.iterMAX
        return

    def fit_spectra(self, variables):
        spectra_parameters = SpectraParameters()
        self.create_parameters(spectra_parameters, variables)

        molENSEMBLE = [SpectraMolecule() for _ in range(variables.molNUM)]
        for molINDX in range(variables.molNUM):
            self.create_molecule(molENSEMBLE[molINDX], molINDX)
            CalculateSpectra(molENSEMBLE[molINDX], spectra_parameters)


class OFC:
    """
    Calculates the ofc system of the molecule
    """

    def __init__(self, ofc_variables, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.muMATRIX = np.ascontiguousarray(self.muMATRIX)

        self.energies = np.ascontiguousarray(self.energies)
        self.levelsNUM = ofc_variables.levelsNUM
        self.frequency, self.field1FREQ, self.field2FREQ = nonuniform_frequency_range_3(ofc_variables)
        self.polarizationEMPTY = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationFIELD = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationINDEX = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationMOLECULE = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALEMPTY = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALFIELD = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)

        # ------------------------------------------------------------------------------------------------------------ #
        #                       DECLARE NEW SET OF VARIABLES FOR ALL N MOLECULES IN ENSEMBLE                           #
        # ------------------------------------------------------------------------------------------------------------ #

    def create_ofc_molecule(self, ofc_molecule, indices):
        ofc_molecule.levelsNUM = self.levelsNUM
        ofc_molecule.energies = self.energies[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.levelsVIBR = self.levelsVIBR[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.levels = self.levels[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.gammaMATRIX = np.ascontiguousarray(self.gammaMATRIX).ctypes.data_as(POINTER(c_double))
        ofc_molecule.muMATRIX = self.muMATRIX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationINDEX = self.polarizationINDEX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationMOLECULE = self.polarizationMOLECULE[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.probabilities = self.probabilities[indices].ctypes.data_as(POINTER(c_double))
        return

    def create_ofc_parameters(self, ofc_parameters, ofc_variables):
        ofc_parameters.excitedNUM = ofc_variables.excitedNUM
        ofc_parameters.ensembleNUM = ofc_variables.ensembleNUM
        ofc_parameters.freqNUM = len(self.frequency)
        ofc_parameters.combNUM = ofc_variables.combNUM
        ofc_parameters.resolutionNUM = ofc_variables.resolutionNUM
        ofc_parameters.frequency = self.frequency.ctypes.data_as(POINTER(c_double))
        ofc_parameters.combGAMMA = ofc_variables.combGAMMA
        ofc_parameters.freqDEL = ofc_variables.freqDEL
        ofc_parameters.termsNUM = ofc_variables.termsNUM
        ofc_parameters.indices = np.ones(3).ctypes.data_as(POINTER(c_int))
        ofc_parameters.modulations = np.zeros(3).ctypes.data_as(POINTER(c_double))
        ofc_parameters.envelopeWIDTH = ofc_variables.envelopeWIDTH
        ofc_parameters.envelopeCENTER = ofc_variables.envelopeCENTER
        return

    def calculate_ofc_system(self, ofc_variables):
        ofc_parameters = OFCParameters()
        self.create_ofc_parameters(ofc_parameters, ofc_variables)

        molENSEMBLE = [OFCMolecule() for _ in range(ofc_variables.molNUM)]
        for molINDX in range(ofc_variables.molNUM):
            self.create_ofc_molecule(molENSEMBLE[molINDX], molINDX)
            for i, modulations in enumerate(list(product(*(3 * [[ofc_variables.omegaM1, ofc_variables.omegaM2]])))):
                if i in range(8):
                    print(i, modulations)
                    m, n, v = (1, 2, 3)
                    ofc_parameters.indices[0] = m
                    ofc_parameters.indices[1] = n
                    ofc_parameters.indices[2] = v
                    ofc_parameters.modulations = np.asarray(modulations).ctypes.data_as(POINTER(c_double))
                    mu_product = self.muMATRIX[molINDX][0, m] * self.muMATRIX[molINDX][m, n] * self.muMATRIX[molINDX][n, v] * self.muMATRIX[molINDX][v, 0]
                    self.polarizationMOLECULE[molINDX][:] = 0.
                    CalculateNLResponse(molENSEMBLE[molINDX], ofc_parameters)
                    self.polarizationMOLECULE[molINDX] *= mu_product
                    for ensembleINDX in range(ofc_variables.ensembleNUM):
                        if (i == 1) or (i == 6):
                            self.polarizationEMPTY[molINDX][ensembleINDX] += self.polarizationMOLECULE[molINDX][ensembleINDX]
                        else:
                            self.polarizationFIELD[molINDX][ensembleINDX] += self.polarizationMOLECULE[molINDX][ensembleINDX]


            self.polarizationFIELD[molINDX] *= 1j * np.pi ** 2
            self.polarizationEMPTY[molINDX] *= 1j * np.pi ** 2

            for ensembleINDX in range(ofc_variables.ensembleNUM):
                self.polarizationTOTALEMPTY[molINDX] += (self.polarizationEMPTY[molINDX])[ensembleINDX]*self.probabilities[molINDX][ensembleINDX]
                self.polarizationTOTALFIELD[molINDX] += (self.polarizationFIELD[molINDX])[ensembleINDX]*self.probabilities[molINDX][ensembleINDX]


if __name__ == '__main__':

    import pickle

    # --------------------------------------------------------- #
    #                       LIST OF CONSTANTS                   #
    # --------------------------------------------------------- #

    energyFACTOR = 1./27.211385
    timeFACTOR = 2.418884e-5
    wavelength2freqFACTOR = 1239.84
    cm_inv2evFACTOR = 1.23984e-4

    # ------------------------------------------------------------------------------------------ #
    #                       MOLECULAR CONSTANTS, VARIABLES, VECTORS & MATRICES                   #
    # ------------------------------------------------------------------------------------------ #

    molNUM = 1
    levelsNUM = 4
    ensembleNUM = 22
    groundNUM = 2
    excitedNUM = levelsNUM - groundNUM

    # ------------------ MOLECULAR ENERGY LEVEL STRUCTURE ------------------ #

    energies = np.empty((molNUM, levelsNUM))
    levelMIN = [558, 760, 780]
    levelMAX = [785, 870, 880]
    levels = [
        np.asarray(wavelength2freqFACTOR * energyFACTOR / np.linspace(levelMIN[i], levelMAX[i], excitedNUM * ensembleNUM)[::-1])
        for i in range(molNUM)]

    vibrations = [1600, 1610, 1590]
    levelsVIBR = [np.asarray([0, vibrations[i]]) * energyFACTOR * cm_inv2evFACTOR for i in range(molNUM)]

    # ------------------------ INITIAL DENSITY MATRIX ---------------------- #

    rho_0 = np.zeros((levelsNUM, levelsNUM), dtype=np.complex)
    rho_0[0, 0] = 1 + 0j

    # ------------------ TRANSITION DIPOLE MOMENT AND DECAY PARAMETERS ------------------ #

    MU = [2., 2., 2.]
    gammaPOPD = [2.418884e-8, 2.518884e-8, 2.618884e-8]
    gammaELEC = [2.418884e-4, 2.518884e-4, 2.618884e-4]
    gammaVIBR = [2.418884e-6, 2.518884e-6, 2.618884e-6]

    muMATRIX = [MU[i]*np.ones((levelsNUM, levelsNUM), dtype=np.complex) for i in range(molNUM)]
    [np.fill_diagonal(muMATRIX[i], 0j) for i in range(molNUM)]

    gammaMATRIXpopd = [np.ones((levelsNUM, levelsNUM)) * gammaPOPD[i] for i in range(molNUM)]
    gammaMATRIXdephasing = [np.ones((levelsNUM, levelsNUM)) * gammaVIBR[i] for i in range(molNUM)]
    for i in range(molNUM):
        np.fill_diagonal(gammaMATRIXpopd[i], 0.0)
        gammaMATRIXpopd[i] = np.tril(gammaMATRIXpopd[i]).T
        np.fill_diagonal(gammaMATRIXdephasing[i], 0.0)
        for j in range(groundNUM):
            for k in range(groundNUM, levelsNUM):
                gammaMATRIXdephasing[i][j, k] = gammaELEC[i]
                gammaMATRIXdephasing[i][k, j] = gammaELEC[i]
    gammaMATRIX = gammaMATRIXdephasing
    for k in range(molNUM):
        for n in range(levelsNUM):
            for m in range(levelsNUM):
                for i in range(levelsNUM):
                    gammaMATRIX[k][n][m] += 0.5 * (gammaMATRIXpopd[k][n][i] + gammaMATRIXpopd[k][m][i])


    # ------------------ SPECTRA FITTING PROBABILITIES  ------------------ #

    probabilities = np.asarray(
        [
            [0.00833965, 0.012831, 0.0141135, 0.0237958, 0.0374953, 0.0323565, 0.073392, 0.0697636, 0.104259, 0.145788,
             0.200278, 0.263304, 0.31854, 0.376063, 0.486379, 0.649557, 0.879894, 0.923599, 0.488522, 0.0950145,
             0.0500529, 0.00654009],
            [0.00833965, 0.012831, 0.0141135, 0.0237958, 0.0374953, 0.0323565, 0.073392, 0.0697636, 0.104259, 0.145788,
             0.200278, 0.263304, 0.31854, 0.376063, 0.486379, 0.649557, 0.879894, 0.923599, 0.488522, 0.0950145,
             0.0500529, 0.00654009],
            [0.00833965, 0.012831, 0.0141135, 0.0237958, 0.0374953, 0.0323565, 0.073392, 0.0697636, 0.104259, 0.145788,
             0.200278, 0.263304, 0.31854, 0.376063, 0.486379, 0.649557, 0.879894, 0.923599, 0.488522, 0.0950145,
             0.0500529, 0.00654009]
        ]
    )

    guessLOWER = np.zeros(ensembleNUM)
    guessUPPER = np.ones(ensembleNUM)

    # ---------------------------------------------------------------------------------------------------------------- #
    #              READ csv-DATA FILES INTO WAVELENGTH & ABSORPTION MATRICES: (SIZE) N x wavelengthNUM                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    data_protein_files_list = ['DataFP/mCHERRY.csv', 'DataFP/FusionRED.csv', 'DataFP/DsRed.csv']
    protein_plot_colors = ['r', 'b', 'k']

    wavelengthNUM = 100
    wavelengths = np.empty([molNUM, wavelengthNUM])
    absorptions = np.empty_like(wavelengths)
    frequencies = np.empty_like(wavelengths)
    for i in range(molNUM):
        wavelengths[i], absorptions[i] = get_experimental_spectra(data_protein_files_list[i], 560, 790, wavelengthNUM)
        frequencies[i] = wavelength2freqFACTOR * energyFACTOR / wavelengths[i]

    # -------------------------------------------#
    #              OFC PARAMETERS                #
    # -------------------------------------------#

    combNUM = 3000
    resolutionNUM = 5
    omegaM1 = 5e-2 * timeFACTOR
    omegaM2 = 7e-2 * timeFACTOR
    freqDEL = 12e-2 * timeFACTOR
    combGAMMA = 1e-7 * timeFACTOR
    termsNUM = 5
    envelopeWIDTH = 60000
    envelopeCENTER = 1125

    SystemArgs = dict(
        gammaMATRIXpopd=gammaMATRIXpopd,
        gammaMATRIXdephasing=gammaMATRIXdephasing,
        gammaMATRIX=gammaMATRIX,
        muMATRIX=muMATRIX,
        energies=energies,
        levelsVIBR=levelsVIBR,
        levels=levels,
        probabilities=probabilities,
        spectra_wavelengths=np.ascontiguousarray(wavelengths),
        spectra_frequencies=np.ascontiguousarray(frequencies),
        spectra_absREF=np.ascontiguousarray(absorptions),
    )

    SystemVars = ADict(
        molNUM=molNUM,
        levelsNUM=levelsNUM,
        excitedNUM=excitedNUM,
        ensembleNUM=ensembleNUM,
        threadNUM=cpu_count(),
        rho_0=rho_0,
        spectra_timeAMP=10000,
        spectra_timeDIM=1000,
        spectra_fieldAMP=8e-6,
        guessLOWER=guessLOWER,
        guessUPPER=guessUPPER,
        iterMAX=1,
        combNUM=combNUM,
        resolutionNUM=resolutionNUM,
        omegaM1=omegaM1,
        omegaM2=omegaM2,
        combGAMMA=combGAMMA,
        freqDEL=freqDEL,
        termsNUM=termsNUM,
        envelopeWIDTH=envelopeWIDTH,
        envelopeCENTER=envelopeCENTER
    )
    start = time.time()

    system = Spectra(SystemVars, **SystemArgs)
    system.fit_spectra(SystemVars)

    print('TIME ELAPSED FOR SPECTRA CALCULATION:', time.time() - start, 'seconds')

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                               PLOT ABSORPTION SPECTRA FIT                                        #
    # ---------------------------------------------------------------------------------------------------------------- #

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    for i in range(SystemVars.molNUM):
        ax.plot(system.spectra_wavelengths[i], system.spectra_absREF[i], protein_plot_colors[i], linestyle='--')
        ax.plot(system.spectra_wavelengths[i], system.spectra_absTOTAL[i], protein_plot_colors[(i+1) % 3])
        ax.set_xlim(system.spectra_wavelengths[i].min(), system.spectra_wavelengths[i].max())
    render_axis(ax, gridLINE='-')

    start = time.time()

    system = OFC(SystemVars, **SystemArgs)
    system.calculate_ofc_system(SystemVars)

    print('TIME ELAPSED FOR OFC RESPONSE CALCULATION:', time.time() - start, 'seconds')

    omega1MOD = system.field1FREQ[:, np.newaxis]
    omega2MOD = system.field2FREQ[:, np.newaxis]

    omegaCOMB = (SystemVars.freqDEL * np.arange(-SystemVars.combNUM, SystemVars.combNUM))[np.newaxis, :]
    gaussian = 1 + 0 * np.exp(-(np.linspace(-SystemVars.combNUM, SystemVars.combNUM, 2 * SystemVars.combNUM)
                                + SystemVars.envelopeCENTER) ** 2 / (2. * SystemVars.envelopeWIDTH ** 2))[np.newaxis, :]
    field1 = (gaussian * SystemVars.combGAMMA / ((omega1MOD - SystemVars.omegaM1 - omegaCOMB) ** 2 + SystemVars.combGAMMA ** 2)).sum(axis=1)
    field2 = (gaussian * SystemVars.combGAMMA / ((omega2MOD - SystemVars.omegaM2 - omegaCOMB) ** 2 + SystemVars.combGAMMA ** 2)).sum(axis=1)

    for molINDX in range(molNUM):
        fig, axes = plt.subplots(nrows=molNUM, ncols=1, sharex=True)
        polMAX = max(np.abs(system.polarizationTOTALEMPTY[molINDX]).max(), np.abs(system.polarizationTOTALFIELD[molINDX]).max())
        with open("polDATA" + str(molINDX) + ".pickle", "wb") as output_file:
            pickle.dump(
                {
                    "field1FREQ": field1 * polMAX / field1.max(),
                    "field2FREQ": field2 * polMAX / field2.max(),
                    "polEMPTY": np.abs(system.polarizationTOTALEMPTY[molINDX]),
                    "polFIELD": np.abs(system.polarizationTOTALFIELD[molINDX])
                },
                output_file
            )
        axes.plot(system.field1FREQ / SystemVars.freqDEL, field1 * polMAX / field1.max(), 'm', alpha=0.4)
        axes.plot(system.field2FREQ / SystemVars.freqDEL, field2 * polMAX / field1.max(), 'g', alpha=0.4)
        axes.plot(system.frequency / SystemVars.freqDEL, np.abs(system.polarizationTOTALEMPTY[molINDX]), 'r', linewidth=1., alpha=0.7, label='EMPTY')
        axes.plot(system.frequency / SystemVars.freqDEL, np.abs(system.polarizationTOTALFIELD[molINDX]), 'b', linewidth=1., alpha=0.7, label='OVERLAP')

    plt.show()