#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:17:21 2020

@author: isabel
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.fftpack import next_fast_len


_logger = logging.getLogger(__name__)


def  kramer_kronig_egt(x, y, N_ZLP = 1, iterations = 1, plot = False):
    #TODO: change variables to correct values
    #N_ZLP = 1 #1 als prob
    E = 200 #keV =e0??
    beta = 30 #mrad
    ri = 3 #refractive index
    nloops = iterations #number of iterations
    delta = 0.3 #eV stability parameter (0.1eV-0.5eV)
    m_0 = 511.06
    a_0 = 5.29 #nm, borhradius
    
    l = len(x)
    
    #extraplolate
    semi_inf = 2**math.floor(math.log2(l)+1)*4 #waar komt dit getal vandaan?? egerton: nn
    EELsample = np.zeros(semi_inf) #egerton: ssd
    EELsample[:l] = y
    I = EELsample #egerton: d
    ddeltaE = (x[-1]-x[0])/(l-1) #energy/channel, egerton: epc
    deltaE = np.linspace(0, semi_inf-1, semi_inf)*ddeltaE + x[0] #egerton: e
    deltaE[deltaE==0] = 1E-14 #very small number
    
    gamma = 1 + E/m_0
    T = E*(1+E/(2*m_0))/gamma**2 #kin energy? egerton: t=mv^2/2
    rk_0 = 2590*gamma*(2*T/m_0)**0.5 #TODO: waar komt 2590 vandaan????
    tgt = E * (2*m_0 + E)/ (m_0 + E)
    tgt = 2*gamma*T
    
    
    
    v = (2*E/m_0 *(1+E/(2*m_0))/gamma**2)**0.5
    theta_E = deltaE/tgt
    log_term = np.log(1+(beta/theta_E)**2)
    
    
    plt.figure()
    fig, axs = plt.subplots(1)
    axs.set_title("eps over loops")
    for i in range(nloops):
        plt.figure()
        plt.plot(deltaE, I)
        plt.title("intensity iteration " +str(i))
        N_S = np.sum(I)*ddeltaE #integral over ssd
        I_ac = I/log_term
        I_ac_over_deltaE_int = np.sum(I_ac/deltaE)*ddeltaE
        
        K = I_ac_over_deltaE_int/ (math.pi/2) / (1-ri**-2) #normilized sum over I_ac/E, egerton: rk
        a_0pi2 = 332.5 #TODO ??????????? WHYY, factor 10???
        t_nm = K * a_0pi2 *T/ N_ZLP
        
        """       
        K = np.sum(Im/eaxis)*ddeltaE #(Im / eaxis).sum(axis=axis.index_in_array, keepdims=True) \
        #* axis.scale
        K = (K / (np.pi / 2) / (1 - 1. / n ** 2))
        # K = (K / (np.pi / 2) / (1 - 1. / n ** 2)).reshape(
        #    np.insert(K.shape, axis.index_in_array, 1))
        # Calculate the thickness only if possible and required
        #if zlp is not None and (full_output is True or
        #                        iterations > 1):
        te = (332.5 * K * ke / i0)
        """
        
        
        
        
        
        N_rat = N_S/N_ZLP #t/lambda blijkbaar, egerton: tol
        lambd = t_nm/N_rat
        
        Im_eps = I_ac/K #Im(-1/eps), egerton: imreps
        S_p = 2*np.imag(np.fft.fft(Im_eps)) / semi_inf #waarom /l ???? gek..., sine transform p(t)
        sgn = np.ones(semi_inf)
        half = math.floor(semi_inf/2)
        sgn[-half:] *= -1
        C_q = sgn*S_p
        Re_eps = np.fft.fft(C_q) #wrs daarom /l: geen inverse maar nog een keer FT
        
        #TODO WHY? correct function for reflected tail (?) #different from book, why?
        Re_eps_mid = np.real(Re_eps[half])
        cor = Re_eps_mid/2*np.power(half/np.linspace(semi_inf-1, half, half),2)
        Re_eps[:half] += 1 - cor
        Re_eps[half:] = 1 + cor #completely overwrite??
        
        eps_sq = np.power(Re_eps, 2) + np.power(Im_eps, 2)
        eps1 = Re_eps/eps_sq
        eps2 = Im_eps/eps_sq
        eps = eps1+ 1j*eps2
        
        axs.plot(deltaE[:2*l], eps1[:2*l], label = "eps1, i="+str(i))
        axs.plot(deltaE[:2*l], eps2[:2*l], label = "eps2, i="+str(i))
        
        
        #surface iter
        srf_eps_term = np.imag(-4/(1+eps)) - Im_eps #egerton: srfelf
        #print("check surface eps factor:", srf_eps_term[330:340], (4 * eps2 / ((eps1 + 1) ** 2 + eps2 ** 2) - Im_eps)[330:340])
        adep = tgt/(deltaE + delta) * np.arctan(beta/theta_E) + beta*1E-3/(beta**2+theta_E**2) #waarom opeens beta mili maken? 
        S_s = 2000*K/rk_0/t_nm*adep*srf_eps_term #TODO 2000: 2*1000 ???, reken termen na
        plt.plot(deltaE[:2*l], S_s[:2*l]/np.max(np.abs(S_s)), label = "S_s, i=" + str(i))
        I = EELsample - S_s
    axs.legend()
    plt.legend()
    
    return eps, t_nm
""" HS def
Srfelf = 4 * e2 / ((e1 + 1) ** 2 + e2 ** 2) - Im
            #adep = (tgt / (eaxis + delta) *
            #        np.arctan(beta * tgt / axis.axis) -
            #        beta / 1000. /
            #        (beta ** 2 + axis.axis ** 2. / tgt ** 2))
            adep = (tgt / (eaxis + delta) *
                    np.arctan(beta * tgt / eaxis) -
                    beta / 1000. /
                    (beta ** 2 + eaxis ** 2. / tgt ** 2))
            Srfint = 2000 * K * adep * Srfelf / rk0 / te * ddeltaE #axis.scale
"""


def kramers_kronig_hs(delatE, I_EELS,
                            zlp=None,
                            iterations=1,
                            n=None,
                            t=None,
                            delta=0.5,
                            full_output=True):
    r"""Calculate the complex
    dielectric function from a single scattering distribution (SSD) using
    the Kramers-Kronig relations.

    It uses the FFT method as in [1]_.  The SSD is an
    EELSSpectrum instance containing SSD low-loss EELS with no zero-loss
    peak. The internal loop is devised to approximately subtract the
    surface plasmon contribution supposing an unoxidized planar surface and
    neglecting coupling between the surfaces. This method does not account
    for retardation effects, instrumental broading and surface plasmon
    excitation in particles.

    Note that either refractive index or thickness are required.
    If both are None or if both are provided an exception is raised.

    Parameters
    ----------
    zlp: {None, number, Signal1D}
        ZLP intensity. It is optional (can be None) if `t` is None and `n`
        is not None and the thickness estimation is not required. If `t`
        is not None, the ZLP is required to perform the normalization and
        if `t` is not None, the ZLP is required to calculate the thickness.
        If the ZLP is the same for all spectra, the integral of the ZLP
        can be provided as a number. Otherwise, if the ZLP intensity is not
        the same for all spectra, it can be provided as i) a Signal1D
        of the same dimensions as the current signal containing the ZLP
        spectra for each location ii) a BaseSignal of signal dimension 0
        and navigation_dimension equal to the current signal containing the
        integrated ZLP intensity.
    iterations: int
        Number of the iterations for the internal loop to remove the
        surface plasmon contribution. If 1 the surface plasmon contribution
        is not estimated and subtracted (the default is 1).
    n: {None, float}
        The medium refractive index. Used for normalization of the
        SSD to obtain the energy loss function. If given the thickness
        is estimated and returned. It is only required when `t` is None.
    t: {None, number, Signal1D}
        The sample thickness in nm. Used for normalization of the
         to obtain the energy loss function. It is only required when
        `n` is None. If the thickness is the same for all spectra it can be
        given by a number. Otherwise, it can be provided as a BaseSignal
        with signal dimension 0 and navigation_dimension equal to the
        current signal.
    delta : float
        A small number (0.1-0.5 eV) added to the energy axis in
        specific steps of the calculation the surface loss correction to
        improve stability.
    full_output : bool
        If True, return a dictionary that contains the estimated
        thickness if `t` is None and the estimated surface plasmon
        excitation and the spectrum corrected from surface plasmon
        excitations if `iterations` > 1.

    Returns
    -------
    eps: DielectricFunction instance
        The complex dielectric function results,

            .. math::
                \epsilon = \epsilon_1 + i*\epsilon_2,

        contained in an DielectricFunction instance.
    output: Dictionary (optional)
        A dictionary of optional outputs with the following keys:

        ``thickness``
            The estimated  thickness in nm calculated by normalization of
            the SSD (only when `t` is None)

        ``surface plasmon estimation``
           The estimated surface plasmon excitation (only if
           `iterations` > 1.)

    Raises
    ------
    ValuerError
        If both `n` and `t` are undefined (None).
    AttribureError
        If the beam_energy or the collection semi-angle are not defined in
        metadata.

    Notes
    -----
    This method is based in Egerton's Matlab code [1]_ with some
    minor differences:

    * The wrap-around problem when computing the ffts is workarounded by
      padding the signal instead of substracting the reflected tail.

    .. [1] Ray Egerton, "Electron Energy-Loss Spectroscopy in the Electron
       Microscope", Springer-Verlag, 2011.

    """
    output = {}
    """
    if iterations == 1:
        # In this case s.data is not modified so there is no need to make
        # a deep copy.
        s = self.isig[0.:]
    else:
        s = self.isig[0.:].deepcopy()

    sorig = self.isig[0.:]
    # Avoid singularity at 0
    if s.axes_manager.signal_axes[0].axis[0] == 0:
        s = s.isig[1:]
        sorig = self.isig[1:]
    """
    # Constants and units
    me = 511.06

    # Mapped parameters
    #self._are_microscope_parameters_missing(
    #    ignore_parameters=['convergence_angle'])
    e0 = 200 #  keV
    beta =30 #mrad

    #axis = s.axes_manager.signal_axes[0]
    eaxis = deltaE #axis.axis.copy()
    eaxis[eaxis==0] = 1E-14 #very small number, avoid singularities
    y = I_EELS
    l = len(eaxis)
    """
    if isinstance(zlp, hyperspy.signal.BaseSignal):
        if (zlp.axes_manager.navigation_dimension ==
                self.axes_manager.navigation_dimension):
            if zlp.axes_manager.signal_dimension == 0:
                i0 = zlp.data
            else:
                i0 = zlp.integrate1D(axis.index_in_axes_manager).data
        else:
            raise ValueError('The ZLP signal dimensions are not '
                             'compatible with the dimensions of the '
                             'low-loss signal')
        # The following prevents errors if the signal is a single spectrum
        if len(i0) != 1:
            i0 = i0.reshape(
                np.insert(i0.shape, axis.index_in_array, 1))
    elif isinstance(zlp, numbers.Number):
        i0 = zlp
    else:
        raise ValueError('The zero-loss peak input is not valid, it must be\
                         in the BaseSignal class or a Number.')
    """
    i0 = N_ZLP
    """
    if isinstance(t, hyperspy.signal.BaseSignal):
        if (t.axes_manager.navigation_dimension ==
                self.axes_manager.navigation_dimension) and (
                t.axes_manager.signal_dimension == 0):
            t = t.data
            t = t.reshape(
                np.insert(t.shape, axis.index_in_array, 1))
        else:
            raise ValueError('The thickness signal dimensions are not '
                             'compatible with the dimensions of the '
                             'low-loss signal')
    elif isinstance(t, np.ndarray) and t.shape and t.shape != (1,):
        raise ValueError("thickness must be a HyperSpy signal or a number,"
                         " not a numpy array.")
    """
    
    # Slicer to get the signal data from 0 to axis.size
    #slicer = s.axes_manager._get_data_slice(
    #    [(axis.index_in_array, slice(None, axis.size)), ])

    # Kinetic definitions
    ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
    tgt = e0 * (2 * me + e0) / (me + e0)
    rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)

    for io in range(iterations):
        # Calculation of the ELF by normalization of the SSD
        # Norm(SSD) = Imag(-1/epsilon) (Energy Loss Funtion, ELF)

        # We start by the "angular corrections"
        Im = y / (np.log(1 + (beta * tgt / eaxis) ** 2)) #/ ddeltaE#axis.scale
        if n is None and t is None:
            raise ValueError("The thickness and the refractive index are "
                             "not defined. Please provide one of them.")
        elif n is not None and t is not None:
            raise ValueError("Please provide the refractive index OR the "
                             "thickness information, not both")
        elif n is not None:
            # normalize using the refractive index.
            K = np.sum(Im/eaxis)*ddeltaE #(Im / eaxis).sum(axis=axis.index_in_array, keepdims=True) \
                #* axis.scale
            K = (K / (np.pi / 2) / (1 - 1. / n ** 2))
            # K = (K / (np.pi / 2) / (1 - 1. / n ** 2)).reshape(
            #    np.insert(K.shape, axis.index_in_array, 1))
            # Calculate the thickness only if possible and required
            #if zlp is not None and (full_output is True or
            #                        iterations > 1):
            te = (332.5 * K * ke / i0)
            if full_output is True:
                output['thickness'] = te
        elif t is not None:
            if zlp is None:
                raise ValueError("The ZLP must be provided when the  "
                                 "thickness is used for normalization.")
            # normalize using the thickness
            K = t * i0 / (332.5 * ke)
            te = t
        Im = Im / K

        # Kramers Kronig Transform:
        # We calculate KKT(Im(-1/epsilon))=1+Re(1/epsilon) with FFT
        # Follows: D W Johnson 1975 J. Phys. A: Math. Gen. 8 490
        # Use an optimal FFT size to speed up the calculation, and
        # make it double the closest upper value to workaround the
        # wrap-around problem.
        #esize = optimal_fft_size(2 * delteE.size)
        esize = next_fast_len(2*l) #2**math.floor(math.log2(l)+1)*4
        #q = -2 * np.fft.fft(Im, esize,
        #                    axis.index_in_array).imag / esize
        q = -2 * np.fft.fft(Im, esize).imag / esize
        

        #q[slicer] *= -1
        q[:l] *= -1
        q = np.fft.fft(q)#, axis=axis.index_in_array)
        # Final touch, we have Re(1/eps)
        #Re = q[slicer].real + 1
        Re = q[:l].real + 1

        # Egerton does this to correct the wrap-around problem, but in our
        # case this is not necessary because we compute the fft on an
        # extended and padded spectrum to avoid this problem.
        # Re=real(q)
        # Tail correction
        # vm=Re[axis.size-1]
        # Re[:(axis.size-1)]=Re[:(axis.size-1)]+1-(0.5*vm*((axis.size-1) /
        #  (axis.size*2-arange(0,axis.size-1)))**2)
        # Re[axis.size:]=1+(0.5*vm*((axis.size-1) /
        #  (axis.size+arange(0,axis.size)))**2)

        # Epsilon appears:
        #  We calculate the real and imaginary parts of the CDF
        e1 = Re / (Re ** 2 + Im ** 2)
        e2 = Im / (Re ** 2 + Im ** 2)

        if iterations > 1 and zlp is not None:
            # Surface losses correction:
            #  Calculates the surface ELF from a vaccumm border effect
            #  A simulated surface plasmon is subtracted from the ELF
            Srfelf = 4 * e2 / ((e1 + 1) ** 2 + e2 ** 2) - Im
            #adep = (tgt / (eaxis + delta) *
            #        np.arctan(beta * tgt / axis.axis) -
            #        beta / 1000. /
            #        (beta ** 2 + axis.axis ** 2. / tgt ** 2))
            adep = (tgt / (eaxis + delta) *
                    np.arctan(beta * tgt / eaxis) -
                    beta / 1000. /
                    (beta ** 2 + eaxis ** 2. / tgt ** 2))
            Srfint = 2000 * K * adep * Srfelf / rk0 / te * ddeltaE #axis.scale
            y = I_EELS - Srfint
            _logger.debug('Iteration number: %d / %d', io + 1, iterations)
            if iterations == io + 1 and full_output is True:
                #sp = sorig._deepcopy_with_new_data(Srfint)
                #sp.metadata.General.title += (
                #    " estimated surface plasmon excitation.")
                output['S_s'] = Srfint
                #del sp
            del Srfint

    eps = (e1 + e2 * 1j)
    del y
    del I_EELS
    #eps.set_signal_type("DielectricFunction")
    #eps.metadata.General.title = (self.metadata.General.title +
    #                              'dielectric function '
    #                              '(from Kramers-Kronig analysis)')
    #if eps.tmp_parameters.has_item('filename'):
    #    eps.tmp_parameters.filename = (
    #        self.tmp_parameters.filename +
    #        '_CDF_after_Kramers_Kronig_transform')
    if 'thickness' in output:
        # As above,prevent errors if the signal is a single spectrum
        """
        if len(te) != 1:
            te = te[self.axes_manager._get_data_slice(
                    [(axis.index_in_array, 0)])]
        thickness = eps._get_navigation_signal(data=te)
        thickness.metadata.General.title = (
            self.metadata.General.title + ' thickness '
            '(calculated using Kramers-Kronig analysis)')
        """
        output['thickness'] = te
    if full_output is False:
        return eps
    else:
        return eps, output


def  kramer_kronig_compare(x, y, N_ZLP = 1, iterations = 1, plot = False):
    #TODO: change variables to correct values
    #N_ZLP = 1 #1 als prob
    #%% EG
    E = 200 #keV =e0??
    beta = 30 #mrad
    ri = 3 #refractive index
    nloops = iterations #number of iterations
    delta = 0.5 #eV stability parameter (0.1eV-0.5eV)
    m_0 = 511.06
    a_0 = 5.29 #nm, borhradius
    
    l = len(x)
    
    
    #%%HS
    # Constants and units
    me = 511.06

    # Mapped parameters
    #self._are_microscope_parameters_missing(
    #    ignore_parameters=['convergence_angle'])
    e0 = 200 #  keV
    beta =30 #mrad

    #axis = s.axes_manager.signal_axes[0]
    eaxis = x #axis.axis.copy()
    eaxis[eaxis==0] = 1E-14 #very small number, avoid singularities
    I_EELS = y
    l = len(eaxis)
    
    i0 = N_ZLP
    
    
    
    
    #%%EG
    #extraplolate
    semi_inf = 2**math.floor(math.log2(l)+1)*4 #waar komt dit getal vandaan?? egerton: nn
    EELsample = np.zeros(semi_inf) #egerton: ssd
    EELsample[0:l] = y
    I = EELsample #egerton: d
    ddeltaE = (x[-1]-x[0])/(l-1) #energy/channel, egerton: epc
    deltaE = np.linspace(0, semi_inf-1, semi_inf)*ddeltaE + x[0] #egerton: e
    deltaE[deltaE==0] = 1E-14 #very small number, avoid singularities

    #dont extrapolate
    #I = y
    #deltaE = x
    
    gamma = 1 + E/m_0
    T = E*(1+E/(2*m_0))/gamma**2 #kin energy? egerton: t=mv^2/2
    rk_0 = 2590*gamma*(2*T/m_0)**0.5 #TODO: waar komt 2590 vandaan????
    tgt = E * (2*m_0 + E)/ (m_0 + E)
    tgt = 2*gamma*T
    v = (2*E/m_0 *(1+E/(2*m_0))/gamma**2)**0.5
    theta_E = deltaE/tgt
    
    #%%HS
    ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
    tgt_hs = e0 * (2 * me + e0) / (me + e0)
    rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)
    
    
    
    #%%EG
    log_term = np.log(1+(beta/theta_E)**2)
    
    
    plt.figure()
    fig, axs = plt.subplots(1)
    axs.set_title("eps over loops")
    for i in range(nloops):
        plt.figure()
        plt.plot(deltaE, I)
        plt.title("intensity iteration " +str(i))
        N_S = np.sum(I)*ddeltaE #integral over ssd
        I_ac = I/log_term
        I_ac_over_deltaE_int = np.sum(I_ac/deltaE)*ddeltaE
        
        K = I_ac_over_deltaE_int/ (math.pi/2) / (1-ri**-2) #normilized sum over I_ac/E, egerton: rk
        a_0pi2 = 332.5 #TODO ??????????? WHYY, factor 10???
        t_nm = K * a_0pi2 *T/ N_ZLP
        Im_eps = I_ac/K #Im(-1/eps), egerton: imreps

        #%%HS
        Im_or = y / (np.log(1 + (beta * tgt_hs / eaxis) ** 2)) #/ ddeltaE#axis.scale     
        K_hs = np.sum(Im_or/eaxis)*ddeltaE #(Im / eaxis).sum(axis=axis.index_in_array, keepdims=True) \
        #* axis.scale
        K_hs = (K_hs / (np.pi / 2) / (1 - 1. / ri ** 2))
        # K = (K / (np.pi / 2) / (1 - 1. / n ** 2)).reshape(
        #    np.insert(K.shape, axis.index_in_array, 1))
        # Calculate the thickness only if possible and required
        #if zlp is not None and (full_output is True or
        #                        iterations > 1):
        te = (332.5 * K_hs * ke / i0)
        Im = Im_or / K
        
        
        #%%EG
        
        #N_rat = N_S/N_ZLP #t/lambda blijkbaar, egerton: tol
        #lambd = t_nm/N_rat
        
        S_p = 2*np.imag(np.fft.fft(Im_eps)) / semi_inf #waarom /l ???? gek..., sine transform p(t)
        sgn = np.ones(semi_inf)
        half = math.floor(semi_inf/2)
        sgn[-half:] *= -1
        C_q = sgn*S_p
        Re_eps = np.fft.fft(C_q) #wrs daarom /l: geen inverse maar nog een keer FT
        
        #TODO WHY? correct function for reflected tail (?) #different from book, why?
        Re_eps_mid = np.real(Re_eps[half])
        cor = Re_eps_mid/2*np.power(half/np.linspace(semi_inf-1, half, half),2)
        Re_eps[:half] += 1 - cor
        Re_eps[half:] = 1 + cor #completely overwrite??
        
        eps_sq = np.power(Re_eps, 2) + np.power(Im_eps, 2)
        eps1 = Re_eps/eps_sq
        eps2 = Im_eps/eps_sq
        eps = eps1+ 1j*eps2
        
        axs.plot(deltaE[:2*l], eps1[:2*l], label = "eps1, i="+str(i))
        axs.plot(deltaE[:2*l], eps2[:2*l], label = "eps2, i="+str(i))
        
        #%%HS
        esize = 2**math.floor(math.log2(l)+1)*4
        #q = -2 * np.fft.fft(Im, esize,
        #                    axis.index_in_array).imag / esize
        p = -2 * np.fft.fft(Im, esize).imag / esize
        
        p_sl = p
        #q[slicer] *= -1
        p_sl[:l] *= -1
        q = np.fft.fft(p_sl)#, axis=axis.index_in_array)
        # Final touch, we have Re(1/eps)
        #Re = q[slicer].real + 1
        Re = q[:l].real + 1
        
        e1 = Re / (Re ** 2 + Im ** 2)
        e2 = Im / (Re ** 2 + Im ** 2)
        
        
        
        #surface iter
        srf_eps_term = np.imag(-4/(1+eps)) - Im_eps #egerton: srfelf
        #print("check surface eps factor:", srf_eps_term[330:340], (4 * eps2 / ((eps1 + 1) ** 2 + eps2 ** 2) - Im_eps)[330:340])
        adep = tgt/(deltaE + delta) * np.arctan(beta/theta_E) + beta*1E-3/(beta**2+theta_E**2) #waarom opeens beta mili maken? 
        S_s = 2000*K/rk_0/t_nm*adep*srf_eps_term #TODO 2000: 2*1000 ???, reken termen na
        plt.plot(deltaE[:2*l], S_s[:2*l]/np.max(np.abs(S_s)), label = "S_s, i=" + str(i))
        I = EELsample - S_s
    axs.legend()
    plt.legend()
    
    return eps, t_nm




"""
MoS2_eloss = np.loadtxt('MoS2_data/MoS2_eloss.txt')
MoS2_eps = np.loadtxt('MoS2_data/MoS2_eps.txt')

deltaE =  MoS2_eloss[:,0]
eloss_x = MoS2_eloss[:,1]
eloss_z = MoS2_eloss[:,2]

#eps_MoS2_x, t_x = kramer_kronig_egt(deltaE, eloss_x, plot=True)#, method = 2)
#eps_MoS2_z, t_z = kramer_kronig_egt(deltaE, eloss_z, plot=True)#, method = 2)

l = len(deltaE)
ddeltaE = (deltaE[-1]-deltaE[0])/l #energy/channel, egerton: epc
deltaE = np.linspace(0, len(eps_MoS2_x)-1, len(eps_MoS2_x))*ddeltaE + deltaE[0] #egerton: e


plt.figure()
plt.title("dieelctric function MoS2, x direction")
plt.plot(deltaE, np.real(eps_MoS2_x), label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_MoS2_x), label = r'$\varepsilon_2$')
plt.legend()
#%%
plt.figure()
plt.title("dieelctric function MoS2, z direction")
plt.plot(deltaE, np.real(eps_MoS2_z), label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_MoS2_z), label = r'$\varepsilon_2$')
plt.legend()



plt.figure()
plt.title("dieelctric function MoS2, x direction")
plt.plot(deltaE[:l], np.real(eps_MoS2_x)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE[:l], np.imag(eps_MoS2_x)[:l], label = r'$\varepsilon_2$')
plt.legend()


plt.figure()
plt.title("dieelctric function MoS2, z direction")
plt.plot(deltaE[:l], np.real(eps_MoS2_z)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE[:l], np.imag(eps_MoS2_z)[:l], label = r'$\varepsilon_2$')
plt.legend()


plt.figure()
plt.title("dieelctric function MoS2, x direction, optocal values")
plt.plot(MoS2_eps[:,0][:l], MoS2_eps[:,1][:l], label = r'$\varepsilon_1$')
plt.plot(MoS2_eps[:,0][:l], MoS2_eps[:,2][:l], label = r'$\varepsilon_2$')
plt.legend()


plt.figure()
plt.title("dieelctric function MoS2, z direction, optocal values")
plt.plot(MoS2_eps[:,0][:l], MoS2_eps[:,3][:l], label = r'$\varepsilon_1$')
plt.plot(MoS2_eps[:,0][:l], MoS2_eps[:,4][:l], label = r'$\varepsilon_2$')
plt.legend()
"""
#%%
xs = total_replicas['x14'].values
Nx = np.sum(total_replicas['x14']==total_replicas['x14'][0])
nx = int(len(total_replicas)/Nx)

x_14 = df_sample.iloc[5].x_shifted
y_14 = df_sample.iloc[5].y

y_ZLPs = total_replicas.match14.values

l = len(x_14)

ZLP_14 = np.zeros((Nx, l))


for i in range(Nx):
    y_ZLP = np.zeros(l)
    y_ZLP[:nx] = y_ZLPs[i*nx:(i+1)*nx]
    ZLP_14[i,:] = y_ZLP

deltaE = x_14
ZLP = np.average(ZLP_14, axis = 0)
EELsample = y_14-ZLP

ddeltaE = (np.max(deltaE)-np.min(deltaE))/l
N_ZLP = np.sum(ZLP) * ddeltaE


#EGERTON
iterations = 1
eps_14, t_14 = kramer_kronig_egt(deltaE, EELsample, N_ZLP = N_ZLP, iterations= iterations, plot=True)#, method = 2)

print("found thickness: ", round(t_14,3), "nm, with ", iterations, " iterations")

plt.figure()
plt.title("dieelctric function spectrum 14, egerton, no iterations: " + str(iterations))
plt.plot(deltaE, np.real(eps_14)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_14)[:l], label = r'$\varepsilon_2$')
plt.legend()

iterations = 2
eps_14, t_14 = kramer_kronig_egt(deltaE, EELsample, N_ZLP = N_ZLP, iterations= iterations, plot=True)#, method = 2)

print("found thickness: ", round(t_14,3), "nm, with ", iterations, " iterations")

plt.figure()
plt.title("dieelctric function spectrum 14, egerton, no iterations: " + str(iterations))
plt.plot(deltaE, np.real(eps_14)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_14)[:l], label = r'$\varepsilon_2$')
plt.legend()



#HYPERSPY
iterations = 1
eps_14_hs, t_14_hs = kramers_kronig_hs(deltaE, EELsample, iterations= iterations, zlp = N_ZLP, n =3)#, method = 2)

plt.figure()
plt.title("dieelctric function spectrum 14, hyperspy, no iterations: " + str(iterations))
plt.plot(deltaE, np.real(eps_14_hs)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_14_hs)[:l], label = r'$\varepsilon_2$')
plt.legend()
print("found thickness: ", round(t_14_hs["thickness"],3), "nm, with ", iterations, " iterations")



iterations = 4
eps_14_hs, t_14_hs = kramers_kronig_hs(deltaE, EELsample, iterations= iterations, zlp = N_ZLP, n =3)#, method = 2)

plt.figure()
plt.title("dieelctric function spectrum 14, hyperspy, no iterations: " + str(iterations))
plt.plot(deltaE, np.real(eps_14_hs)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_14_hs)[:l], label = r'$\varepsilon_2$')
plt.legend()
print("found thickness: ", round(t_14_hs["thickness"],3), "nm, with ", iterations, " iterations")

plt.figure()
plt.plot(deltaE, EELsample, label = "$I_{EELS}$")
plt.plot(deltaE, t_14_hs["S_s"], label = "$S_s$, estimation")
plt.title("Inelastic and surface scattering of spectrum 14")
plt.legend()


#%%
"""
#COMPARISON
iterations = 1
eps_14, t_14 = kramer_kronig_compare(deltaE, EELsample, N_ZLP = N_ZLP, iterations= iterations, plot=True)#, method = 2)

print("found thickness: ", round(t_14,3), "nm, with ", iterations, " iterations")

plt.figure()
plt.title("dieelctric function spectrum 14, egerton, no iterations: " + str(iterations))
plt.plot(deltaE, np.real(eps_14)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_14)[:l], label = r'$\varepsilon_2$')
plt.legend()

iterations = 2
eps_14, t_14 = kramer_kronig_compare(deltaE, EELsample, N_ZLP = N_ZLP, iterations= iterations, plot=True)#, method = 2)

print("found thickness: ", round(t_14,3), "nm, with ", iterations, " iterations")

plt.figure()
plt.title("dieelctric function spectrum 14, egerton, no iterations: " + str(iterations))
plt.plot(deltaE, np.real(eps_14)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_14)[:l], label = r'$\varepsilon_2$')
plt.legend()
"""