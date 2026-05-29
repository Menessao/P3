import numpy as np
import pandas as pd
from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem.frequencyDomain import frequencyDomain

fname = 'EKARUS'

rMod = 5
tau = 1.5e-3
gaintol = 1e-2
savedir = f'/raid1/mmenessini/results/{fname}/eb_csv'

min_gain = 0.1
max_gain = 1.0
seeings = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
mags = np.array([1, 3, 5, 7, 9, 11, 13])
freqs = np.arange(50, 1050, step=50)

fao = fourierModel(path_ini=f'/raid1/mmenessini/git/P3/p3/aoSystem/parFiles/{fname}.ini',display=False, verbose=True,getErrorBreakDown=True)
fao.verbose = False


def get_flux(fao, mag:float, B0=10**10):
    thrp = fao.ao.wfs.detector[0].transmittance[0]
    area = np.pi/4*fao.ao.tel.D**2*(1-fao.ao.tel.obsRatio**2)
    # thrp *= 1.0-np.exp(-(0.35+0.72*fao.ao.wfs.optics[0].modulation))
    return area * thrp * B0 * 10**(-mag/2.5)

def update_ao_pars(fao, delay=None, gain=None, mag=None, seeing=None, fs=None, Nsubap=None, nModes=None, rMod=None, binning=None):

    if seeing is not None:
        fao.ao.atm.r0 = 0.98*500e-9/seeing*180/np.pi*3600
    
    if gain is not None:
        fao.ao.rtc.holoop['gain'] = gain
    if delay is not None:
        fao.ao.rtc.holoop['delay'] = delay
    if fs is not None:
        fao.ao.rtc.holoop['rate'] = fs
    if rMod is not None:
        fao.ao.wfs.optics[0].modulation = rMod
    if Nsubap is not None:
        fao.ao.wfs.optics[0].nL = Nsubap
        fao.ao.wfs.optics[0].dsub = fao.ao.tel.D/Nsubap

    if binning is not None:
        fao.ao.wfs.detector[0].binning = binning
                
    if mag is not None:
        total_flux = get_flux(fao, mag)
    else:
        total_flux = fao.ao.wfs.detector[0].nph * fao.ao.rtc.holoop['rate'] * fao.ao.wfs.optics[0].nL**2 * np.pi/4
    fs = fao.ao.rtc.holoop['rate']
    nSubap = fao.ao.wfs.optics[0].nL**2 * np.pi/4
    flux_per_sub = total_flux / fs / nSubap
    fao.ao.wfs.detector[0].nph = flux_per_sub

    if nModes is not None:
        nActs = np.sqrt(nModes*4/np.pi)
        fao.ao.dms.nActu1D = nActs
        fao.ao.dms.pitch = fao.ao.tel.D/nActs
        fao.freq = frequencyDomain(
                    fao.ao,
                    nyquistSampling=fao.nyquistSampling,
                    computeFocalAnisoCov=fao.computeFocalAnisoCov,
                    dtype=fao.dtype
                )
    
    fao.ao.wfs.processing.noiseVar = [None]
    

def get_error_breakdown(fao):
    fao.ao.error = False
    fao.getErrorBreakdown = True
    fao.initComputations()
    idCenter = fao.ao.src.zenith.argmin()
    error = {
        'fitting': fao.wfeFit, 
        'lag': fao.wfeS, 
        'noise': fao.wfeN[idCenter], 
        'aliasing': fao.wfeAl, 
        'total': fao.wfeTot[idCenter]
    }
    return error

# Initialize arrays to store results
fitvec = np.zeros([len(seeings), len(mags)])
noisevec = np.zeros([len(seeings), len(mags)])
lagvec = np.zeros([len(seeings), len(mags)])
aliasvec = np.zeros([len(seeings), len(mags)])
fluxvec = np.zeros([len(seeings), len(mags)])
bestgain = np.zeros([len(seeings), len(mags)])
bestfreq = np.zeros([len(seeings), len(mags)])


for i, seeing in enumerate(seeings):
    for j, mag in enumerate(mags):
        print(f'Optimizing parameters for: {seeing}", magnitude = {mag}')
        update_ao_pars(fao, mag=mag, seeing=seeing, rMod=rMod)
        idCenter = fao.ao.src.zenith.argmin()
        min_tot = np.inf
        
        for freq in freqs:
            # --- Ternary Search (Interval Bisection for Optimization) ---
            # Search bounds for the gain variable
            a, b = min_gain, max_gain
            tol = gaintol  # Target precision for the gain optimization
            
            while (b - a) > tol:
                m1 = a + (b - a) / 3
                m2 = b - (b - a) / 3
                
                # Evaluate wavefront error at the first midpoint m1
                update_ao_pars(fao, delay=1+tau*freq, gain=m1, fs=freq)
                get_error_breakdown(fao)
                wfe1 = fao.wfeTot[idCenter]
                
                # Evaluate wavefront error at the second midpoint m2
                update_ao_pars(fao, delay=1+tau*freq, gain=m2, fs=freq)
                get_error_breakdown(fao)
                wfe2 = fao.wfeTot[idCenter]
                
                # Narrow the interval based on which midpoint yields a smaller error
                if wfe1 < wfe2:
                    b = m2  # Minimum is in the lower 2/3 interval [a, m2]
                else:
                    a = m1  # Minimum is in the upper 2/3 interval [m1, b]
            
            # The optimized gain is the midpoint of our converged interval
            g_opt = (a + b) / 2
            
            # Final evaluation at the optimal gain for this specific frequency
            update_ao_pars(fao, delay=1+tau*freq, gain=g_opt, fs=freq)
            eb = get_error_breakdown(fao)
            
            # Global check across all frequencies for the current seeing/mag combo
            if fao.wfeTot[idCenter] < min_tot:
                min_tot = fao.wfeTot[idCenter]
                fitvec[i, j] = fao.wfeFit
                noisevec[i, j] = fao.wfeN[idCenter]
                lagvec[i, j] = fao.wfeS
                aliasvec[i, j] = fao.wfeAl
                fluxvec[i,j] = fao.ao.wfs.detector[0].nph
                bestgain[i, j] = g_opt
                bestfreq[i, j] = freq

# --- Save Results to a Pandas DataFrame and Export to CSV ---

# Generate coordinate grids that perfectly match the shape of the 2D matrices
seeing_grid, mag_grid = np.meshgrid(seeings, mags, indexing='ij')

# Construct the structured DataFrame by flattening the 2D arrays
df_results = pd.DataFrame({
    'seeing': seeing_grid.flatten(),
    'magnitude': mag_grid.flatten(),
    'fitting_error': fitvec.flatten(),
    'noise_error': noisevec.flatten(),
    'lag_error': lagvec.flatten(),
    'aliasing_error': aliasvec.flatten(),
    'best_flux': fluxvec.flatten(),
    'best_gain': bestgain.flatten(),
    'best_frequency': bestfreq.flatten()
})

# Save the tidy dataset into a CSV file
df_results.to_csv(f'{savedir}/eb_ekarus_rMod{rMod:1.1f}_delay{tau*1e+3:1.2f}ms.csv', index=False)