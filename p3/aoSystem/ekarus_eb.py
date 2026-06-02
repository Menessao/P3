import numpy as np
import pandas as pd
from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem.frequencyDomain import frequencyDomain
import configparser
import os

fname = 'EKARUS'

rMod = 5
gaintol = 2e-2
savedir = f'/raid1/mmenessini/results/{fname}/eb_csv'

min_gain = 0.1
max_gain = 1.0
taus = np.array([0.5,1.5,1.0,0.75])*1e-3
seeings = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
mags = np.arange(13)
freqs = np.arange(50, 1050, step=50)

binVals = np.arange(4,dtype=int)
nModesBin = np.array([452,220,100,54])
nSubapsBin = np.array([42,21,14,10.5])

fao = fourierModel(path_ini=f'/raid1/mmenessini/git/P3/p3/aoSystem/parFiles/{fname}.ini',display=False, verbose=False,getErrorBreakDown=True)


def get_flux(fao, mag:float, B0=10**10):
    thrp = fao.ao.wfs.detector[0].transmittance[0]
    area = np.pi/4*fao.ao.tel.D**2*(1-fao.ao.tel.obsRatio**2)
    # thrp *= 1.0-np.exp(-(0.35+0.72*fao.ao.wfs.optics[0].modulation))
    return area * thrp * B0 * 10**(-mag/2.5)

# def update_ao_pars(fao, delay=None, gain=None, mag=None, seeing=None, fs=None, Nsubap=None, nModes=None, rMod=None):

#     if seeing is not None:
#         fao.ao.atm.r0 = 0.98*500e-9/seeing*180/np.pi*3600
    
#     if gain is not None:
#         fao.ao.rtc.holoop['gain'] = gain
#     if delay is not None:
#         fao.ao.rtc.holoop['delay'] = delay
#     if fs is not None:
#         fao.ao.rtc.holoop['rate'] = fs
#     if rMod is not None:
#         fao.ao.wfs.optics[0].modulation = rMod
#     if Nsubap is not None:
#         fao.ao.wfs.optics[0].nL = Nsubap
#         fao.ao.wfs.optics[0].dsub = fao.ao.tel.D/Nsubap
                
#     if mag is not None:
#         total_flux = get_flux(fao, mag)
#     else:
#         total_flux = fao.ao.wfs.detector[0].nph * fao.ao.rtc.holoop['rate'] * fao.ao.wfs.optics[0].nL**2 * np.pi/4
#     fs = fao.ao.rtc.holoop['rate']
#     nSubap = fao.ao.wfs.optics[0].nL**2 * np.pi/4
#     flux_per_sub = total_flux / fs / nSubap
#     fao.ao.wfs.detector[0].nph = flux_per_sub

#     if nModes is not None:
#         nActs = np.sqrt(nModes*4/np.pi)
#         fao.ao.dms.nActu1D = nActs
#         fao.ao.dms.pitch = fao.ao.tel.D/nActs
#         fao.freq = frequencyDomain(
#                     fao.ao,
#                     nyquistSampling=fao.nyquistSampling,
#                     computeFocalAnisoCov=fao.computeFocalAnisoCov,
#                     dtype=fao.dtype
#                 )
    
#     fao.ao.wfs.processing.noiseVar = [None]


ini_path = os.path.join(os.path.dirname(__file__), 'parFiles', f'{fname}.ini')
config = configparser.RawConfigParser()
config.optionxform = str

def update_pars_file(delay=None, gain=None, mag=None, seeing=None, fs=None, Nsubap=None, nModes=None, rMod=None, RON=None):
    # --- Persist changes back to the INI file by overwriting it ---
    config.read(ini_path)

    # Update atmosphere seeing
    if seeing is not None:
        config.set('atmosphere', 'Seeing', str(seeing))

    # Update RTC parameters
    if gain is not None:
        config.set('RTC', 'LoopGain_HO', str(gain))
    if delay is not None:
        config.set('RTC', 'LoopDelaySteps_HO', str(delay))
    if fs is not None:
        config.set('RTC', 'SensorFrameRate_HO', str(fs))

    # Update sensor (high-order) parameters
    if rMod is not None:
        config.set('sensor_HO', 'Modulation', str(rMod))
    if Nsubap is not None:
        # store as a single-element list to match existing file style
        config.set('sensor_HO', 'NumberLenslets', f'[{Nsubap}]')

    # Update number photons per subaperture (NumberPhotons) based on magnitude if provided
    try:
        # determine current fs and nSubap for flux per sub calculation
        fs_val = fs if fs is not None else float(config.get('RTC', 'SensorFrameRate_HO'))
    except Exception:
        fs_val = float(fao.ao.rtc.holoop.get('rate', fao.ao.rtc.holoop.get('SensorFrameRate_HO', None)))

    if Nsubap is not None:
        nL_val = Nsubap
    else:
        # try to read from ini
        nLens = config.get('sensor_HO', 'NumberLenslets')
        nL_val = float(nLens.strip().lstrip('[').rstrip(']'))

    if mag is not None:
        total_flux = get_flux(fao, mag)
        nSubap_val = nL_val**2 * np.pi/4
        flux_per_sub = total_flux / fs_val / nSubap_val
        config.set('sensor_HO', 'NumberPhotons', f'[{flux_per_sub}]')

    # Update DM actuator count if nModes is provided
    if nModes is not None:
        nActs = int(np.round(np.sqrt(nModes*4/np.pi)))
        config.set('DM', 'NumberActuators', f'[{nActs}]')
        D = float(config.get('telescope', 'TelescopeDiameter'))
        pitch = D / nActs
        config.set('DM', 'DmPitchs', f'[{pitch}]')

    if RON is not None:
        config.set('sensor_HO', 'SigmaRON', f'{RON}')

    # Write back (overwrite) the INI file
    with open(ini_path, 'w') as fh:
        config.write(fh)
    

# def get_error_breakdown(fao):
#     fao.ao.error = False
#     fao.getErrorBreakdown = True
#     fao.initComputations()
#     idCenter = fao.ao.src.zenith.argmin()
#     error = {
#         'fitting': fao.wfeFit, 
#         'lag': fao.wfeS, 
#         'noise': fao.wfeN[idCenter], 
#         'aliasing': fao.wfeAl, 
#         'total': fao.wfeTot[idCenter]
#     }
#     return error

def get_error_breakdown():
    ini_path = f'/raid1/mmenessini/git/P3/p3/aoSystem/parFiles/{fname}.ini'
    fao = fourierModel(path_ini=ini_path, display=False, getErrorBreakDown=True, verbose=False)
    return fao


def load_existing_results(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        print(f'Warning: could not load existing results from {csv_path}: {exc}')
        return None


def initialize_result_arrays():
    shape = (len(seeings), len(mags))
    return {
        'fit': np.full(shape, np.nan),
        'noise': np.full(shape, np.nan),
        'lag': np.full(shape, np.nan),
        'alias': np.full(shape, np.nan),
        'flux': np.full(shape, np.nan),
        'gain': np.full(shape, np.nan),
        'freq': np.full(shape, np.nan),
    }


def populate_vectors_from_df(df, arrays):
    if df is None:
        return arrays
    for _, row in df.iterrows():
        try:
            i = int(np.where(seeings == float(row['seeing']))[0][0])
            j = int(np.where(mags == float(row['magnitude']))[0][0])
        except Exception:
            continue
        arrays['fit'][i, j] = row.get('fitting_error', np.nan)
        arrays['noise'][i, j] = row.get('noise_error', np.nan)
        arrays['lag'][i, j] = row.get('lag_error', np.nan)
        arrays['alias'][i, j] = row.get('aliasing_error', np.nan)
        arrays['flux'][i, j] = row.get('best_flux', np.nan)
        arrays['gain'][i, j] = row.get('best_gain', np.nan)
        arrays['freq'][i, j] = row.get('best_frequency', np.nan)
    return arrays


def optimize_gain_for_frequency(idCenter, freq, mag, seeing, tau):
    update_pars_file(mag=mag, seeing=seeing, rMod=rMod, delay=1+tau*freq, fs=freq)
    a, b = min_gain, max_gain
    if tau*freq >= 2:
        b = 0.67
    while (b - a) > gaintol:
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3

        update_pars_file(gain=m1)
        fao1 = get_error_breakdown()
        wfe1 = fao1.wfeTot[idCenter]

        update_pars_file(gain=m2)
        fao2 = get_error_breakdown()
        wfe2 = fao2.wfeTot[idCenter]

        if wfe1 < wfe2:
            b = m2
        else:
            a = m1

    g_opt = (a + b) / 2
    update_pars_file(gain=g_opt)
    fao_opt = get_error_breakdown()
    return g_opt, fao_opt


for binVal in binVals:
    print(f'Now using bin: {binVal+1}')
    update_pars_file(Nsubap=nSubapsBin[binVal],nModes=nModesBin[binVal],RON=0.4+float(binVal)/10)
    for tau in taus:
        csv_path = f'{savedir}/eb_ekarus_rMod{rMod:1.1f}_delay{tau*1e+3:1.2f}ms_bin{binVal+1}.csv'
        existing_df = load_existing_results(csv_path)
        arrays = initialize_result_arrays()
        arrays = populate_vectors_from_df(existing_df, arrays)

        idCenter = fao.ao.src.zenith.argmin()
        for i, seeing in enumerate(seeings):
            current_opt_freq = freqs[-1]
            for j, mag in enumerate(mags):
                if j > 0 and not np.isnan(arrays['freq'][i, j-1]):
                    current_opt_freq = arrays['freq'][i, j-1]

                if not np.isnan(arrays['freq'][i, j]) and not np.isnan(arrays['gain'][i, j]):
                    current_opt_freq = arrays['freq'][i, j]
                    continue

                print(f'Optimizing parameters for: {seeing}", magnitude = {mag}')
                best_freq_for_mag = current_opt_freq
                best_gain_for_mag = None
                best_fao_for_mag = None
                previous_best_wfe = np.inf

                freq_candidates = freqs[freqs <= current_opt_freq][::-1]
                for freq in freq_candidates:
                    g_opt, fao_candidate = optimize_gain_for_frequency(idCenter, freq, mag, seeing, tau)
                    wfe = fao_candidate.wfeTot[idCenter]

                    if wfe < previous_best_wfe:
                        previous_best_wfe = wfe
                        best_freq_for_mag = freq
                        best_gain_for_mag = g_opt
                        best_fao_for_mag = fao_candidate
                        continue
                    else:
                        if best_gain_for_mag is not None:
                            update_pars_file(mag=mag, seeing=seeing, rMod=rMod, delay=1+tau*best_freq_for_mag, fs=best_freq_for_mag, gain=best_gain_for_mag)
                        break

                if best_fao_for_mag is not None:
                    arrays['fit'][i, j] = best_fao_for_mag.wfeFit
                    arrays['noise'][i, j] = best_fao_for_mag.wfeN[idCenter]
                    arrays['lag'][i, j] = best_fao_for_mag.wfeS
                    arrays['alias'][i, j] = best_fao_for_mag.wfeAl
                    arrays['flux'][i, j] = best_fao_for_mag.ao.wfs.detector[0].nph
                    arrays['gain'][i, j] = best_gain_for_mag
                    arrays['freq'][i, j] = best_freq_for_mag
                    current_opt_freq = best_freq_for_mag
                else:
                    current_opt_freq = current_opt_freq

        # --- Save Results to a Pandas DataFrame and Export to CSV ---

        # Generate coordinate grids that perfectly match the shape of the 2D matrices
        seeing_grid, mag_grid = np.meshgrid(seeings, mags, indexing='ij')

        # Construct the structured DataFrame by flattening the 2D arrays
        df_results = pd.DataFrame({
            'seeing': seeing_grid.flatten(),
            'magnitude': mag_grid.flatten(),
            'fitting_error': arrays['fit'].flatten(),
            'noise_error': arrays['noise'].flatten(),
            'lag_error': arrays['lag'].flatten(),
            'aliasing_error': arrays['alias'].flatten(),
            'best_flux': arrays['flux'].flatten(),
            'best_gain': arrays['gain'].flatten(),
            'best_frequency': arrays['freq'].flatten()
        })

        # Save the tidy dataset into a CSV file
        df_results.to_csv(f'{savedir}/eb_ekarus_rMod{rMod:1.1f}_delay{tau*1e+3:1.2f}ms_bin{binVal+1}.csv', index=False)