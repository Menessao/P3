import os
import configparser
import numpy as np
import pandas as pd
from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem.frequencyDomain import frequencyDomain

# --- Configuration & Parameters ---
fname = 'EKARUS'

rMod = 5
gaintol = 1e-2
savedir = f'/raid1/mmenessini/results/{fname}/eb_csv'

min_gain = 0.1
max_gain = 1.0
seeings = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
mags = np.arange(14)

nModes = np.array([350])
maxFreqs = np.array([500, 700, 900])
windSpeeds = np.array([3, 5, 7])
taus = np.array([1.1e-3, 0.8e-3])

binVals = np.arange(2)
nSubapsBin = np.array([38, 19])
nModesBin = np.array([350, 275])

# --- AO System Initialization ---
fao = fourierModel(
    path_ini=f'/raid1/mmenessini/git/P3/p3/aoSystem/parFiles/{fname}.ini',
    display=False,
    verbose=False,
    getErrorBreakDown=True
)

ADU_PER_PHOTON = 1.0

def get_flux(fao, mag: float, B0=10**10):
    thrp = fao.ao.wfs.detector[0].transmittance[0]
    area = np.pi / 4 * fao.ao.tel.D**2 * (1 - fao.ao.tel.obsRatio**2)
    return area * thrp * B0 * 10**(-mag / 2.5) * ADU_PER_PHOTON


ini_path = os.path.join(os.path.dirname(__file__), 'parFiles', f'{fname}.ini')
config = configparser.RawConfigParser()
config.optionxform = str


def update_pars_file(delay=None, gain=None, mag=None, seeing=None, fs=None, Nsubap=None, nModes=None, rMod=None, RON=None, V=None):
    config.read(ini_path)

    if seeing is not None:
        config.set('atmosphere', 'Seeing', str(seeing))

    if V is not None:
        config.set('atmosphere', 'WindSpeed', f'[{5./9.22*V}, {10./9.22*V}, {15./9.22*V}, {20./9.22*V}]')

    if gain is not None:
        config.set('RTC', 'LoopGain_HO', str(gain))
    if delay is not None:
        config.set('RTC', 'LoopDelaySteps_HO', str(delay))
    if fs is not None:
        config.set('RTC', 'SensorFrameRate_HO', str(fs))

    if rMod is not None:
        config.set('sensor_HO', 'Modulation', str(rMod))
    if Nsubap is not None:
        config.set('sensor_HO', 'NumberLenslets', f'[{Nsubap}]')

    try:
        fs_val = fs if fs is not None else float(config.get('RTC', 'SensorFrameRate_HO'))
    except Exception:
        fs_val = float(fao.ao.rtc.holoop.get('rate', fao.ao.rtc.holoop.get('SensorFrameRate_HO', None)))

    if Nsubap is not None:
        nL_val = Nsubap
    else:
        nLens = config.get('sensor_HO', 'NumberLenslets')
        nL_val = float(nLens.strip().lstrip('[').rstrip(']'))

    if mag is not None:
        total_flux = get_flux(fao, mag)
        nSubap_val = nL_val**2 * np.pi / 4 * (1 - fao.ao.tel.obsRatio**2)
        flux_per_sub = total_flux / fs_val / nSubap_val
        config.set('sensor_HO', 'NumberPhotons', f'[{flux_per_sub}]')

    if nModes is not None:
        nActs = int(np.round(np.sqrt(nModes * 4 / np.pi)))
        config.set('DM', 'NumberActuators', f'[{nActs}]')
        D = float(config.get('telescope', 'TelescopeDiameter'))
        pitch = D / nActs
        config.set('DM', 'DmPitchs', f'[{pitch}]')

    if RON is not None:
        config.set('sensor_HO', 'SigmaRON', f'{RON}')

    with open(ini_path, 'w') as fh:
        config.write(fh)


def get_error_breakdown():
    ini_path = f'/raid1/mmenessini/git/P3/p3/aoSystem/parFiles/{fname}.ini'
    fao_inst = fourierModel(path_ini=ini_path, display=False, getErrorBreakDown=True, verbose=False)
    return fao_inst


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
        'nmodes': np.full(shape, np.nan),
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
        arrays['nmodes'][i, j] = row.get('best_nmodes', np.nan)
    return arrays


def optimize_gain_for_frequency(idCenter, freq, mag, seeing, tau, nmodes):
    update_pars_file(mag=mag, seeing=seeing, rMod=rMod, delay=1 + tau * freq, fs=freq, nModes=nmodes)
    a, b = min_gain, max_gain
    if tau * freq >= 2:
        b = 0.7
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


def optimize_gain_and_nmodes_for_frequency(idCenter, freq, mag, seeing, tau, max_modes):
    """
    Optimizes gain at max_modes first. If noise/lag variance is small relative
    to fitting variance, skips lower mode search and returns max_modes immediately.
    Otherwise, steps down in modes of 25 to find the minimal WFE.
    """
    g_max, fao_max = optimize_gain_for_frequency(idCenter, freq, mag, seeing, tau, int(max_modes))
    wfe_max = fao_max.wfeTot[idCenter]

    # Extract component WFEs (RMS)
    fit_err = fao_max.wfeFit
    noise_err = fao_max.wfeN[idCenter]
    lag_err = fao_max.wfeS

    # If combined noise & lag variance is significantly smaller than fitting variance,
    # reducing modes will only INCREASE fitting error and worsen total WFE.
    # We can safely lock max_modes and exit immediately.
    if noise_err < fit_err:
        return g_max, fao_max, int(max_modes)

    # If noise is dominant (faint star), search downwards in steps of 25
    wfe_opt = wfe_max
    g_opt = g_max
    fao_opt = fao_max
    nmodes_opt = int(max_modes)

    candidate_modes = np.arange(max_modes - 25, 0, -25)

    for nmodes_curr in candidate_modes:
        g_curr, fao_curr = optimize_gain_for_frequency(idCenter, freq, mag, seeing, tau, int(nmodes_curr))
        wfe_curr = fao_curr.wfeTot[idCenter]

        if wfe_curr < wfe_opt:
            wfe_opt = wfe_curr
            fao_opt = fao_curr
            g_opt = g_curr
            nmodes_opt = int(nmodes_curr)
        else:
            # WFE started increasing; break early due to convexity
            break

    return g_opt, fao_opt, nmodes_opt

# --- Main Optimization Loops ---
for binVal in binVals:
    print(f'Now using bin: {binVal+1}')
    tau = taus[binVal]
    update_pars_file(Nsubap=nSubapsBin[binVal], nModes=nModesBin[binVal], RON=0.4 + float(binVal) / 10)

    for maxFreq in maxFreqs:
        freqs = np.arange(50, maxFreq + 50, 50)

        for nmodes_base in nModes:
            nmodes_cap = int(np.min((nModesBin[binVal], nmodes_base)))

            for windSpeed in windSpeeds:
                update_pars_file(nModes=nmodes_cap, V=windSpeed)
                binSuffix = f'_bin{binVal+1}'
                csv_path = f'{savedir}/eb_ekarus_rMod{rMod:1.1f}_max{maxFreq}Hz_opt{nmodes_cap}modes_delay{tau*1e+3:1.2f}ms_V{windSpeed}mps' + binSuffix + '.csv'

                existing_df = load_existing_results(csv_path)
                arrays = initialize_result_arrays()
                arrays = populate_vectors_from_df(existing_df, arrays)

                idCenter = fao.ao.src.zenith.argmin()

                for i, seeing in enumerate(seeings):
                    current_opt_freq = freqs[-1]

                    for j, mag in enumerate(mags):
                        if j > 0 and not np.isnan(arrays['freq'][i, j - 1]):
                            current_opt_freq = arrays['freq'][i, j - 1]

                        if not np.isnan(arrays['freq'][i, j]) and not np.isnan(arrays['gain'][i, j]):
                            current_opt_freq = arrays['freq'][i, j]
                            continue

                        print(f'Optimizing parameters for: {seeing}", magnitude = {mag}')
                        best_freq_for_mag = current_opt_freq
                        best_gain_for_mag = None
                        best_fao_for_mag = None
                        best_nmodes_for_mag = None
                        previous_best_wfe = np.inf

                        freq_candidates = freqs[freqs <= current_opt_freq][::-1]

                        for freq in freq_candidates:
                            g_opt, fao_candidate, best_nmodes = optimize_gain_and_nmodes_for_frequency(
                                idCenter, freq, mag, seeing, tau, nmodes_cap
                            )
                            wfe = fao_candidate.wfeTot[idCenter]

                            if wfe < previous_best_wfe:
                                previous_best_wfe = wfe
                                best_freq_for_mag = freq
                                best_gain_for_mag = g_opt
                                best_fao_for_mag = fao_candidate
                                best_nmodes_for_mag = best_nmodes
                                continue
                            else:
                                if best_gain_for_mag is not None:
                                    update_pars_file(
                                        mag=mag,
                                        seeing=seeing,
                                        rMod=rMod,
                                        delay=1 + tau * best_freq_for_mag,
                                        fs=best_freq_for_mag,
                                        gain=best_gain_for_mag,
                                        nModes=best_nmodes_for_mag
                                    )
                                break

                        if best_nmodes_for_mag is not None and best_nmodes_for_mag < nmodes_cap:
                            print(f'\t\tfound {best_nmodes_for_mag} modes to be optimal for magnitude {mag}')

                        if best_fao_for_mag is not None:
                            arrays['fit'][i, j] = best_fao_for_mag.wfeFit
                            arrays['noise'][i, j] = best_fao_for_mag.wfeN[idCenter]
                            arrays['lag'][i, j] = best_fao_for_mag.wfeS
                            arrays['alias'][i, j] = best_fao_for_mag.wfeAl
                            arrays['flux'][i, j] = best_fao_for_mag.ao.wfs.detector[0].nph
                            arrays['gain'][i, j] = best_gain_for_mag
                            arrays['freq'][i, j] = best_freq_for_mag
                            arrays['nmodes'][i, j] = best_nmodes_for_mag
                            current_opt_freq = best_freq_for_mag

                seeing_grid, mag_grid = np.meshgrid(seeings, mags, indexing='ij')

                df_results = pd.DataFrame({
                    'seeing': seeing_grid.flatten(),
                    'magnitude': mag_grid.flatten(),
                    'fitting_error': arrays['fit'].flatten(),
                    'noise_error': arrays['noise'].flatten(),
                    'lag_error': arrays['lag'].flatten(),
                    'aliasing_error': arrays['alias'].flatten(),
                    'best_flux': arrays['flux'].flatten(),
                    'best_gain': arrays['gain'].flatten(),
                    'best_frequency': arrays['freq'].flatten(),
                    'best_nmodes': arrays['nmodes'].flatten(),
                })

                df_results.to_csv(csv_path, index=False)



# for ekarus_config in ekarus_configs:
#     print(f'Simulating configuration: {ekarus_config}')
#     if ekarus_config == 'baseline':
#         tau = 1.5e-3
#         freqs = np.arange(50,550,50)
#         binVals = np.arange(1,dtype=int)
#     elif ekarus_config == 'goal':
#         tau = 1.0e-3
#         freqs = np.arange(50,1050,50)
#         binVals = np.arange(2,dtype=int)
#     elif ekarus_config == 'best':
#         tau = 0.5e-3
#         freqs = np.arange(50,2050,50)
#         binVals = np.arange(4,dtype=int)

#     for binVal in binVals:
#         print(f'Now using bin: {binVal+1}')
#         update_pars_file(Nsubap=nSubapsBin[binVal],nModes=nModesBin[binVal],RON=0.4+float(binVal)/10)
#         csv_path = f'{savedir}/eb_ekarus_{ekarus_config}_rMod{rMod:1.1f}_delay{tau*1e+3:1.2f}ms_bin{binVal+1}.csv'
#         existing_df = load_existing_results(csv_path)
#         arrays = initialize_result_arrays()
#         arrays = populate_vectors_from_df(existing_df, arrays)

#         idCenter = fao.ao.src.zenith.argmin()
#         for i, seeing in enumerate(seeings):
#             current_opt_freq = freqs[-1]
#             for j, mag in enumerate(mags):
#                 if j > 0 and not np.isnan(arrays['freq'][i, j-1]):
#                     current_opt_freq = arrays['freq'][i, j-1]

#                 if not np.isnan(arrays['freq'][i, j]) and not np.isnan(arrays['gain'][i, j]):
#                     current_opt_freq = arrays['freq'][i, j]
#                     continue

#                 print(f'Optimizing parameters for: {seeing}", magnitude = {mag}')
#                 best_freq_for_mag = current_opt_freq
#                 best_gain_for_mag = None
#                 best_fao_for_mag = None
#                 previous_best_wfe = np.inf

#                 freq_candidates = freqs[freqs <= current_opt_freq][::-1]
#                 for freq in freq_candidates:
#                     g_opt, fao_candidate = optimize_gain_for_frequency(idCenter, freq, mag, seeing, tau)
#                     wfe = fao_candidate.wfeTot[idCenter]

#                     if wfe < previous_best_wfe:
#                         previous_best_wfe = wfe
#                         best_freq_for_mag = freq
#                         best_gain_for_mag = g_opt
#                         best_fao_for_mag = fao_candidate
#                         continue
#                     else:
#                         if best_gain_for_mag is not None:
#                             update_pars_file(mag=mag, seeing=seeing, rMod=rMod, delay=1+tau*best_freq_for_mag, fs=best_freq_for_mag, gain=best_gain_for_mag)
#                         break

#                 if best_fao_for_mag is not None:
#                     arrays['fit'][i, j] = best_fao_for_mag.wfeFit
#                     arrays['noise'][i, j] = best_fao_for_mag.wfeN[idCenter]
#                     arrays['lag'][i, j] = best_fao_for_mag.wfeS
#                     arrays['alias'][i, j] = best_fao_for_mag.wfeAl
#                     arrays['flux'][i, j] = best_fao_for_mag.ao.wfs.detector[0].nph
#                     arrays['gain'][i, j] = best_gain_for_mag
#                     arrays['freq'][i, j] = best_freq_for_mag
#                     current_opt_freq = best_freq_for_mag
#                 else:
#                     current_opt_freq = current_opt_freq

#         # --- Save Results to a Pandas DataFrame and Export to CSV ---

#         # Generate coordinate grids that perfectly match the shape of the 2D matrices
#         seeing_grid, mag_grid = np.meshgrid(seeings, mags, indexing='ij')

#         # Construct the structured DataFrame by flattening the 2D arrays
#         df_results = pd.DataFrame({
#             'seeing': seeing_grid.flatten(),
#             'magnitude': mag_grid.flatten(),
#             'fitting_error': arrays['fit'].flatten(),
#             'noise_error': arrays['noise'].flatten(),
#             'lag_error': arrays['lag'].flatten(),
#             'aliasing_error': arrays['alias'].flatten(),
#             'best_flux': arrays['flux'].flatten(),
#             'best_gain': arrays['gain'].flatten(),
#             'best_frequency': arrays['freq'].flatten()
#         })

#         # Save the tidy dataset into a CSV file
#         df_results.to_csv(csv_path, index=False)

# for binVal in binVals:
#     print(f'Now using bin: {binVal+1}')
#     update_pars_file(Nsubap=nSubapsBin[binVal],nModes=nModesBin[binVal],RON=0.4+float(binVal)/10)
#     for tau in taus:
#         csv_path = f'{savedir}/eb_ekarus_rMod{rMod:1.1f}_delay{tau*1e+3:1.2f}ms_bin{binVal+1}.csv'
#         existing_df = load_existing_results(csv_path)
#         arrays = initialize_result_arrays()
#         arrays = populate_vectors_from_df(existing_df, arrays)

#         idCenter = fao.ao.src.zenith.argmin()
#         for i, seeing in enumerate(seeings):
#             current_opt_freq = freqs[-1]
#             for j, mag in enumerate(mags):
#                 if j > 0 and not np.isnan(arrays['freq'][i, j-1]):
#                     current_opt_freq = arrays['freq'][i, j-1]

#                 if not np.isnan(arrays['freq'][i, j]) and not np.isnan(arrays['gain'][i, j]):
#                     current_opt_freq = arrays['freq'][i, j]
#                     continue

#                 print(f'Optimizing parameters for: {seeing}", magnitude = {mag}')
#                 best_freq_for_mag = current_opt_freq
#                 best_gain_for_mag = None
#                 best_fao_for_mag = None
#                 previous_best_wfe = np.inf

#                 freq_candidates = freqs[freqs <= current_opt_freq][::-1]
#                 for freq in freq_candidates:
#                     g_opt, fao_candidate = optimize_gain_for_frequency(idCenter, freq, mag, seeing, tau-(readoutBin[0]-readoutBin[binVal]))
#                     wfe = fao_candidate.wfeTot[idCenter]

#                     if wfe < previous_best_wfe:
#                         previous_best_wfe = wfe
#                         best_freq_for_mag = freq
#                         best_gain_for_mag = g_opt
#                         best_fao_for_mag = fao_candidate
#                         continue
#                     else:
#                         if best_gain_for_mag is not None:
#                             update_pars_file(mag=mag, seeing=seeing, rMod=rMod, delay=1+tau*best_freq_for_mag, fs=best_freq_for_mag, gain=best_gain_for_mag)
#                         break

#                 if best_fao_for_mag is not None:
#                     arrays['fit'][i, j] = best_fao_for_mag.wfeFit
#                     arrays['noise'][i, j] = best_fao_for_mag.wfeN[idCenter]
#                     arrays['lag'][i, j] = best_fao_for_mag.wfeS
#                     arrays['alias'][i, j] = best_fao_for_mag.wfeAl
#                     arrays['flux'][i, j] = best_fao_for_mag.ao.wfs.detector[0].nph
#                     arrays['gain'][i, j] = best_gain_for_mag
#                     arrays['freq'][i, j] = best_freq_for_mag
#                     current_opt_freq = best_freq_for_mag
#                 else:
#                     current_opt_freq = current_opt_freq

#         # --- Save Results to a Pandas DataFrame and Export to CSV ---

#         # Generate coordinate grids that perfectly match the shape of the 2D matrices
#         seeing_grid, mag_grid = np.meshgrid(seeings, mags, indexing='ij')

#         # Construct the structured DataFrame by flattening the 2D arrays
#         df_results = pd.DataFrame({
#             'seeing': seeing_grid.flatten(),
#             'magnitude': mag_grid.flatten(),
#             'fitting_error': arrays['fit'].flatten(),
#             'noise_error': arrays['noise'].flatten(),
#             'lag_error': arrays['lag'].flatten(),
#             'aliasing_error': arrays['alias'].flatten(),
#             'best_flux': arrays['flux'].flatten(),
#             'best_gain': arrays['gain'].flatten(),
#             'best_frequency': arrays['freq'].flatten()
#         })

#         # Save the tidy dataset into a CSV file
#         df_results.to_csv(f'{savedir}/eb_ekarus_rMod{rMod:1.1f}_delay{tau*1e+3:1.2f}ms_bin{binVal+1}.csv', index=False)
