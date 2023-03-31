# builtin
import os
import sys
from pathlib import Path
import itertools
import functools
import argparse
import logging
import json
import dataclasses
import re

# external
import numpy as np
import pandas as pd
from pyteomics import mzml
from pyteomics import mass as pmass
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from tqdm import tqdm

# local
from .utils import qsnip, snip, nearest, where_nearest, margined, n_tag, \
    empirical_cdf, int_hist, find, cosine_similarity, scalar_proj, \
    plot_qcut_boxplot, scatter_with_kde
from .iso_pred import proton_mass
from .iso_pred import peptide_isotope_peaks as _peptide_isotope_peaks
from .modifications import AA_COMP, translate_maxquant_modified_sequence


#### Logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w',
    force=True,
)


#### Methods for processing sequence properties

def peptide_monoiso_mz(sequence, charge):
    return pmass.calculate_mass(
        sequence = sequence,
        charge = charge,
        aa_comp=AA_COMP
    )


def peptide_monoiso_mass(sequence):
    return pmass.calculate_mass(
        sequence = sequence,
        aa_comp=AA_COMP
    )

@functools.lru_cache(maxsize=1024)
def peptide_isotope_peaks(sequence):
    return _peptide_isotope_peaks(
        sequence,
        num_peaks=3,
        mean_combined=True,
        aa_comp=AA_COMP
    )


#### Extraction helper methods
@dataclasses.dataclass
class Parameters:
    CENTROIDING_MIN_PROFILE_PEAK_INTENSITY              : float = 100
    CENTROIDING_MIN_CENTROIDED_PEAK_INTENSITY           : float = 500
    
    CALIBRATION_MAX_ABS_MZ_OFFSET                       : float = 0.05 # [Da]
    CALIBRATION_MAX_RETENTION_LENGTH                    : float = 5.0 # [min]
    CALIBRATION_MINIMUM_REQUIRED_PEPTIDE_COUNT          : int   = 5

    EXTRACTION_MAX_CHARGE                               : int   = 5

    # Below parameters are chosen during extraction if set to None 
    EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH              : float = None # [Da]
    EXTRACTION_MINIMUM_MZ_OFFSET_FOR_NO_MATCH           : float = None # [Da]
    EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS   : float = None # [Da]
    EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH       : float = None


def process_parameters(parameters=None):
    if parameters is None:
        parameters = {}
    
    parameters = Parameters(**parameters)
    return parameters


def preprocessing_for_output(
        output_dir,
        save_plots,
    ):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_plots:
        savefig_dir = Path(output_dir) / 'fig'
        savefig_dir.mkdir(parents=True, exist_ok=True)
    else:
        savefig_dir = None
        
    fh = logging.FileHandler(output_dir / ('out' + '.log'))
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    ))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    return savefig_dir


def compute_ion_spec_pairs(ion_elution_intervals, rt_array, parameters):
    """
    Creates dataframe of ion spectrum pairs given ion elution times. Ion
    spectrum pairs are pairs of ions and spectrum ids where the ion is 
    (confidently) known to be in the spectrum. These are computed from the
    ion elution times, which denote the start and finish times of identified
    ions. An ion is defined as a peptide charge pair.
    
    Parameters
    ----------
    ion_elution_intervals : DataFrame
        A dataframe of start and finish times of identified ions. The dataframe 
        should have four columns: sequence, charge, start, and finish.
    
    rt_array : list or array-like
        A list containing the retention times of the spectrums.
        
    Returns
    -------
    ion_spec_pairs : DataFrame
        A collection of ion spectrum pairs. DataFrame has three columns:
        'sequence', 'charge', and 'spec_id'.
    """
    ion_spec_pairs = []
    for index, row in ion_elution_intervals.iterrows():
        sequence = row['sequence']
        charge = row['charge']
        start = row['start']
        finish = row['finish']
        if finish - start < parameters.CALIBRATION_MAX_RETENTION_LENGTH:
            spec_ids = find((rt_array > start) & (rt_array < finish))
            ion_spec_pairs.extend([
                (sequence, charge, spec_id)
                for spec_id in spec_ids
            ])
            
    ion_spec_pairs = pd.DataFrame(
        ion_spec_pairs,
        columns = ['sequence', 'charge', 'spec_id']
    )
    return ion_spec_pairs
    

def compute_peptide_spec_pairs(ion_elution_intervals, rt_array, parameters):
    """
    Creates dataframe of ion spectrum pairs given peptide elution times.
    Peptide spectrum pairs are pairs of peptide and spectrum ids where a
    peptide is (confidently) known to be in the spectrum. These are computed
    from the ion elution times, which denote the start and finish times of
    identified ions. An ion is defined as a peptide charge pair.
    
    Parameters
    ----------
    ion_elution_intervals : DataFrame
        A dataframe of start and finish times of identified ions. The dataframe 
        should have four columns: sequence, charge, start, and finish.
    
    rt_array : list or array-like
        A list containing the retention times of the spectrums.
        
    Returns
    -------
    ion_spec_pairs : DataFrame
        A collection of peptide spectrum pairs. DataFrame has two columns:
        'sequence' and 'spec_id'.
    """
    
    peptide_elution_intervals = \
        ion_elution_intervals.groupby('sequence').agg({
            'start' : 'min',
            'finish' : 'max'
        }).reset_index()
    
    peptide_spec_pairs = []
    for index, row in peptide_elution_intervals.iterrows():
        sequence = row['sequence']
        start = row['start']
        finish = row['finish']
        if finish - start < parameters.CALIBRATION_MAX_RETENTION_LENGTH:
            spec_ids = find((rt_array > start) & (rt_array < finish))
            peptide_spec_pairs.extend([
                (sequence, spec_id)
                for spec_id in spec_ids
            ])
            
    peptide_spec_pairs = pd.DataFrame(
        peptide_spec_pairs,
        columns = ['sequence', 'spec_id']
        )
    return peptide_spec_pairs
    

@functools.lru_cache
def ion_target_mz(sequence, charge):
    isotope_peaks = peptide_isotope_peaks(sequence)
    isotope_masses = isotope_peaks[0]
    
    mass_to_search = np.hstack([
        isotope_masses,
        isotope_masses[0] - (isotope_masses[1] - isotope_masses[0]),
        isotope_masses[0] - (isotope_masses[1] - isotope_masses[0])/2,
        (isotope_masses[1:] + isotope_masses[:-1])/2
    ])
    mz_to_search = mass_to_search / charge + proton_mass
    
    return mz_to_search


def compute_monoiso_mz_offsets(spectrums, ion_spec_pairs):
    mz_offsets = []
    for sequence, charge, spec_id in tqdm(zip(
            ion_spec_pairs['sequence'],
            ion_spec_pairs['charge'],
            ion_spec_pairs['spec_id']
        )):
        spectrum = spectrums[spec_id]
        mz = peptide_monoiso_mz(sequence, charge)    
        offset = nearest(spectrum[0], mz, is_sorted=True) - mz
        mz_offsets.append(offset)
    return np.array(mz_offsets)
    

def compute_fwhm_roughly(arr):
    return 2*np.quantile(np.abs(arr - np.median(arr)), 0.50)


#### Extraction subprocesses

def extract_ion_elution_intervals_from_maxquant(
        maxquant_txt_dir,
        raw_file_name=None,
        show_plots=False,
        plot_manager=None
    ):
    """
    Extracts ion start and finish elution times from MaxQuant output.
    
    Ions identified through MaxQuant are extracted, along with their stated
    start and finish elution times.
    
    
    Parameters
    ----------
    maxquant_txt_dir : path
        A path to the txt output folder of MaxQuant. The txt folder should
        contain evidence.txt and summary.txt
        
    Returns
    -------
    ion_elution_intervals : DataFrame
        A dataframe of start and finish times of known ions. The dataframe 
        has four columns: sequence, charge, start, and finish.
        
    Notes
    -----
    The summary.txt is only used to determine the fixed modifications used in
    the MaxQuant search. For ions with multiple readings in evidence.txt, the
    start and finish elution time is taken as the min and max, respectively,
    over all the readings.
    """
    evidence_file = Path(maxquant_txt_dir) / 'evidence.txt'
    summary_file = Path(maxquant_txt_dir) / 'summary.txt'
    
    evidence = pd.read_csv(evidence_file, sep='\t', low_memory=False)
    if raw_file_name:
        evidence = evidence[evidence['Raw file'] == raw_file_name]
        evidence = evidence.reset_index(drop=True)
        
    evidence = evidence[~evidence[[
        'Calibrated retention time start',
        'Calibrated retention time finish',
        'Retention time calibration'
    ]].isna().any(axis=1)]
    
    evidence = \
        evidence[evidence['Type'] != 'MULTI-MATCH'].reset_index(drop=True)
        
    logger.info(f'Evidence has {len(evidence)} rows after filtering.')
    
    df = pd.concat({
        'sequence' : evidence['Modified sequence'],
        'charge' : evidence['Charge'],
        'start' : (
            evidence['Calibrated retention time start']
            - evidence['Retention time calibration']
        ),
        'finish' : (
            evidence['Calibrated retention time finish']
            - evidence['Retention time calibration']
        )
    }, axis=1)
    
    df = df.groupby(['sequence','charge']).agg({
        'start' : 'min',
        'finish' : 'max'
    })
    ion_elution_intervals = df.reset_index()
   
    summary = pd.read_csv(summary_file, sep='\t')
    summary = \
        summary[summary['Raw file'].str.startswith(raw_file_name)].iloc[0]
    fixed_modifications = summary['Fixed modifications'].split(';')
    
    sequences = ion_elution_intervals['sequence']
    sequences = [
        translate_maxquant_modified_sequence(sequence, fixed_modifications)
        for sequence in sequences
    ]
    ion_elution_intervals['sequence'] = sequences
    
    if show_plots:
        sanity_plots_for_evidence(
            evidence,
            plot_manager=plot_manager
        )

    return ion_elution_intervals


def sanity_plots_for_evidence(
        evidence,
        plot_manager=None
    ):
    """
    The following plots checks that ions of the same peptide elutes at similar
    retention times.
    """
    
    if plot_manager is None:
        plot_manager = PlotManager()
        
    try:
        with plot_manager as pm:
            plt.plot(pm.rt_array, pm.tic_array)
            plt.xlabel('retention time [min]')
            plt.ylabel('tic [a.u.]')
            plt.title('TIC vs RT')
            pm.savefig_label('')
    except:
        pass
        
    ion_min_start_time = pd.concat({
        'sequence' : evidence['Modified sequence'],
        'charge' : evidence['Charge'],
        'start' : evidence['Calibrated retention time start'] \
            - evidence['Retention time calibration']
        }, axis=1).groupby(['sequence','charge']).min().squeeze()
        
    ion_max_finish_time = pd.concat({
        'sequence' : evidence['Modified sequence'],
        'charge' : evidence['Charge'],
        'finish' : evidence['Calibrated retention time finish'] \
            - evidence['Retention time calibration']
        }, axis=1).groupby(['sequence','charge']).max().squeeze()
    
    with plot_manager as pm:
        values = ion_max_finish_time - ion_min_start_time
        snipped = qsnip(values, 0, 0.95)
        plt.hist(snipped, 50)
        plt.xlim(margined(snipped, [0.05, 0]))
        plt.xlabel('retention length [min]')
        plt.ylabel('frequency')
        plt.title(
            'Ion retention length\n(based on evidence file)'
            + n_tag(len(ion_max_finish_time - ion_min_start_time))
        )
        plt.gca().twinx()
        empirical_cdf(
            ion_max_finish_time - ion_min_start_time,
            color='orange', density=True
        )
        plt.ylabel('cdf')
        pm.savefig_label('')
        
    with plot_manager as pm:
        values = ion_min_start_time.groupby('sequence').std().dropna()
        snipped = qsnip(values, 0, 0.95)
        plt.hist(snipped, 120)
        plt.xlim(margined(snipped, [0.05, 0]))
        plt.xlabel('standard deviation [min]')
        plt.ylabel('frequency')
        plt.title(
            'Stdev of ion start time per peptide\n(based on evidence file)'
            + n_tag(len(values))
        )
        plt.gca().twinx()
        empirical_cdf(values, color='orange', density=True)
        plt.ylim([0,1])
        plt.ylabel('cdf')
        pm.savefig_label('')
    
    with plot_manager as pm:    
        values = ion_max_finish_time.groupby('sequence').std().dropna()
        snipped = qsnip(values, 0, 0.95)
        plt.hist(snipped, 120)
        plt.xlim(margined(snipped, [0.05, 0]))
        plt.xlabel('standard deviation [min]')
        plt.ylabel('frequency')
        plt.title(
            'Stdev of ion finish time per peptide\n(based on evidence file)'
            + n_tag(len(values))
        )
        plt.gca().twinx()
        empirical_cdf(values, color='orange', density=True)
        plt.ylim([0,1])
        plt.ylabel('cdf')
        pm.savefig_label('')
        
    with plot_manager as pm:
        values = (
            (ion_max_finish_time - ion_min_start_time)
            .groupby('sequence').std().dropna()
        )
        snipped = qsnip(values, 0, 0.95)
        plt.hist(snipped, 120)
        plt.xlim(margined(snipped, [0.05, 0]))
        plt.xlabel('standard deviation')
        plt.ylabel('frequency')
        plt.title(
            'Stdev of ion retention length per peptide\n'
            '(based on evidence file)' + n_tag(len(values))
        )
        plt.gca().twinx()
        empirical_cdf(values, color='orange', density=True)
        plt.ylim([0,1])
        plt.ylabel('cdf')
        pm.savefig_label('')
        
    with plot_manager as pm:
        values = (
            (
                (ion_max_finish_time - ion_min_start_time)
                .groupby('sequence').std()
            ) / (
                (ion_max_finish_time - ion_min_start_time)
                .groupby('sequence').mean()
            )
        ).dropna()
        snipped = qsnip(values, 0, 0.95)
        plt.hist(snipped, 120)
        plt.xlim(margined(snipped, [0.05, 0]))
        plt.xlabel('coefficient of variation')
        plt.ylabel('frequency')
        plt.title(
            'Coefficient of variation of ion retention length per peptide\n'
            '(based on evidence file)' + n_tag(len(values))
        )
        plt.gca().twinx()
        empirical_cdf(values, color='orange', density=True)
        plt.ylabel('cdf')
        pm.savefig_label('')
        
    peptide_min_start_time = pd.concat({
        'sequence' : evidence['Modified sequence'],
        'start' : evidence['Calibrated retention time start'] \
            - evidence['Retention time calibration']
    }, axis=1).groupby('sequence').min().squeeze()
        
    peptide_max_finish_time = pd.concat({
        'sequence' : evidence['Modified sequence'],
        'finish' : evidence['Calibrated retention time finish'] \
            - evidence['Retention time calibration']
    }, axis=1).groupby('sequence').max().squeeze()
    
    with plot_manager as pm:
        snipped = qsnip(
            peptide_max_finish_time - peptide_min_start_time, 0, 0.99
        )
        plt.hist(snipped, 50)
        plt.xlim(margined(snipped, [0.05, 0]))
        plt.xlabel('retention length [min]')
        plt.ylabel('frequency')
        plt.title(
            'Peptide retention length (based on evidence file)'
            + n_tag(len(ion_max_finish_time - ion_min_start_time))
        )
        plt.gca().twinx()
        empirical_cdf(
            peptide_max_finish_time - peptide_min_start_time,
            color='orange', density=True
        )
        plt.ylim([0,1])
        plt.ylabel('cdf')
        pm.savefig_label('')
    
    with plot_manager as pm:    
        snipped = snip(peptide_max_finish_time - peptide_min_start_time, 0, 5)
        plt.hist(snipped, 50)
        plt.xlim(margined(snipped, [0.05, 0]))
        plt.xlabel('retention length [min]')
        plt.ylabel('frequency')
        plt.title(
            'Peptide retention length (based on evidence file)'
            + n_tag(len(ion_max_finish_time - ion_min_start_time))
        )
        plt.gca().twinx()
        empirical_cdf(
            peptide_max_finish_time - peptide_min_start_time,
            color='orange', density=True
        )
        plt.ylim([0,1])
        plt.ylabel('cdf')
        pm.savefig_label('')
        
    with plot_manager as pm:
        values = qsnip(
            evidence['Uncalibrated - Calibrated m/z [Da]'].dropna(),
            0.005, 0.995
        )
        plt.hist(values, 120)
        plt.xlim([min(values), max(values)])
        plt.xlabel('m/z offset [Da]')
        plt.ylabel('frequency')
        plt.title(
            'm/z Offset between uncalibrated and calibrated m/z in '
            'evidence file\n'
            + n_tag(len(evidence['Uncalibrated - Calibrated m/z [Da]']))
        )
        plt.gca().twinx()
        empirical_cdf(
            evidence['Uncalibrated - Calibrated m/z [Da]'].dropna(),
            color='orange', density=True
        )
        plt.ylim([0,1])
        plt.ylabel('cdf')
        pm.savefig_label('')
        

def merge_and_centroid_tof_spectrums(
        spectrums,
        parameters,
        show_plots=False,
        plot_manager=None
    ):
    """
    Merge ion mobility axis and centroid (in m/z) the resulting peaks.
    
    Parameters
    ----------
    spectrums : list or dict of array_like
        A list of spectrums. A spectrum is a 3xN array where spectrum[0] is 
        the m/z position of the N peaks, spectrum[1] is the intensity of
        the N peaks, and spectrum[2] is the inverse ion mobility of the N
        peaks.
        
    Returns
    -------
    centroided_spectrums : list or dict of array_like
        A dictionary of spectrum ids to centroided spectrums. A spectrum is
        a 2xN array where spectrum[0] is the m/z position of the N peaks and
        spectrum[1] is the intensity of the N peaks.
    
    Notes
    -----
    Centroiding is performed through a simple procedure, where peaks are
    combined if they are adjacent in m/z. Adjacency is determined based on
    discretized m/z, where the internal m/z discretization is inferred from
    the peaks.
    """
    
    if show_plots and plot_manager is None:
        plot_manager = PlotManager()
    
    #### Discretization
    INTERNAL_UNIT_MZ_SCALING = 1/2
    
    deltas = []
    for spec_id, spectrum in enumerate(spectrums):
        mz = np.sort(spectrum[0])
        u = mz**(INTERNAL_UNIT_MZ_SCALING)
        delta_u = np.ediff1d(u)[np.ediff1d(u) > 0]
        min_delta_u = delta_u.min()
        
        temp = delta_u[(delta_u > 0.5*min_delta_u) & (delta_u < 1.5*min_delta_u)]
        delta = temp.mean()
        
        if temp.std()/delta > 1e-4:
            logger.warning(f'Consecutive differences of internal units in spec_id = {spec_id} vary more than expected.')
        
        deltas.append(delta)
    deltas = np.array(deltas)
    
    if np.std(deltas) / np.mean(deltas) > 1e-4:
        logger.warning('Discretization deltas over scans varies more than expected.')
    
    
    max_discretization_values = np.array([spectrum[0].max() for spectrum in spectrums])**INTERNAL_UNIT_MZ_SCALING / deltas
    if not all(np.abs(deltas / np.mean(deltas) - 1) < 1/(2*max_discretization_values)):
        logger.warning('Averaging discretization deltas over scans results in recovery errors.')
    
    INTERNAL_UNIT_DELTA = np.mean(deltas)
    logger.info(f'INTERNAL_UNIT_DELTA = {INTERNAL_UNIT_DELTA}')
    
    x = spectrums[0][0][:-1]
    y = np.ediff1d(spectrums[0][0])
    c = y/np.sqrt(x)
    min_c = (c[c > 0]).min()
    idx = (c > min_c/2) & (c < min_c * 3/2)
    FUDGE_FACTOR = 10
    if not np.abs(c[idx].mean() / (INTERNAL_UNIT_DELTA / INTERNAL_UNIT_MZ_SCALING) - 1)  < FUDGE_FACTOR * (1/INTERNAL_UNIT_MZ_SCALING - 1) / (2*x.max()**INTERNAL_UNIT_MZ_SCALING) * INTERNAL_UNIT_DELTA:
        logger.warning('Mz discretization not matching with internal unit discretization delta.')
    
    def get_discretized_tof_values(spectrum):
        iu_values = spectrum[0]**INTERNAL_UNIT_MZ_SCALING
        return (iu_values - iu_values[0]) / INTERNAL_UNIT_DELTA
    
    ### Plots
    if show_plots:
        # Check that 5 random spectrums look good
        spec_ids_to_check = np.random.choice(range(len(spectrums)), 5)
        for spec_id in spec_ids_to_check:
            spectrum = spectrums[spec_id]
            mz = np.sort(spectrum[0])
            u = mz**(INTERNAL_UNIT_MZ_SCALING)
            min_delta_u = np.ediff1d(u)[np.ediff1d(u) > 0].min()
            
            with plot_manager as pm:
                plt.scatter(u[:-1], np.ediff1d(u), s=6)
                plt.ylim([0, 3.5*min_delta_u])
                plt.xlabel(f'internal unit (= m/z^{INTERNAL_UNIT_MZ_SCALING:g}) [Da^{INTERNAL_UNIT_MZ_SCALING:g}]')
                plt.ylabel(f'delta internal unit [Da^{INTERNAL_UNIT_MZ_SCALING:g}]')
                plt.title(f'Difference of internal unit (= m/z^{INTERNAL_UNIT_MZ_SCALING:g}) of consecutive\npeaks vs internal unit for spec_id={spec_id}, rt={pm.rt_array[spec_id]:.2f}' + n_tag(len(u)-1))
                pm.savefig_label('')
        
    
    #### Centroiding    
    def centroid_spectrum(spectrum):
        spectrum = spectrum[:, spectrum[1] > parameters.CENTROIDING_MIN_PROFILE_PEAK_INTENSITY]
        spectrum = spectrum[:, np.argsort(spectrum[0])]
        
        if len(spectrum[0]) == 0:
            return np.array([[],[],[]])

        mz_values, intensity_values = spectrum[:2]
        discretized_tof_values = get_discretized_tof_values(spectrum)
        
        groups = np.concatenate(([0], np.cumsum(np.ediff1d(discretized_tof_values) >= 1.5)))
        centroided_intensity_values = np.bincount(groups, weights=intensity_values)
        centroided_mz_values = np.bincount(groups, weights=mz_values * intensity_values) / centroided_intensity_values
        
        keep_idx = centroided_intensity_values > parameters.CENTROIDING_MIN_CENTROIDED_PEAK_INTENSITY
        centroided_mz_values = centroided_mz_values[keep_idx]
        centroided_intensity_values = centroided_intensity_values[keep_idx]
        return np.vstack([centroided_mz_values, centroided_intensity_values])
        
    
    #### Plots
    if show_plots:
        # Check that 5 random spectrums look good
        spec_ids_to_check = np.random.choice(range(len(spectrums)), 5)
        for spec_id in spec_ids_to_check:
            spectrum = spectrums[spec_id]
            
            with plot_manager as pm:
                plt.hist(np.log10(spectrum[1]), 120)
                plt.gca().axvline(50)
                plt.xlabel('log10(intensity)')
                plt.ylabel('frequency')
                plt.title(f'Histogram of log10(intensity) of profile peaks\nfor spec_id={spec_id}, rt={pm.rt_array[spec_id]:.2f}' + n_tag(len(spectrum[1])))
                pm.savefig_label('')
                
        spec_ids_to_check = np.random.choice(range(len(spectrums)), 5)
        for spec_id in spec_ids_to_check:
            spectrum = spectrums[spec_id]
            discretized_tof_values = get_discretized_tof_values(spectrum)
            discretized_tof_values = np.sort(discretized_tof_values)
            
            groups = np.concatenate(([0], np.cumsum(np.ediff1d(discretized_tof_values) >= 0.5)))
            widths = pd.DataFrame({'i' : discretized_tof_values, 'group' : groups}).groupby('group')['i'].apply(lambda x : int(x.max() - x.min()) + 1).to_numpy()
        
            with plot_manager as pm:
                int_hist(np.clip(widths, 0, 11), 120)
                plt.xticks(ticks = range(1,12), labels=list(range(1,11)) + ['>10'])
                plt.xlabel('number of scans')
                plt.ylabel('frequency')
                plt.title(f'Histogram of number of scans for each\ninternal unit discretization point, for spec_id={spec_id}, rt={pm.rt_array[spec_id]:.2f}' + n_tag(len(widths)))
                pm.savefig_label('')
                
        spec_ids_to_check = np.random.choice(range(len(spectrums)), 5)
        for spec_id in spec_ids_to_check:
            spectrum = spectrums[spec_id]
            discretized_tof_values = get_discretized_tof_values(spectrum)
            discretized_tof_values = np.sort(discretized_tof_values)
            
            groups = np.concatenate(([0], np.cumsum(np.ediff1d(discretized_tof_values) >= 1.5)))
            widths = pd.DataFrame({'i' : discretized_tof_values, 'group' : groups}).groupby('group')['i'].apply(lambda x : int(x.max() - x.min()) + 1).to_numpy()
        
            with plot_manager as pm:
                int_hist(np.clip(widths, 0, 11), 120)
                plt.xticks(ticks = range(1,12), labels=list(range(1,11)) + ['>10'])
                plt.xlabel('centroid width')
                plt.ylabel('frequency')
                plt.title(f'Histogram of centroid widths (number of consecutive profile peaks\nin tof discretization) for spec_id={spec_id}, rt={pm.rt_array[spec_id]:.2f}' + n_tag(len(widths)))
                pm.savefig_label('')
    
    
    centroided_spectrums = map(centroid_spectrum, tqdm(spectrums))
    centroided_spectrums = {
        spec_id : spectrum
        for spec_id, spectrum in enumerate(centroided_spectrums)
        if len(spectrum[0]) > 0
    }
    logger.info(f'len(centroided_spectrums) = {len(centroided_spectrums)}')

    # Check that spectrums are properly merged (i.e. the mobility axis is removed)
    for spectrum in centroided_spectrums.values():
        if len(spectrum[0]) > 1:
            assert np.ediff1d(np.sort(spectrum[0])).min() > 0
    
    return centroided_spectrums


def calibrate_spectrums(
        centroided_spectrums,
        ion_spec_pairs,
        parameters,
        show_plots=False,
        plot_manager=None
    ):
    """
    Calibrate m/z of centroided spectrums.
    
    The m/z of peaks in spectrums are calibrated based on known ions that are
    present in the spectrums.
    
    Spectrums with less than or equal to
    parameters.CALIBRATION_MINIMUM_REQUIRED_PEPTIDE_COUNT
    valid ions are removed.

    Parameters
    ----------
    centroided_spectrums : list or dict of array_like
        A list of centroided spectrums, or a dictionary of spectrum ids to 
        centroided spectrums. A spectrum is a 2xN array where spectrum[0] is 
        the m/z position of the N peaks and spectrums[1] is the intensity of
        the N peaks.
    
    ion_spec_pairs : DataFrame
        A collection of known/confident ion spectrum pairs. DataFrame has three
        columns 'sequence', 'charge', and 'spec_id'.
    
    Returns
    -------
    calibrated_spectrums dict of array_like
        A dictionary of spectrum ids to calibrated spectrums. A spectrum is a 
        2xN array where spectrum[0] is the m/z position of the N peaks and
        spectrum[1] is the intensity of the N peaks.
        
    Notes
    -----
    Calibration of m/z for a spectrum is performed by:
        1. finding m/z offset of each known ion to its nearest peak
        2. filtering all m/z offsets >= MAXIMUM_ABS_MZ_OFFSET
        3. shifting m/z of spectrum by the average of remaining m/z offsets
    """
    if show_plots and plot_manager is None:
        plot_manager = PlotManager()
    
    if isinstance(centroided_spectrums, dict):
        ion_spec_pairs = ion_spec_pairs[
            ion_spec_pairs['spec_id'].isin(centroided_spectrums)
        ]

    ### Calibrate m/z
    mz_offsets = compute_monoiso_mz_offsets(
        centroided_spectrums,
        ion_spec_pairs
    )
    
    fwhm = compute_fwhm_roughly(snip(
        mz_offsets,
        -parameters.CALIBRATION_MAX_ABS_MZ_OFFSET,
        parameters.CALIBRATION_MAX_ABS_MZ_OFFSET
    ))
    MAXIMUM_ABS_MZ_OFFSET = 3*fwhm

    idx = abs(mz_offsets) < MAXIMUM_ABS_MZ_OFFSET
    df = pd.DataFrame({
        'spec_id' : ion_spec_pairs['spec_id'][idx],
        'mz' : ion_spec_pairs[idx].apply(
            lambda x : peptide_monoiso_mz(x.sequence, x.charge),
            axis=1
        ),
        'offset' : mz_offsets[idx]
    }).sort_values(by='spec_id')
    
    num_ions = df.groupby('spec_id').apply(len)
    spectrums_to_calibrate = find(
        num_ions > parameters.CALIBRATION_MINIMUM_REQUIRED_PEPTIDE_COUNT
    )

    mz_calibration = df.groupby('spec_id').offset.mean()
    calibrated_spectrums = {
        spec_id : centroided_spectrums[spec_id] - mz_calibration[spec_id]
        for spec_id in spectrums_to_calibrate
    }
    logger.info(f'len(calibrated_spectrums) = {len(calibrated_spectrums)}')

    #### Plots
    if show_plots:
        plot_manager.max_abs_mz_offset = MAXIMUM_ABS_MZ_OFFSET
        
        with plot_manager as pm:
            plt.hist(snip(
                mz_offsets,
                -3*parameters.CALIBRATION_MAX_ABS_MZ_OFFSET,
                3*parameters.CALIBRATION_MAX_ABS_MZ_OFFSET
            ), 120)
            plt.axvline(
                -parameters.CALIBRATION_MAX_ABS_MZ_OFFSET,
                color='k',
                linestyle='--',
                alpha=0.8
            )
            plt.axvline(
                parameters.CALIBRATION_MAX_ABS_MZ_OFFSET,
                color='k',
                linestyle='--',
                alpha=0.8
            )
            plt.xlabel('m/z offset [Da]')
            plt.ylabel('frequency')
            plt.title(
                'Histogram of m/z offsets \n(using CALIBRATION_MAX_ABS_MZ_OFF'
                'SET={parameters.CALIBRATION_MAX_ABS_MZ_OFFSET:.4f} [Da])'
            )
            pm.savefig_label('')

        with plot_manager as pm:
            plt.scatter(pm.rt_array[num_ions.index], num_ions, s=1)
            plt.axhline(
                parameters.CALIBRATION_MINIMUM_REQUIRED_PEPTIDE_COUNT,
                color='k',
                linestyle='--',
                alpha=0.8
            )
            plt.xlabel('retention time [min]')
            plt.ylabel('number of peptides')
            plt.title(
                'Number of calibration ions per scan\n(using maximum abs m/z '
                f'offset={MAXIMUM_ABS_MZ_OFFSET:.4f} [Da])'
            )
            pm.savefig_label('')
        
        # Check that 5 random scans look good
        spectrums_to_check = np.random.choice(spectrums_to_calibrate, 5)
        for spec_id in spectrums_to_check:
            mz = df['mz'][df['spec_id'] == spec_id]
            offset = df['offset'][df['spec_id'] == spec_id]
            with plot_manager as pm:
                plt.scatter(mz, offset)
                plt.ylim([-pm.max_abs_mz_offset, pm.max_abs_mz_offset])    
                plt.xlabel('m/z [Da]')
                plt.ylabel('offset')
                plt.title(
                    f'm/z Offset for identified ions in spec_id={spec_id}, '
                    f'rt={pm.rt_array[spec_id]:.2f}'
                    + n_tag(len(mz))
                )
                pm.savefig_label('')
                
        with plot_manager as pm:
            plt.scatter(pm.rt_array[mz_calibration.index], mz_calibration, s=1)
            plt.xlabel('retention time [min]')
            plt.ylabel('mz calibration [Da]')    
            plt.title('m/z Calibration per scan')
            pm.savefig_label('')
            
    return calibrated_spectrums

def before_after_calibration_plots(
        ion_spec_pairs,
        centroided_spectrums,
        calibrated_spectrums,
        rt_array, parameters,
        plot_manager=None,
        max_abs_mz_offset=None
    ): 
    """
    The following plots compares m/z offsets of known ions before and after
    calibration.
    """
    if plot_manager is None:
        plot_manager = PlotManager()
    
    def mz_offset_plots(
            spectrums,
            ion_spec_pairs,
            title_tag,
            plot_manager=plot_manager
        ):
        if isinstance(spectrums, dict):
            ion_spec_pairs = ion_spec_pairs[ion_spec_pairs['spec_id'].isin(spectrums)]
        
        with plot_manager as pm:
            df = ion_spec_pairs.groupby('spec_id').apply(len)
            plt.scatter(pm.rt_array[df.index], df.values, s=1)
            plt.xlabel('retention time [min]')
            plt.ylabel('number of ions')
            plt.title('Number of (confident) ions per spectrum' + title_tag)
            pm.savefig_label('')
    
        mz_offsets = compute_monoiso_mz_offsets(spectrums, ion_spec_pairs)
        with plot_manager as pm:
            plt.hist(snip(mz_offsets, -max_abs_mz_offset, max_abs_mz_offset), 120)
            plt.xlim([-max_abs_mz_offset, max_abs_mz_offset])
            plt.xlabel('m/z offset [Da]')
            plt.ylabel('frequency')
            plt.title('m/z Offset between ion and nearest peak' + title_tag + n_tag(len(mz_offsets)))
            plt.gca().twinx()
            empirical_cdf(mz_offsets, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            pm.savefig_label('')
        
        df = ion_spec_pairs.copy()
        df['offset'] = mz_offsets
        df['mz'] = df.apply(lambda x : peptide_monoiso_mz(x.sequence, x.charge), axis=1)
        
        with plot_manager as pm:
            valid_idx = find(abs(df['offset']) < max_abs_mz_offset)
            idx = np.random.choice(valid_idx, size=100000)
            plt.scatter(df['mz'].loc[idx], df['offset'].loc[idx], s=1, alpha=0.02)
            plt.xlim(margined(pm.mz_bounds, 0.05))
            plt.ylim([-max_abs_mz_offset, max_abs_mz_offset])
            plt.xlabel('ion m/z [Da]')
            plt.ylabel('m/z offset [Da]')
            plt.title('m/z Offset vs m/z' + title_tag + n_tag(100000, len(valid_idx)))
            pm.savefig_label('')
            
            
        with plot_manager as pm:
            valid_idx = find(abs(df['offset']) < max_abs_mz_offset)
            idx = np.random.choice(valid_idx, size=100000)
            plt.scatter(pm.rt_array[df['spec_id'].loc[idx]], df['offset'].loc[idx], s=1, alpha=0.02)
            plt.xlim(margined(pm.rt_bounds, 0.05))
            plt.ylim([-max_abs_mz_offset, max_abs_mz_offset])
            plt.xlabel('retention time [min]')
            plt.ylabel('m/z offset [Da]')
            plt.title('m/z Offset vs retention time' + title_tag + n_tag(100000, len(valid_idx)))
            pm.savefig_label('')
    
    temp = ion_spec_pairs.copy()
    temp['rt'] = ion_spec_pairs['spec_id'].apply(lambda idx : rt_array[idx])
    temp = temp.groupby(['sequence', 'charge'])['rt'].mean()
    temp = temp.map(lambda rt : where_nearest(rt_array, rt)).rename('spec_id')
    apex_ion_spec_pairs = temp.reset_index()
    
    
    if not max_abs_mz_offset:
        if plot_manager.max_abs_mz_offset:
            max_abs_mz_offset = plot_manager.max_abs_mz_offset
        else:
            mz_offsets = compute_monoiso_mz_offsets(centroided_spectrums, ion_spec_pairs)
            fwhm = compute_fwhm_roughly(snip(mz_offsets, -parameters.CALIBRATION_MAX_ABS_MZ_OFFSET, parameters.CALIBRATION_MAX_ABS_MZ_OFFSET))
            max_abs_mz_offset = 3*fwhm
    
    mz_offset_plots(centroided_spectrums, apex_ion_spec_pairs, title_tag='\nfor ion spec pairs based on peak rt\n(before calibration)')
    mz_offset_plots(centroided_spectrums, ion_spec_pairs, title_tag='\nfor ions spec pairs based on rt range\n(before calibration)')
    mz_offset_plots(calibrated_spectrums, apex_ion_spec_pairs, title_tag='\nfor ion spec pairs based on peak rt\n(after calibration)')
    mz_offset_plots(calibrated_spectrums, ion_spec_pairs, title_tag='\nfor ions spec pairs based on rt range\n(after calibration)')


	

def extract_nearest_peak_dataframe(
        calibrated_spectrums,
        peptide_spec_pairs,
        parameters,
        show_plots=False,
        plot_manager=None
    ):
    """
    Extract nearest peaks for each known peptide spectrum pair.
    
    For each charge state of each known peptide, the nearest peak in the
    spectrum (in absolute m/z distance) is found for each predicted isotope 
    peak. Isotope peaks considered are '#0' (monoisotopic), '#1' (one extra
    neutron), and '#2' (two extra neutrons) as well as 'imaginary' peaks
    '#-1', '#-1/2', '#1/2', and '#3/2' (where '#k' means k neutrons from
    the monoisotopic mass). Information about these nearest peaks are used in
    downstream filtering.

    Parameters
    ----------
    calibrated_spectrums : list or dict of array_like
        A list of spectrums, or a dictionary of spectrum ids to spectrums.
        A spectrum is a 2xN array where spectrum[0] is 
        the m/z position of the N peaks and spectrum[1] is the intensity of
        the N peaks.
    
    peptide_spec_pairs : DataFrame
        A collection of known/confident peptide spectrum pairs. DataFrame has
        two columns 'sequence' and 'spec_id'.
    
    Returns
    -------
    nearest_peak_df : DataFrame
        Output dataframe with the m/z's and intensities of nearest peaks for
        each peptide spectrum pair.

    """
    
    if isinstance(calibrated_spectrums, dict):
        peptide_spec_pairs = peptide_spec_pairs[peptide_spec_pairs['spec_id'].isin(set(calibrated_spectrums.keys()))]

    temp = {}
    charges = np.arange(1, parameters.EXTRACTION_MAX_CHARGE + 1)
    for spec_id, sequences in tqdm(list((
            peptide_spec_pairs
            .groupby('spec_id')['sequence']
            .apply(list)
            .items()
        ))):
        spectrum = calibrated_spectrums[spec_id]
        mz_to_search = np.stack(
            [
                [
                    ion_target_mz(sequence,charge)
                    for charge in charges
                ] for sequence in sequences
            ]
        )
        
        idx = where_nearest(spectrum[0], mz_to_search, is_sorted=True)
        nearest_peaks = spectrum[:,idx]
        temp.update({
            (sequence,charge,spec_id) : nearest_peaks[:,i,charge-1, :]
            for i,sequence in enumerate(sequences) for charge in charges
        })
    
    index, values = zip(*temp.items())
    values = np.array(values)
    hash_peaks = ['#0','#1','#2','#-1','#-1/2','#1/2','#3/2']
    nearest_peak_df = pd.DataFrame(data=values.reshape(values.shape[0], -1), index=index, columns=pd.MultiIndex.from_tuples( itertools.product(['mz','intensity'], hash_peaks))).sort_index()
    
    return nearest_peak_df


def create_summary_dataframe(nearest_peak_df):
    """
    Create summary statistics from nearest peak dataframe.
    
    For each peptide and spectrum found in the nearest peak dataframe, summary
    information is calculated for every charge state. Summary information
    includes m/z offsets of nearest peaks to each isotope, similarity score 
    between observed and predicted isotope distribution, information about
    isotope intensities, etc. Summary information is used in downstream
    filtering.
    
    
    Parameters
    ----------
    nearest_peak_df : DataFrame
        Dataframe with the m/z's and intensities of nearest peaks for isotope
        peaks for each peptide spectrum pair. Indexed by peptide spectrum
        pairs.
    
    Returns
    -------
    summary_df : DataFrame
        Output dataframe with information about every charge state for each
        peptide spectrum pair in nearest_peak_df. Indexed by peptide spectrum
        pairs.

    """
    
    ions = ( (seq, charge) for seq, charge, spec_id in nearest_peak_df.index )
    mz_offsets = nearest_peak_df['mz'] - np.array([ion_target_mz(*ion) for ion in ions])
    
    summary_df = pd.DataFrame(index = nearest_peak_df.index, dtype='object')
    
    hash_peaks = ['#0','#1','#2', '#-1', '#-1/2','#1/2', '#3/2']
    summary_df[[col + ' mz_offset' for col in hash_peaks]] = mz_offsets[hash_peaks]
    
    sequences = nearest_peak_df.index.get_level_values(0)
    iso_prob = np.stack([peptide_isotope_peaks(sequence)[1] for sequence in sequences])
    nearest_iso_intensity = nearest_peak_df['intensity'][['#0', '#1', '#2']].values
    u = nearest_iso_intensity
    v = iso_prob
    summary_df['similarity_score'] = cosine_similarity(u, v, axis=1)
    summary_df['extracted_intensity'] = scalar_proj(u, v, axis=1)
    
    summary_df['#-1 intensity_is_nonnegligible'] = nearest_peak_df['intensity']['#-1'] >  nearest_peak_df['intensity']['#0']/2
    summary_df['#-1/2 intensity_is_nonnegligible'] = nearest_peak_df['intensity']['#-1/2'] >  nearest_peak_df['intensity']['#0']/2
    summary_df['#1/2 intensity_is_nonnegligible'] = nearest_peak_df['intensity']['#1/2'] >  nearest_peak_df['intensity'][['#0','#1']].min(1)/2
    summary_df['#3/2 intensity_is_nonnegligible'] = nearest_peak_df['intensity']['#3/2'] >  nearest_peak_df['intensity'][['#1','#2']].min(1)/2

    return summary_df


def extract_csd_from_summary_dataframe(
        summary_df,
        parameters,
        show_plots=False,
        plot_manager=None
    ):
    """
    Extract intensity and CSD readings from summary dataframe.
    
    For each peptide spectrum pair, the estimated intensity of each charge
    state is collected into extracted_intensity_df. The extracted intensity
    is set to np.nan if the charge state does not pass the filtering steps.
    The scan-to-scan extracted intensities are combined into CSD estimates for
    each peptide.
    
    Parameters
    ----------
    summary_df : DataFrame
        Dataframe with information about every charge state for each peptide
        spectrum pair. Indexed by peptide spectrum pairs.
    
    Returns
    -------
    csd : DataFrame
        Output dataframe with charge state distributions. Indexed by peptides.
        Columns are charge states from 1 to 
        ``parameters.EXTRACTION_MAX_CHARGE``.
        
    extracted_intensity_df : DataFrame
        Output dataframe with extracted intensities of charge states.
        Multiindexed by peptide spectrum pairs. Columns are charge states from
        1 to ``parameters.EXTRACTION_MAX_CHARGE``.
        
    Notes
    -----
    For each charge state, filtering includes checking presense of spectrum
    peaks around the isotope peaks, cosine similarity of observed and
    predicted isotope distribution, etc.
    """
    #### Set EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH parameter
    if show_plots and plot_manager is None:
        plot_manager = PlotManager()
    
    
    fwhm = compute_fwhm_roughly(snip(
        summary_df['#0 mz_offset'],
        -parameters.CALIBRATION_MAX_ABS_MZ_OFFSET,
        parameters.CALIBRATION_MAX_ABS_MZ_OFFSET
    ))
    
    if not parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH:
         parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH = 1 * fwhm
         
    if not parameters.EXTRACTION_MINIMUM_MZ_OFFSET_FOR_NO_MATCH:
         parameters.EXTRACTION_MINIMUM_MZ_OFFSET_FOR_NO_MATCH = 5 * fwhm
         
    if not parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS:
         parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS = (
             2.5 * fwhm
         )
    
    
    if show_plots:
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df['#0 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.title('#0 mz_offset in summary_df\n' + n_tag(len(values)))
            plt.xlabel('m/z [Da]')
    
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df['#1 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title('#1 mz_offset in summary_df\n' + n_tag(len(values)))
            
            
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df['#2 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title('#2 mz_offset in summary_df\n' + n_tag(len(values)))
            
    
    #### Set EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH parameter
    if not parameters.EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH:
        parameters.EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH = 0.98
    
    
    idx = (
        (
            summary_df['#0 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['#1 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['#2 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        )
    )
    
    
    if show_plots:
        with plot_manager as pm:
            pm.savefig_label('')
            plt.hist(
                np.log10(1-summary_df[idx]['similarity_score'] + 1e-12),
                240
            )
            plt.xlim([-8,0])
            plt.gca().axvline(
                np.log10(
                    1
                    - parameters.EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH
                ),
                color='k',
                linestyle='--'
            )
            plt.gca().twinx()
            empirical_cdf(
                np.log10(1 - summary_df[idx]['similarity_score'] + 1e-12),
                color='orange',
                density=True
            )
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('cosine similarity')
                        
    logger.info(f"Fraction of similarity scores > threshold: {np.mean((summary_df[idx]['similarity_score'] > parameters.EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH)):.3f}.")

    
    summary = {}
    summary['has_hash_peaks'] = (
        (
            summary_df['#0 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['#1 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['#2 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        )
    )
    summary['has_high_similarity_scores'] = (
        summary_df['similarity_score']
        > parameters.EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH
    )
     
    summary['has_no_hash_neg_half_peak'] = ~(
        (
            summary_df['#-1/2 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS
        ) & summary_df['#-1/2 intensity_is_nonnegligible']
    )
    summary['has_no_hash_half_peak'] = ~(
        (
            summary_df['#1/2 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS
        ) & summary_df['#-1/2 intensity_is_nonnegligible']
    )
    summary['has_no_hash_3_half_peak'] = ~(
        (
            summary_df['#3/2 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS
        ) & summary_df['#-1/2 intensity_is_nonnegligible']
    )
    summary['has_no_hash_neg_peak'] = ~(
        (
            summary_df['#-1 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS
        ) & summary_df['#-1/2 intensity_is_nonnegligible']
    )
    summary['has_no_extraneous_peaks'] = (
        summary['has_no_hash_neg_half_peak']
        & summary['has_no_hash_half_peak']
        & summary['has_no_hash_3_half_peak']
        & summary['has_no_hash_neg_peak']
    )
    
    events = [
        summary['has_no_hash_neg_half_peak'],
        summary['has_no_hash_half_peak'],
        summary['has_no_hash_3_half_peak'],
        summary['has_no_hash_neg_peak'],
        summary['has_no_extraneous_peaks']
    ]
    
    givens = [
        summary['has_hash_peaks'],
        summary['has_hash_peaks'] & summary['has_high_similarity_scores']
    ]
    
    summary_conditional_prob = np.array([
        [ sum(A & B) / sum(B) for B in givens ] for A in events
    ])
    
    logger.info(summary_conditional_prob)
    
    
    if show_plots:
        idx = (
            (
                summary_df['#0 mz_offset'].abs()
                < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
            ) & (
                summary_df['#1 mz_offset'].abs()
                < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
            ) & (
                summary_df['#2 mz_offset'].abs()
                < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
            ) & (
                summary_df['similarity_score']
                > parameters.EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH
            )
        )
        
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df[idx]['#-1 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title('#-1 mz_offset in summary_df\n' + n_tag(len(values)))
            
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df[idx]['#-1/2 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title('#-1/2 mz_offset in summary_df\n' + n_tag(len(values)))
            
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df[idx]['#1/2 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title('#1/2 mz_offset in summary_df\n' + n_tag(len(values)))
            
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df[idx]['#3/2 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title('#3/2 mz_offset in summary_df\n' + n_tag(len(values)))
        
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df[
                idx
                & summary_df['#-1 intensity_is_nonnegligible']
            ]['#-1 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title(
                '#-1 mz_offset in summary_df given nonnegligible intensity\n'
                + n_tag(len(values))
            )
            
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df[
                idx
                & summary_df['#-1/2 intensity_is_nonnegligible']
            ]['#-1/2 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title(
                '#-1/2 mz_offset in summary_df given nonnegligible intensity\n'
                + n_tag(len(values))
            )
            
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df[
                idx
                & summary_df['#1/2 intensity_is_nonnegligible']
            ]['#1/2 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title(
                '#1/2 mz_offset in summary_df given nonnegligible intensity\n'
                + n_tag(len(values))
            )
            
        with plot_manager as pm:
            pm.savefig_label('')
            values = summary_df[
                idx
                & summary_df['#3/2 intensity_is_nonnegligible']
            ]['#3/2 mz_offset']
            plt.hist(snip(values, -2*fwhm, 2*fwhm), 120)
            plt.xlim([-2*fwhm, 2*fwhm])
            plt.gca().twinx()
            empirical_cdf(values, color='orange', density=True)
            plt.ylim([0,1])
            plt.ylabel('cdf')
            plt.xlabel('m/z [Da]')
            plt.title(
                '#3/2 mz_offset in summary_df given nonnegligible intensity\n'
                + n_tag(len(values))
            )
        
    
    #### Filter and extract intensity and CSD readings
    is_match = (
        (
            summary_df['#0 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['#1 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['#2 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['similarity_score'].abs()
            > parameters.EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH
        ) & ~(
            (
                summary_df['#-1 mz_offset'].abs()
                < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS
            ) & summary_df['#-1 intensity_is_nonnegligible']
        ) & ~(
            (
                summary_df['#-1/2 mz_offset'].abs()
                < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS
            ) & summary_df['#-1/2 intensity_is_nonnegligible']
        ) & ~(
            (
                summary_df['#1/2 mz_offset'].abs()
                < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS
            ) & summary_df['#1/2 intensity_is_nonnegligible']
        ) & ~(
            (
                summary_df['#3/2 mz_offset'].abs()
                < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_EXTRANEOUS_PEAKS
            ) & summary_df['#3/2 intensity_is_nonnegligible']
        )
    )
    
    is_no_match = (
        (
            summary_df['#0 mz_offset'].abs()
            > parameters.EXTRACTION_MINIMUM_MZ_OFFSET_FOR_NO_MATCH
        ) & (
            summary_df['#1 mz_offset'].abs()
            > parameters.EXTRACTION_MINIMUM_MZ_OFFSET_FOR_NO_MATCH
        )
    )
    
    extracted_intensity_df = pd.Series(np.nan, index = summary_df.index)
    extracted_intensity_df[is_match] = \
        summary_df['extracted_intensity'][is_match]
    extracted_intensity_df[is_no_match] = 0.0
    extracted_intensity_df = extracted_intensity_df.unstack(level=1)
    
    csi = extracted_intensity_df.dropna().groupby(level = 0).sum()
    csi = csi[csi.sum(1) > 0]
    csd = csi.divide(csi.sum(1), 0)
    
    
    if show_plots:
        with plot_manager as pm:
            pm.savefig_label('')
            ion_intensities = csi.stack()
            ion_intensities = ion_intensities[ion_intensities > 0]
            mz = np.array([
                peptide_monoiso_mz(sequence, charge)
                for sequence, charge in ion_intensities.index
            ])
            idx = np.random.choice(range(len(mz)), size=100000)
            plt.scatter(
                mz[idx], np.log10(ion_intensities)[idx], s=1, alpha=0.02
            )
            plt.xlim(margined(pm.mz_bounds, 0.05))
            plt.ylim(margined([3, 9], 0.05))
            plt.xlabel('m/z [Da]')
            plt.ylabel('extracted ion intensity')
            a = np.percentile(mz, 5)
            b = np.percentile(mz, 95)
            idx = (mz >= a) & (mz <= b)
            plot_qcut_boxplot(
                mz[idx], np.log10(ion_intensities)[idx], whis=[5,95]
            )
            
        with plot_manager as pm:
            pm.savefig_label('')
            peptide_intensities = csi.sum(1)
            peptide_intensities = peptide_intensities[peptide_intensities > 0]
            mass = np.array([
                peptide_monoiso_mass(sequence)
                for sequence in peptide_intensities.index
                ])
            idx = np.random.choice(range(len(mass)), size=100000)
            plt.scatter(
                mass[idx], np.log10(peptide_intensities)[idx], s=1, alpha=0.02
            )
            plt.xlim(margined([400, 4000], 0.05))
            plt.ylim(margined([3, 9], 0.05))
            plt.xlabel('m/z [Da]')
            plt.ylabel('extracted ion intensity')
            a = np.percentile(mass, 5)
            b = np.percentile(mass, 95)
            idx = (mass >= a) & (mass <= b)
            plot_qcut_boxplot(
                mass[idx], np.log10(peptide_intensities)[idx], whis=[5,95]
            )
        
    logger.info(f'len(csd) = {len(csd)}')
    logger.info(f'len(extracted_intensity_df) = {len(extracted_intensity_df)}')
    
    return csd, extracted_intensity_df
    

def isotope_distribution_plots(
        ion_spec_pairs,
        nearest_peak_df,
        summary_df,
        parameters,
        plot_manager=None
    ):
    """
    The following plots checks that cosine similarity of isotope distributions
    of extracted peptides.
    """
    if plot_manager is None:
        plot_manager = PlotManager()
    
    valid_idx = (
        (
            summary_df['#0 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['#1 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        ) & (
            summary_df['#2 mz_offset'].abs()
            < parameters.EXTRACTION_MAXIMUM_MZ_OFFSET_FOR_MATCH
        )
    )
    
    
    temp = set(nearest_peak_df[valid_idx].index)
    relevant_idx = ion_spec_pairs.apply(lambda r : tuple(r) in temp, axis=1)
    relevant_ion_scan_pairs = ion_spec_pairs[relevant_idx]    
    
    iso_distribution = np.stack([
        peptide_isotope_peaks(sequence)[1]
        for sequence in relevant_ion_scan_pairs['sequence']
    ])
    
    nearest_iso_intensity = (
        nearest_peak_df['intensity'][['#0', '#1', '#2']]
        .loc[relevant_ion_scan_pairs.values.tolist()]
        .values
    )
    
    u = nearest_iso_intensity
    v = iso_distribution
    scores = cosine_similarity(u, v, axis=1)
    control_scores = cosine_similarity(
        u[np.random.permutation(np.arange(len(u)))],
        v,
        axis=1
    )
    
    with plot_manager as pm:   
        pm.savefig_label('')
        plt.hist(np.log10(1-scores+1e-12),240, density=True)
        plt.hist(
            np.log10(1-control_scores+1e-12), 240,
            color='C1',alpha=0.5, density=True
        )
        plt.xlabel('log10(1-CS(P,Q))')
        plt.ylabel('Density')
        plt.legend([
            'P=Theoretical, Q=Empirical',
            'P=Theoretical, Q=Random Empirical'
        ])
        plt.xlim([-8,0])
        plt.xticks(np.arange(-8,1,1))
        plt.title(
            'Cosine Simlarity of Isotope Distributions from #0-#2'
            f'(n={len(u)})'
        )
        plt.gca().axvline(
            np.log10(
                1
                - parameters.EXTRACTION_MINIMUM_SIMILARITY_SCORE_FOR_MATCH
            ), color='k', linestyle='--'
        )

    with plot_manager as pm:
        pm.savefig_label('')
        mz = np.array([
            ion_target_mz(sequence, charge)[0]
            for sequence, charge in zip(
                summary_df.index.get_level_values(0)[valid_idx],
                summary_df.index.get_level_values(1)[valid_idx]
            )]
        )
        y = np.log10(1 - summary_df['similarity_score'][valid_idx])
        idx = np.random.choice(range(sum(valid_idx)), size=100000)
        plt.scatter(mz[idx], y[idx], s=1, alpha=0.02)
        plt.xlim(margined(pm.mz_bounds, 0.05))
        plt.ylim(margined([-8, 0], 0.05))
        plt.xlabel('m/z [Da]')
        plt.ylabel('log10(1 - cosine_simlarity)')
        plot_qcut_boxplot(mz, y, qlim=[0.05,0.95], whis=[5,95])
    
    with plot_manager as pm:
        pm.savefig_label('')
        log10_intensity = np.log10(
            summary_df['extracted_intensity'][valid_idx]
        )
        y = np.log10(1 - summary_df['similarity_score'][valid_idx])
        idx = np.random.choice(range(sum(valid_idx)), size=100000)
        plt.scatter(log10_intensity[idx], y[idx], s=1, alpha=0.02)
        plt.ylim(margined([-8, 0], 0.05))
        plt.xlabel('log10(extracted_intensity) [a.u.]')
        plt.ylabel('log10(1 - cosine_simlarity)')
        plot_qcut_boxplot(
            log10_intensity[idx], y[idx], qlim=[0.05, 0.95], whis=[5,95]
        )
    
    with plot_manager as pm:
        pm.savefig_label('')
        mz = np.array([
            ion_target_mz(sequence, charge)[0]
            for sequence, charge in zip(
                summary_df.index.get_level_values(0)[valid_idx],
                summary_df.index.get_level_values(1)[valid_idx]
            )]
        )
        log10_intensity = np.log10(
            summary_df['extracted_intensity'][valid_idx]
        )
        idx = np.random.choice(range(sum(valid_idx)), size=100000)
        plt.scatter(mz[idx], log10_intensity[idx], s=1, alpha=0.02)
        plt.xlim(margined(pm.mz_bounds, 0.05))
        plt.xlabel('m/z [Da]')
        plt.ylabel('log10(extracted_intensity) [a.u.]')
        plot_qcut_boxplot(
            mz[idx], log10_intensity[idx], qlim=[0.05, 0.95], whis=[5,95]
        )
        
    with plot_manager as pm:
        pm.savefig_label('')
        scatter_with_kde(
            mz[idx], log10_intensity[idx],
            s=1, alpha=0.02, subsampling_size=10000
        )
        plt.xlim(margined(pm.mz_bounds, 0.05))
        plt.xlabel('m/z [Da]')
        plt.ylabel('log10(extracted_intensity) [a.u.]')
        plot_qcut_boxplot(
            mz[idx], log10_intensity[idx], qlim=[0.05, 0.95], whis=[5,95]
        )
    
    
#### Auxilliary classes and methods

class PlotManager():
    def __init__(
            self,
            savefig_dir=None,
            raw_file_name=None,
            rt_array=None,
            tic_array=None,
            mz_bounds=None,
            rt_bounds=None,
            max_abs_mz_offset=None
        ):
        
        self.savefig_dir = Path(savefig_dir) if savefig_dir else None
        self.raw_file_name = raw_file_name
        self.rt_array = rt_array
        self.tic_array = tic_array
        self.mz_bounds = mz_bounds
        self.rt_bounds = rt_bounds
        self.max_abs_mz_offset = max_abs_mz_offset
        self.fig_counter = 1
    
    def savefig_label(self, label):
        self._savefig_label = label
    
    def __enter__(self):
        plt.figure()
        self._savefig_label = None
        return self
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.raw_file_name:
            plt.suptitle(self.raw_file_name, y=1.1)
        
        if self.savefig_dir:
            savefig_label_tag = (
                '__' + self._savefig_label
                if self._savefig_label
                else ''
            )
            plt.savefig(
                self.savefig_dir / (
                    'fig' + str(self.fig_counter) + savefig_label_tag + '.png'
                ),
                bbox_inches='tight'
            )
            self.fig_counter += 1
            
        plt.show()
        plt.close()


def read_mzml(mzml_file, return_spectrums=True):
    out = mzml.read(mzml_file)    
    spectrums = []
    rt_array = []
    tic_array = []
    iit_array = []
    mz_bounds = None
    for scan_properties in tqdm(out):
        if scan_properties['ms level'] == 1:
            if not mz_bounds:
                mz_bounds = (
                    scan_properties['scanList']['scan'][0] \
                        ['scanWindowList']['scanWindow'][0] \
                            ['scan window lower limit'],
                    scan_properties['scanList']['scan'][0] \
                        ['scanWindowList']['scanWindow'][0] \
                            ['scan window upper limit']
                )
                
            if return_spectrums:
                mz_array = scan_properties['m/z array']
                intensity_array = scan_properties['intensity array']

                try:
                    irim_array = scan_properties['mean inverse reduced ion mobility array']
                    spectrum = np.array([mz_array, intensity_array, irim_array])
                except KeyError:
                    spectrum = np.array([mz_array, intensity_array])
                spectrums.append(spectrum)


            rt = scan_properties['scanList']['scan'][0]['scan start time']
            tic = scan_properties['total ion current']
            try:
                iit = scan_properties['scanList']['scan'][0]['ion injection time']                
                iit_array.append(iit)
            except KeyError:
                pass
            
            if rt.unit_info != 'minute':
                if rt.unit_info == 'second':
                    rt = rt/60
                else:
                    raise Exception('Retention time has units.')
            
            rt_array.append(rt)
            tic_array.append(tic)
            

    rt_bounds = (min(rt_array), max(rt_array))
    output = {}
    if return_spectrums:
        output['spectrums'] = spectrums
    output['rt_array'] = np.array(rt_array)
    output['tic_array'] = np.array(tic_array)
    if iit_array:
        output['iit_array'] = np.array(iit_array)
    output['mz_bounds'] = mz_bounds
    output['rt_bounds'] = rt_bounds
    return output


def read_bruker_d_folder(bruker_d_folder, return_spectrums=True):
    import alphatims.bruker

    timstof_data = alphatims.bruker.TimsTOF(bruker_d_folder)
    
    ms1_frames = list(timstof_data.frames.index[timstof_data.frames['MsMsType'] == 0])
    ms1_frames.remove(0)
    ms1_frames = np.array(ms1_frames)
    
    spectrums = []
    rt_array = timstof_data.rt_values[ms1_frames] / 60
    tic_array = timstof_data.frames['SummedIntensities'][ms1_frames]
    mz_bounds = (float(timstof_data.meta_data['MzAcqRangeLower']), float(timstof_data.meta_data['MzAcqRangeUpper']))
    irim_bounds = (float(timstof_data.meta_data['OneOverK0AcqRangeLower']), float(timstof_data.meta_data['OneOverK0AcqRangeUpper']))
    
    for f in tqdm(ms1_frames):
        mz_values = timstof_data[f]['mz_values'].to_numpy()
        intensity_values = timstof_data[f]['intensity_values'].to_numpy()
        irim_values = timstof_data[f]['mobility_values'].to_numpy()
    
        spectrum = np.array([mz_values, intensity_values, irim_values])
        spectrums.append(spectrum)

    rt_bounds = (min(rt_array), max(rt_array))
    
    output = {}
    if return_spectrums:
        output['spectrums'] = spectrums
    output['rt_array'] = np.array(rt_array)
    output['tic_array'] = np.array(tic_array)
    output['mz_bounds'] = mz_bounds
    output['rt_bounds'] = rt_bounds
    output['irim_bounds'] = irim_bounds
    return output


#### Extraction entry-point
def extract_csd(
        ms_data_file,
        maxquant_txt_dir,
        ms_instrument=None,
        output_dir=None,
        parameters=None,
        raw_file_name=None,
        show_plots=False,
        save_plots=False
    ):
    """
    Extract charge state distributions and scan-to-scan intensity readings of
    charge states for MaxQuant identified peptides from LC-MS data.
    
    The extraction scheme accepts .mzML files or Bruker .d folders. If
    ``ms_instrument == 'orbitrap'`` then `ms_data_file` should be a .mzML file
    with profile peaks centroided. If ``ms_instrument == 'timstof'`` then
    `ms_data_file` should be a .mzML file or Bruker .d folder, and the
    spectrums will be centroided during extraction.
    
    
    Parameters
    ----------
    ms_data_file : str, path object, or file-like object
        Path to input mass spectrum data file.
    
    maxquant_txt_dir : str, path object, or file-like object
        Path to MaxQuant's output txt folder. The folder should contain
        evidence.txt and summary.txt. Identified ions from evidence.txt are
        used for extraction. Fixed modifications are read from summary.txt. 
    
    ms_instrument : str
        Should be equal to ``'orbitrap'`` or ``'timstof'``.  If
        ``'orbitrap'`` then `ms_data_file` should be a .mzML file
        with profile peaks centroided. If ``'timstof'`` then `ms_data_file`
        should be a .mzML file or Bruker .d folder, and the spectrums will be
        centroided during extraction.
    
    output_dir : str, path object, or file-like object
        If provided, saves the outputs, properties of the LC-MS run, and
        logging of the extraction scheme. The outputs, `csd` and
        `extracted_intensity_df`, and the .log file are saved directly to
        `output_dir`. Properties of the LC-MS run are read from `ms_data_file`,
        and saved to `output_dir` / raw_properties. 
    
    parameters : dict
        Dictionary of parameters for extraction. Uses default parameters for
        those not provided. See ``Parameters`` class for the list of
        extraction parameters.
    
    raw_file_name : str
        Name of the raw MS file, if  different from the filename of
        `ms_data_file`. Used for determining the correct experiment in the
        MaxQuant txt folder.
        
    show_plots : bool
        If enabled, the extraction scheme will generate figures regarding the
        extraction process.
        
    save_plots : bool
        If enabled, figures generated during extraction will be saved to
        directory `output_dir` / fig' /. Also sets ``show_plots == True``.
        Requires `output_dir` to be provided.
        
    Returns
    -------
    csd : DataFrame
        Output dataframe with charge state distributions. Indexed by peptides.
        Columns are charge states from 1 to 
        ``parameters.EXTRACTION_MAX_CHARGE``.
        
    extracted_intensity_df : DataFrame
        Output dataframe with extracted intensities of charge states.
        Multiindexed by peptide spectrum pairs. Columns are charge states from
        1 to ``parameters.EXTRACTION_MAX_CHARGE``.
        
    Notes
    -----
    A brief overview of the extraction scheme:
        1. Reads spectrums from `ms_data_file`. A spectrum is represented as
        a MxN numpy array, where M = 2 or 3, and N = the nnumber of peaks in
        the spectrum. ``spectrum[0]`` contains m/z, ``spectrum[1]`` contains
        intensity, and ``spectrum[2]`` (for timstof data only) contains inverse
        ion mobility.
        2. Reads start and finish times of identified ions from MaxQuant's
        evidence.txt file.
        3. Centroids profile peaks for timstof data. 
        4. Calibrates m/z of spectrums using MaxQuant's identified ions.
        5. Extracts information about spectrum peaks surrounding the predicted
        isotope distributions of each charge state of each identified peptide.
        6. Filters peptide charge states based on previously extracted
        information. Collects the estimated spectrum intensity for remaining
        peptide charge states, and saves to output.
    """
    
    if not (ms_instrument == 'orbitrap' or ms_instrument == 'timstof'):
        raise ValueError('ms_instrument needs to be equal to "orbitrap" or "timstof".')
        
    parameters = process_parameters(parameters)
    logger.handlers = []

    if raw_file_name is None:
        raw_file_name = Path(ms_data_file).stem
        
    if save_plots:
        assert output_dir 
        show_plots = True
    
    if show_plots and 'matplotlib.pyplot' not in sys.modules:
        show_plots = False
        save_plots = False
        logger.info(
            'Importing matplotlib failed. '
            'Proceeding with show_plots = False, save_plots = False.'
        )
        
    
    if output_dir:
        output_dir = Path(output_dir)
        savefig_dir = preprocessing_for_output(output_dir, save_plots)
    else:
        savefig_dir=None
        
    logger.info(f'ms_data_file = {ms_data_file}')
    logger.info(f'maxquant_txt_dir = {maxquant_txt_dir}')
    logger.info(f'ms_instrument = {ms_instrument}')
    logger.info(f'output_dir = {output_dir}')
    logger.info(f'parameters = {parameters}')
    logger.info(f'show_plots = {show_plots}')
    logger.info(f'save_plots = {save_plots}')
    
    logger.info('Reading spectrums from MS data file...')
    ext = os.path.splitext(ms_data_file)[1].lower()
    if  ext == '.mzml':        
        output = read_mzml(ms_data_file)
    elif ext == '.d':
        output = read_bruker_d_folder(ms_data_file)
    else: 
        raise ValueError(f"Unknown file extension: {ext}. Expected .mzML or .d.")
    spectrums = output['spectrums']
    rt_array = output['rt_array']
    tic_array = output['tic_array']
    try:
        # Try since iit_array not available for timstof
        iit_array = output['iit_array']
    except KeyError:
        pass
    mz_bounds = output['mz_bounds']
    rt_bounds = output['rt_bounds']
    try:
        # Try since irim_bounds not available for orbitrap and not implemented for read_mzml
        irim_bounds = output['irim_bounds']
    except KeyError:
        pass
    del output
    
    logger.info(f'len(spectrums) = {len(spectrums)}')
    logger.info(f'len(rt_array) = {len(rt_array)}')
    logger.info(f'len(tic_array) = {len(tic_array)}')
    try:
        logger.info(f'len(iit_array) = {len(iit_array)}')
    except NameError:
        pass
    logger.info(f'mz_bounds = {mz_bounds}')
    logger.info(f'rt_bounds = {rt_bounds}')
    try:
        logger.info(f'irim_bounds = {irim_bounds}')
    except NameError:
        pass
    
    if output_dir:
        raw_properties_output_dir = output_dir / 'raw_properties'
        raw_properties_output_dir.mkdir(parents=True, exist_ok=True)

        def to_csv(path, array):
            pd.Series(array).to_csv(path, header=False, index=False)
        
        to_csv(raw_properties_output_dir / 'rt_array.csv', rt_array)
        to_csv(raw_properties_output_dir / 'tic_array.csv', tic_array)
        try:            
            to_csv(raw_properties_output_dir / 'iit_array.csv', iit_array)
        except NameError:
            pass
        to_csv(raw_properties_output_dir / 'mz_bounds.csv', mz_bounds)
        to_csv(raw_properties_output_dir / 'rt_bounds.csv', rt_bounds)
        try:            
            to_csv(raw_properties_output_dir / 'irim_bounds.csv', irim_bounds)
        except NameError:
            pass
    
    
    plot_manager = PlotManager(
        savefig_dir=savefig_dir,
        raw_file_name=raw_file_name,
        rt_array=rt_array,
        tic_array=tic_array,
        mz_bounds=mz_bounds,
        rt_bounds=rt_bounds
    )
    
    ion_elution_intervals = extract_ion_elution_intervals_from_maxquant(
        maxquant_txt_dir,
        raw_file_name,
        show_plots=show_plots,
        plot_manager=plot_manager
    )
    
    if ms_instrument == 'timstof':
        logger.info('Centroiding spectrums...')
        centroided_spectrums = merge_and_centroid_tof_spectrums(
            spectrums,
            parameters=parameters,
            show_plots=show_plots,
            plot_manager=plot_manager
        )
    else:
        centroided_spectrums = spectrums
    del spectrums


    logger.info('Calibrating spectrums...')
    ion_spec_pairs = compute_ion_spec_pairs(
        ion_elution_intervals,
        rt_array, parameters
    )
    calibrated_spectrums = calibrate_spectrums(
        centroided_spectrums,
        ion_spec_pairs,
        parameters=parameters,
        show_plots=show_plots,
        plot_manager=plot_manager
    )
    
    if show_plots:
        logger.info('Plotting calibration before and after figures...')
        before_after_calibration_plots(
            ion_spec_pairs,
            centroided_spectrums,
            calibrated_spectrums,
            rt_array,
            parameters=parameters,
            plot_manager=plot_manager
        )
    del centroided_spectrums
    
    
    logger.info('Extracting peak information from spectrums...')
    peptide_spec_pairs = compute_peptide_spec_pairs(
        ion_elution_intervals,
        rt_array,
        parameters
    )
    nearest_peak_df = extract_nearest_peak_dataframe(
        calibrated_spectrums,
        peptide_spec_pairs,
        parameters=parameters,
        show_plots=show_plots,
        plot_manager=plot_manager
    )
    

    logger.info('Filtering extracted charge state intensities...')
    summary_df = create_summary_dataframe(nearest_peak_df)
    csd, extracted_intensity_df = extract_csd_from_summary_dataframe(
        summary_df,
        parameters=parameters,
        show_plots=show_plots,
        plot_manager=plot_manager
    )

    if output_dir:
        logger.info('Saving outputs...')
        csd.to_csv(output_dir / ('csd.csv'))
        extracted_intensity_df.to_csv(output_dir / ('extracted_intensity_df.csv'))
                
        with open(output_dir / ('parameters.txt'),'w') as json_file:
            json.dump(dataclasses.asdict(parameters), json_file, indent=4)
    
    if show_plots:
        logger.info('Plotting post-extraction figures...')
        isotope_distribution_plots(
            ion_spec_pairs,
            nearest_peak_df,
            summary_df,
            parameters,
            plot_manager=plot_manager
        )
        
    logger.info('Extraction complete.')
    return (csd, extracted_intensity_df)


#### Command-line interface

class SmartRawDescriptionHelpFormatter(argparse.HelpFormatter):
    """Help message formatter which retains any formatting in descriptions.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """
    def __init__(self, prog):
        super().__init__(prog, max_help_position=40)
    
    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join(action.option_strings) + ' ' + args_string


    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        text = re.sub(r'[ ]*\|N[ ]*', '\n', text)

        import textwrap
        text = '\n'.join(
            textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent)
            if not line.startswith('|*')
            else
            '\t'  + textwrap.fill(line[2:], width - 8, initial_indent=indent, subsequent_indent=indent).replace('\n','\n\t   ')
            for line in text.splitlines(keepends=True)
        )
        return text 


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='python -m extract_csd',
        description="""
        Extract charge state distributions and scan-to-scan intensity readings of
        charge states for MaxQuant identified peptides from LC-MS data.|N
        |N
        The extraction scheme accepts .mzML files or Bruker .d folders. If
        `--orbitrap` then `ms_data_file` should be a .mzML file
        with profile peaks centroided. If `--timstof` then
        `ms_data_file` should be a .mzML file or Bruker .d folder, and the
        spectrums will be centroided during extraction.
        """,
        epilog="""
        A brief overview of the extraction scheme:|N
            |*1. Reads spectrums from `ms_data_file`.|N
            |*2. Reads start and finish times of identified ions from MaxQuant's
            evidence.txt file.|N
            |*3. Centroids profile peaks (if --timstof enabled).|N
            |*4. Calibrates m/z of spectrums using MaxQuant's identified ions.|N
            |*5. Extracts information about spectrum peaks surrounding the predicted
            isotope distributions of each charge state of each identified peptide.|N
            |*6. Filters peptide charge states based on previously extracted
            information. Collects the estimated spectrum intensity for remaining
            peptide charge states, and saves to output.
        """,
        formatter_class=SmartRawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--orbitrap',
        action='store_true',
        default=False,
        help="""
        Denotes type of mass spectrometer instrument used to generate MS data.
        If enabled, then `ms_data_file` should be a .mzML file with profile 
        peaks centroided.
        """)
    group.add_argument(
        '--timstof',
        action='store_true',
        default=False,
        help = """
        Denotes type of mass spectrometer instrument used to generate MS data.
        If enabled, then then `ms_data_file` should be a .mzML file or Bruker 
        .d folder, and the spectrums will be centroided during extraction.
        """ )

    parser.add_argument(
        '-f',
        '--ms_data_file',
        type=str,
        dest='ms_data_file',
        metavar='FILE',
        help="""
        Path to input mass spectrum data file.
        (required)
        """,
        required=True
    )

    parser.add_argument(
        '-t',
        '--maxquant_txt_dir',
        type=str,
        dest='maxquant_txt_dir',
        metavar='DIR',
        help="""
        Path to MaxQuant's output txt folder. The folder should contain
        evidence.txt and summary.txt. Identified ions from evidence.txt are
        used for extraction. Fixed modifications are read from summary.txt.
        (required)
        """,
        required=True
    )

    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        dest='output_dir',
        metavar='DIR',
        help="""
        Path to save the outputs, properties of the LC-MS run, and
        logging of the extraction pipeline. The outputs, `csd` and
        `extracted_intensity_df`, and a .log file are saved directly to
        `output_dir`. Properties of the LC-MS run are read from `ms_data_file`,
        and saved to `output_dir` / raw_properties. 
        (required)
        """,
        required=True
    )

    parser.add_argument(
        '-r',
        '--raw_file_name',
        type=str,
        dest='raw_file_name',
        metavar='NAME',
        help="""
        Name of the raw MS file, if  different from the filename of
        `ms_data_file`. Used for determining the correct experiment in the
        MaxQuant txt folder.
        (optional)
        """)
    
    parser.add_argument(
        '-p',
        '--save_plots',
        action='store_true',
        help="""
        If enabled, figures generated during extraction will be saved to
        directory `output_dir` / fig /. Requires `output_dir` to be provided.
        (optional)
        """
    )

    results = parser.parse_args()
    return results


def cli():
    arguments = parse_arguments()
    
    ms_data_file = arguments.ms_data_file
    maxquant_txt_dir = arguments.maxquant_txt_dir
    output_dir = arguments.output_dir
    raw_file_name = arguments.raw_file_name
    save_plots = arguments.save_plots
    
    if arguments.orbitrap:
        ms_instrument = 'orbitrap'
    elif arguments.timstof:
        ms_instrument = 'timstof'
    
    extract_csd(
        ms_data_file=ms_data_file,
        maxquant_txt_dir=maxquant_txt_dir,
        ms_instrument=ms_instrument,
        output_dir=output_dir,
        raw_file_name=raw_file_name,
        save_plots=save_plots
    )


