import numpy as np
from scipy.stats import binom
from scipy.special import gammaln, xlogy
from functools import reduce
from pyteomics import mass as pmass


# proton_mass = 1.00727646677
proton_mass = pmass.nist_mass['H+'][0][0]

_element_list  = ['H','C','N','O','S']

_permissible_element_isotopes = {
    'H' : [1,2],
    'C' : [12,13],
    'N' : [14,15],
    'O' : [16,17,18],
    'S' : [32,33,34,36]
    }

_element_iso_atomic_num = {
    element : np.array(nums) for element, nums in _permissible_element_isotopes.items()
    }

_element_iso_mass = {
    element : np.array([
        pmass.nist_mass[element][i][0] for i in _permissible_element_isotopes[element]
        ]) for element in _element_list
    }

_element_iso_abundance = {
    element : np.array([
        pmass.nist_mass[element][i][1] for i in _permissible_element_isotopes[element]
        ]) for element in _element_list
    }


def element_isotope_peaks(element, n, num_peaks=7):
    if n == 0:
        raise Exception('Number of given element needs to be > 0.')
    
    if not isinstance(element,str):
        raise TypeError('Element needs to be a string.')
    
    if element in {'H','C','N'}:
        p = _element_iso_abundance[element][1]
        prob = binom.pmf(np.arange(0,num_peaks),n,p)
        mass = np.arange(n,n-num_peaks,-1)*_element_iso_mass[element][0] + np.arange(0,num_peaks)*_element_iso_mass[element][1]
        return mass,prob
    else:
        num_isotopes = len(_element_iso_abundance[element])
        p = _element_iso_abundance[element]
        extra_neutrons = _element_iso_atomic_num[element][1:] - _element_iso_atomic_num[element][0]
        x = np.moveaxis( np.array(np.meshgrid(*((np.arange(0,num_peaks),)*(num_isotopes-1)))) ,0,-1).reshape(-1,num_isotopes-1)

        keep = np.where((np.sum(x,1) <= n) & (np.dot(x,extra_neutrons) < num_peaks))
        x = x[keep]
        x = np.concatenate( (n-np.sum(x,1,keepdims=True), x),axis=1)
        
        log_prob = gammaln(n+1) + np.sum(xlogy(x, p) - gammaln(x+1), axis=-1)
        prob = np.exp(log_prob)
        mass = np.dot(x,_element_iso_mass[element])
        return mass,prob


def peptide_isotope_peaks(peptide, charge=None, mean_combined=False, argmax_combined=False, num_peaks=7, prob_cutoff=0, return_numpy=True, aa_comp=pmass.std_aa_comp):
    assert type(peptide) == str
    
    element_count = pmass.Composition(sequence = peptide, aa_comp=aa_comp)
    
    element_peaks_list = [np.array(element_isotope_peaks(el, element_count[el], num_peaks=num_peaks)) for el in _element_list if element_count[el] > 0]
    
    prob = reduce(np.multiply,np.ix_(*( peaks[1] for peaks in element_peaks_list)))
    mass = reduce(np.add,np.ix_(*( peaks[0] for peaks in element_peaks_list)))
    keep_index = np.where((mass - np.min(mass) < num_peaks-0.5) & (prob > prob_cutoff))
    prob = prob[keep_index]
    mass = mass[keep_index]
    
    if argmax_combined:
        # Combines m/z at the location of the maximum
        # Combines intensities by summing intensities
        extra_neutrons = np.round(mass - mass.min())
        mass_combined = np.array([ mass[np.argmax(prob*(extra_neutrons==i))] for i in range(num_peaks)])
        prob_combined = np.array([ prob[extra_neutrons == i].sum() for i in range(num_peaks) ])
        
        mass = mass_combined
        prob = prob_combined
     
    if mean_combined:
        # Combines m/z at the location of the mean of m/z's for that # peak
        # Combines intensities by summing intensities
        extra_neutrons = np.round(mass - mass.min())
        
        mass_combined = np.array([np.dot(
            mass[extra_neutrons==i],
            prob[extra_neutrons==i]/sum(prob[extra_neutrons==i])
            ) for i in range(num_peaks) ])
        prob_combined = np.array([ prob[extra_neutrons == i].sum() for i in range(num_peaks) ])
        
        mass = mass_combined
        prob = prob_combined
     
    
    output = (mass,prob)
    if charge is not None:
        output = (mass/charge + proton_mass,prob)
    
    if return_numpy:
        return np.array(output)
    
    return output
    