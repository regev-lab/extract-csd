# builtin
import os, sys

# external
import numpy as np

# local
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except:
    current_dir = os.getcwd()    
sys.path.insert(0, os.path.join(current_dir, ".."))

from extract_csd.main import peptide_monoiso_mass
from extract_csd.modifications import translate_maxquant_modified_sequence



"""
Unit tests to verify that the following work:
    * translating modified sequences from MaxQuant's modification
      representation to pyteomics' modX representation.
    * calculating peptide masses.
"""

if __name__ == '__main__':
    
    sequence = '_(Acetyl (Protein N-term))ADKPDM(Oxidation (M))GEIC(Carbamidomethyl (C))ASFDK_'
    assert translate_maxquant_modified_sequence(sequence) == 'ac-ADKPDoxMGEIcamCASFDK'
    
    sequence = '_(Acetyl (Protein N-term))ADKPDM(Oxidation (M))GEICASFDK_'
    assert translate_maxquant_modified_sequence(sequence, 'NEM (C)') == 'ac-ADKPDoxMGEInemCASFDK'
    
    sequence = '_YYEIQNAPEQACL_'
    assert translate_maxquant_modified_sequence(sequence) == 'YYEIQNAPEQACL'
    
    
    sequence = 'NYQWTEQRTFPYMRTIVRYYFPTWYLSKSSTVLPTKQVEVSLDNWPPNPWVYMWQEQTK' \
        'QSSTHSRNWPSCWMQQQGLSTAKVYMVIQVFEHRWSSWVVNWKLLIYWYYLSRYTRWWTNGQYRLV' \
        'GRYGSYHPRWNMTIHIRSSCRNNVRMTWPTMGTIPFMDFMHNKPHWNKIPQLVYIYYMRVTKKLWV' \
        'PVYNRVQVMDGQHYTYWWS'
    assert np.isclose(peptide_monoiso_mass(sequence.replace('C','camC')), 26391.922332347225)
    assert np.isclose(peptide_monoiso_mass(sequence.replace('C','nemC')), 26527.974759372868)
    assert np.isclose(peptide_monoiso_mass(sequence.replace('M','oxM')), 26453.82346325121)
    assert np.isclose(peptide_monoiso_mass('ac-' + sequence), 26319.88996711975)
    
    print('All tests passed.')