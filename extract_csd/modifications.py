# builtin
import os
import json
import re

# external
import numpy as np
from pyteomics import mass as pmass

# local
from .utils import LIB_PATH

# `mod_comp` contains compositions modifications as specified by pyteomics'
# modX sequence representation
with open(os.path.join(LIB_PATH, 'mod_comp.json')) as json_file:
    mod_comp = json.load(json_file)
mod_comp = { k : pmass.Composition(v) for k,v in mod_comp.items()}

AA_COMP = dict(pmass.std_aa_comp)
AA_COMP.update(mod_comp)


# `maxquant_mod_to_modx` contains conversions of modifications from MaxQuant's
# representation to pyteomics' modX framework
with open(os.path.join(LIB_PATH, 'maxquant_mod_to_modx.json')) as json_file:
    maxquant_mod_to_modx = json.load(json_file)
    

def translate_maxquant_modified_sequence(sequence, fixed_modifications=None):
    """
    Translates modfications in a sequence from from MaxQuant's representation
    to pyteomics' modX representation.
    
    Modifications are translated as specified in `maxquant_mod_to_modx`. Fixed
    modifications are also made explicit if `fixed_modifications` are given.
    
    Parameters
    ----------
    sequence : str
        A sequence with modifications represented using MaxQuant's
        representation.
    
    fixed_modifications : list of str or str
        A list of fixed modifications represented using MaxQuant's
        representation.
        
    Returns
    -------
    str
        A sequence with modifications represented using pyteomics' modX
        representation with all fixed modifications made explicit.
    
    Examples
    --------
    >>> translate_maxquant_modified_sequence(
    ... '_(Acetyl (Protein N-term))ADKPDM(Oxidation (M))GEICASFDK_'
    ... 'Carbamidomethyl (C)'
    ... )
    'ac-ADKPDoxMGEIcamCASFDK'
    """
    if sequence[0] != '_':
        sequence = '_' + sequence
    if sequence[-1] != '_':
        sequence = sequence + '_'
        
    # Make fixed modifications explicit
    if fixed_modifications is not None:
        if isinstance(fixed_modifications, str):
            fixed_modifications = [fixed_modifications]
            
        for modification in fixed_modifications:
            amino_acids = re.search(r'\(([A-Z]+)\)', modification)[1]
            
            left = np.array([ char == '(' for char in sequence ])
            right = np.array([ char == ')' for char in sequence ])
            parentheses_depth = np.cumsum(left) - np.cumsum(right)
            
            sequence = ''.join([
                char + '(' + modification + ')'
                if char in amino_acids and depth == 0 and next_char != '('
                else char
                for depth, char, next_char
                in zip(parentheses_depth, sequence, sequence[1:] + ' ')
                ])
    
    # Translate from MaxQuant to modX
    for mq_mod_label, modx_label in maxquant_mod_to_modx.items():
        sequence = re.sub(
            '([A-Z_])' + re.escape('(' + mq_mod_label + ')'),
            modx_label + r'\g<1>',
            sequence
        )

    sequence = sequence.replace('_', '-')
    sequence = sequence.strip('-')
    return sequence

