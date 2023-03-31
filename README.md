# extract-csd
Extraction scheme to extract charge state distributions from raw LC-MS/MS data.


* [**extract-csd**](#alphatims)
  * [**Installation**](#installation)
  * [**Run Time and RAM Requirements**](#run-time-and-ram-requirements)
  * [**Preprocessing Raw LC-MS/MS Files**](#preprocessing-raw-lc-msms-files)
    * [**Thermo .raw files**](#thermo-raw-files)
    * [**Bruker .d folders**](#bruker-d-folders)
  * [**Usage**](#usage)
    * [**Command-line interface**](#command-line-interface)
    * [**Python package**](#python-package)
  * [**Citation**](#citation)

## Installation
The extraction scheme has the following Python package dependencies. They can be installed using "pip install" or "conda" (tested versions for Python 3.8.12 indicated):
- [*numpy*](https://pypi.org/project/numpy/) = 1.19.5
- [*scipy*](https://pypi.org/project/scipy/) = 1.7.1
- [*pandas*](https://pypi.org/project/pandas/) = 1.3.4
- [*pyteomics*](https://pypi.org/project/pyteomics/) = 4.5.1
- [*lxml*](https://pypi.org/project/lxml/) = 4.6.3
- [*tqdm*](https://pypi.org/project/tqdm/) = 4.62.3
- [*matplotlib*](https://pypi.org/project/matplotlib/) = 3.5.0 (optional; required for plotting and saving figures generated during extraction)

To process Bruker .d folders, an additional package is required

- [*alphatims*](https://pypi.org/project/alphatims/) = 1.0.0

along with the required Bruker libraries, as described in [AlphaTims](https://github.com/MannLabs/alphatims).

The figure generating code requires following additional packages:
- [*matplotlib*](https://pypi.org/project/matplotlib/) = 3.5.0
- [*seaborn*](https://pypi.org/project/seaborn/) = 0.11.2
- [*scikit-learn*](https://pypi.org/project/scikit-learn/) = 1.01
- [*pygam*](https://pypi.org/project/pygam/) = 0.8.0

## Run Time and RAM requirements
For a 400MB .mzML file containing MS1 scans only (a typical size when converting a 2GB LC-MS/MS .raw file), the extraction scheme takes approximately 30 minutes on a standard laptop/desktop.

The extraction scheme requires more RAM than the .mzML file or .d folder, as it loads all the MS1 spectrums into memory. It should be sufficient to have 3x times the size of the .mzML file or .d folder (containing MS1 scans only) in RAM.

## Preprocessing Raw LC-MS/MS Files
### Thermo .raw files
Prior to running the extraction scheme, Thermo .raw files should be converted into .mzML format with profile peaks centroided.
This can be achieved with data conversion tools, such as [msConvert](https://proteowizard.sourceforge.io/tools/msconvert.html), which is found in the [ProteoWizard](https://proteowizard.sourceforge.io/) toolkit.

### Bruker .d folders
Bruker .d folders can be read directly by the extraction scheme, provided that the above libraries and Python packages are installed.

Alternatively, Bruker .d folders can be converted into .mzML format (see above). Profile peaks in Bruker .d folders do not require centroiding; the extraction scheme will centroid the profile peaks.

## Usage
### Command-line interface
The extraction scheme can be run with the following command:

```bash
python -m extract_csd [OPTIONS]
```

with arguments as described in the help `-h` flag:

```
usage: python -m extract_csd [-h] (--orbitrap | --timstof) -f FILE -d DIR -o DIR [-r NAME] [-p]

Extract charge state distributions and scan-to-scan intensity readings of charge states for MaxQuant
identified peptides from LC-MS data.

The extraction scheme accepts .mzML files or Bruker .d folders. If `--orbitrap` then `ms_data_file` should
be a .mzML file with profile peaks centroided. If `--timstof` then `ms_data_file` should be a .mzML file or
Bruker .d folder, and the spectrums will be centroided during extraction.

optional arguments:
  -h, --help                  show this help message and exit
  --orbitrap                  Denotes type of mass spectrometer instrument used to generate MS data. If
                              enabled, then `ms_data_file` should be a .mzML file with profile peaks
                              centroided.
  --timstof                   Denotes type of mass spectrometer instrument used to generate MS data. If
                              enabled, then then `ms_data_file` should be a .mzML file or Bruker .d
                              folder, and the spectrums will be centroided during extraction.
  -f, --ms_data_file FILE     Path to input mass spectrum data file. (required)
  -t, --maxquant_txt_dir DIR  Path to MaxQuant's output txt folder. The folder should contain evidence.txt
                              and summary.txt. Identified ions from evidence.txt are used for extraction.
                              Fixed modifications are read from summary.txt. (required)
  -o, --output_dir DIR        Path to save the outputs, properties of the LC-MS run, and logging of the
                              extraction scheme. The outputs, `csd` and `extracted_intensity_df`, and a
                              .log file are saved directly to `output_dir`. Properties of the LC-MS run
                               are read from `ms_data_file`, and saved to `output_dir` / raw_properties.
                               (required)
  -r, --raw_file_name NAME    Name of the raw MS file, if different from the filename of `ms_data_file`.
                              Used for determining the correct experiment in the MaxQuant txt folder.
                              (optional)
  -p, --save_plots            If enabled, figures generated during extraction will be saved to directory
                              `output_dir` / fig /. Requires `output_dir` to be provided. (optional)

A brief overview of the extraction scheme:
        1. Reads spectrums from `ms_data_file`.
        2. Reads start and finish times of identified ions from MaxQuant's evidence.txt file.
        3. Centroids profile peaks (if --timstof enabled).
        4. Calibrates m/z of spectrums using MaxQuant's identified ions.
        5. Extracts information about spectrum peaks surrounding the predicted isotope distributions of
           each charge state of each identified peptide.
        6. Filters peptide charge states based on previously extracted information. Collects the estimated
           spectrum intensity for remaining peptide charge states, and saves to output.
```

The following is an example command using the example data in `/data/example_input/`:

```bash
unzip -n 20210420_MJ_LFQ_Hela_standard_gradient_short_Voltage_1p5_01.zip -d ./data/example_input
unzip -n txt.zip -d ./data/example_input
python -m extract_csd --orbitrap -f ./data/example_input/20210420_MJ_LFQ_Hela_standard_gradient_short_Voltage_1p5_01.mzML -t ./data/example_input/txt/ -o ./output/ -p
```

### Python package
The extraction scheme can also be run as a python package. To do so, first import the extract_csd package:
```python
import extract_csd
```
Then the extraction can be run as:
```python
csd, extracted_intensity_df = extract_csd.extract_csd(...)
```

with arguments as described in the docstring:

```python
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
```

The following is an example code using the example data in `/data/example_input/` (also found in run_extract_csd_example.py):

```python
import zipfile
import extract_csd

with zipfile.ZipFile('./data/example_input/20210420_MJ_LFQ_Hela_standard_gradient_short_Voltage_1p5_01.zip', 'r') as zip_ref:
    zip_ref.extractall('./data/example_input/')
    
with zipfile.ZipFile('./data/example_input/txt.zip', 'r') as zip_ref:
    zip_ref.extractall('./data/example_input/')
    
csd, extracted_intensity_df = extract_csd.extract_csd(
    ms_data_file='./data/example_input/20210420_MJ_LFQ_Hela_standard_gradient_short_Voltage_1p5_01.mzML',
    maxquant_txt_dir='./data/example_input/txt/',
    ms_instrument='orbitrap',
    output_dir='./output/',
    parameters=None,
    raw_file_name=None,
    show_plots=True,
    save_plots=True
    )
```

## Citation
Xu, A. M., Tang, L. C., Jovanovic, M. & Regev, O. A high-throughput approach reveals distinct
peptide charging behaviors in electrospray ionization mass spectrometry. (2023). Manuscript in submission.
