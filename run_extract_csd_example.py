import zipfile
import extract_csd

if __name__ == '__main__':
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
        
    print(csd)
    print(extracted_intensity_df)
