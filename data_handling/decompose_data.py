from signal_decomposition.eemd import EEMDWrapper
from signal_decomposition.wavelet import WaveletWrapper
from data_loader import load_mavir_data
import pandas as pd
from utils import min_max_norm

if __name__ == '__main__':
    df = load_mavir_data('../data/mavir_data/mavir.csv')
    df['Power'] = min_max_norm(df['Power'].values)

    eemdWrapper = EEMDWrapper(df['Power'].values)
    imfs = eemdWrapper.get_imfs().tolist()
    residue = eemdWrapper.get_residue().tolist()

    new_df = pd.DataFrame({k: v for k, v in
                           zip(['Time', 'Original', 'Residue'] + [f'IMF-{i}' for i in range(len(imfs))],
                               [df['Time'].values, df['Power'].values, residue] + imfs[::-1])})

    new_df.to_csv('../data/mavir_data/mavir_eemd.csv', index=False)

    print('EMD done!')

