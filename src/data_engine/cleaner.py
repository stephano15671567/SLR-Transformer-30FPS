import h5py
import numpy as np
import os
from tqdm import tqdm

# Rutas de archivos definidas en hparams o relativas
ORIGINAL_H5 = 'data/dataset_30fps.h5'
CLEAN_H5 = 'data/dataset_30fps_cleaned.h5'

def apply_linear_interpolation(data, error_indices):
    """
    Corrige saltos bruscos en los landmarks mediante interpolacion lineal simple.
    """
    for idx in error_indices:
        # Evitar desbordamiento de indice al final del array
        if idx + 1 < data.shape[0]:
            # El frame afectado se promedia con los frames adyacentes
            data[idx+1] = (data[idx] + data[min(idx+2, data.shape[0]-1)]) / 2
    return data

def run_dataset_cleaning(threshold=0.15, max_gap=10):
    if not os.path.exists(ORIGINAL_H5):
        print(f"Error: {ORIGINAL_H5} not found.")
        return

    with h5py.File(ORIGINAL_H5, 'r') as f_in, h5py.File(CLEAN_H5, 'w') as f_out:
        video_ids = list(f_in.keys())
        print(f"INFO: Initing dataset cleaning. Total: {len(video_ids)} samples.")

        processed_count = 0
        discarded_count = 0

        for vid_id in tqdm(video_ids, desc="Processing"):
            data = f_in[vid_id][:]
            if data.shape[0] < 2:
                discarded_count += 1
                continue

            # Analisis de discontinuidades espaciales en el landmark de referencia (nariz)
            # data[:, :2] asume que las primeras dos columnas son X e Y
            reference_coords = data[:, :2]
            deltas = np.linalg.norm(np.diff(reference_coords, axis=0), axis=1)
            error_indices = np.where(deltas > threshold)[0]

            # Criterio de inclusion: el ruido debe ser menor al limite de interpolacion (max_gap)
            if len(error_indices) < max_gap:
                if len(error_indices) > 0:
                    data = apply_linear_interpolation(data, error_indices)
                
                f_out.create_dataset(vid_id, data=data, compression="gzip", compression_opts=4)
                processed_count += 1
            else:
                discarded_count += 1

        print("\n--- Summary Report ---")
        print(f"Samples Processed: {processed_count}")
        print(f"Samples Discarded: {discarded_count}")
        print(f"Output File: {CLEAN_H5}")

if __name__ == "__main__":
    run_dataset_cleaning()