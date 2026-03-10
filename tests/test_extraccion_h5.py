import pandas as pd
import h5py
import os
import tqdm
import numpy as np
from src.data_engine.extractor import FeatureExtractor
from config.hparams import VIDEO_ROOT, CSV_PATH

def prueba_rapida_h5():
    OUT_TEST = 'data/test_30fps.h5'
    os.makedirs('data', exist_ok=True)
    
    # IMPORTANTE: Cargamos los primeros 5 del CSV
    df = pd.read_csv(CSV_PATH).head(5)
    extractor = FeatureExtractor()
    
    print(f"--- Iniciando Grabacion HDF5 (30 FPS + Normalizacion) ---")
    
    # Abrimos en modo 'a' (append) para no borrar lo anterior o 'w' para sobreescribir
    with h5py.File(OUT_TEST, 'w') as f:
        for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Procesando"):
            video_id = row['video_id']
            video_path = extractor.get_video_path(VIDEO_ROOT, video_id)
            
            if os.path.exists(video_path):
                data = extractor.process_video(video_path)
                
                # Verificamos si ya existe para no chocar
                if video_id in f:
                    del f[video_id] # Lo borramos para grabarlo de nuevo limpio
                
                grp = f.create_group(video_id)
                grp.create_dataset('landmarks', data=data)
                grp.attrs['gloss'] = str(row['gloss_asl'])
                print(f" [Guardado] {video_id} - Shape: {data.shape}")
            else:
                print(f" [Salteado] No existe: {video_id}")
    
    print(f"\n--- Test Terminado Exitosamente ---")

if __name__ == '__main__':
    prueba_rapida_h5()