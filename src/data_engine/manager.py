import pandas as pd
import h5py
import os
import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.data_engine.extractor import FeatureExtractor
from config.hparams import VIDEO_ROOT, CSV_PATH, H5_OUTPUT

def procesar_video_seguro(video_id):
    try:
        extractor = FeatureExtractor()
        v_path = extractor.get_video_path(VIDEO_ROOT, video_id)
        if not os.path.exists(v_path): return None
        data = extractor.process_video(v_path)
        return (video_id, data)
    except:
        return None

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    video_ids = df['video_id'].unique()
    
    # 1. Verificamos que ya procesamos para no repetir
    procesados = []
    if os.path.exists(H5_OUTPUT):
        with h5py.File(H5_OUTPUT, 'r') as f:
            procesados = list(f.keys())
    
    pendientes = [v for v in video_ids if v not in procesados]
    print(f"--- Extraccion Reanudada ---")
    print(f"Total: {len(video_ids)} | Ya listos: {len(procesados)} | Pendientes: {len(pendientes)}")

    if len(pendientes) == 0:
        print("¡Todo el dataset ya esta procesado!")
    else:
        # 2. Procesamos en lotes pequeños para no romper el Pool
        batch_size = 50
        with h5py.File(H5_OUTPUT, 'a') as f:
            for i in range(0, len(pendientes), batch_size):
                lote = pendientes[i:i+batch_size]
                with ProcessPoolExecutor(max_workers=6) as executor: # Bajamos a 6 para dar mas aire al Xeon
                    resultados = list(tqdm.tqdm(executor.map(procesar_video_seguro, lote), 
                                              total=len(lote), 
                                              desc=f"Lote {i//batch_size + 1}"))
                    for res in resultados:
                        if res and res[0] not in f:
                            f.create_dataset(res[0], data=res[1])