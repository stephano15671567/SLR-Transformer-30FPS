import pandas as pd
import h5py
import os
import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.data_engine.extractor import FeatureExtractor
from config.hparams import VIDEO_ROOT, CSV_PATH, H5_OUTPUT

def time_to_sec(t):
    if isinstance(t, (float, int)): return float(t)
    h, m, s = str(t).split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def procesar_lote_video(video_id, grupo_clips):
    extractor = FeatureExtractor()
    v_path = extractor.get_video_path(VIDEO_ROOT, video_id)
    
    if not os.path.exists(v_path): return None
    
    # Extraemos TODO el video a 30 FPS de una sola vez (Eficiencia Xeon)
    try:
        data_full = extractor.process_video(v_path)
        return (video_id, data_full, grupo_clips)
    except:
        return None

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    df['s_sec'] = df['start_time'].apply(time_to_sec)
    
    # Agrupamos para abrir cada video una sola vez
    grupos = list(df.groupby('video_id'))
    
    print(f"--- Iniciando Extracción Masiva Xeon V6 (30 FPS) ---")
    
    # Usamos h5py para ir guardando en tiempo real (más seguro que npy sueltos)
    with h5py.File(H5_OUTPUT, 'a') as f:
        with ProcessPoolExecutor(max_workers=10) as exe:
            # Enviamos los videos a los 10 núcleos del Xeon
            tareas = [exe.submit(procesar_lote_video, vid, g) for vid, g in grupos]
            
            for t in tqdm.tqdm(tareas, desc="Progreso Videos"):
                res = t.result()
                if res:
                    vid, data, clips = res
                    if vid not in f:
                        f.create_dataset(vid, data=data)
                        f[vid].attrs['total_frames'] = len(data)
