import h5py
import numpy as np
import os

H5_PATH = 'data/dataset_30fps.h5'

def run_integrity_diagnostics(threshold_error=0.15, max_gap_fixable=10):
    """
    Evalúa la viabilidad de la recuperación de datos mediante interpolación lineal.
    """
    if not os.path.exists(H5_PATH):
        print(f"ERROR: Archivo objetivo {H5_PATH} no encontrado.")
        return

    with h5py.File(H5_PATH, 'r') as f:
        video_ids = list(f.keys())
        total_samples = len(video_ids)
        
        reparable_samples = 0
        discard_samples = 0
        nominal_samples = 0

        print(f"INFO: Ejecutando diagnóstico en {total_samples} muestras...")

        for vid_id in video_ids:
            data = f[vid_id][:]
            if data.shape[0] < 2: 
                discard_samples += 1
                continue

            # Análisis de trayectoria de referencia (Nariz/Centro del torso)
            reference_trajectory = data[:, :2]
            spatial_discontinuities = np.linalg.norm(np.diff(reference_trajectory, axis=0), axis=1)
            
            # Detectar índices de anomalías
            error_indices = np.where(spatial_discontinuities > threshold_error)[0]
            
            if len(error_indices) == 0:
                nominal_samples += 1
            else:
                # Criterio de inclusión basado en la densidad de error temporal
                if len(error_indices) < max_gap_fixable:
                    reparable_samples += 1
                else:
                    discard_samples += 1

        print("\n" + "="*40)
        print("REPORTE DE DIAGNÓSTICO DE INTEGRIDAD")
        print("="*40)
        print(f"Muestras Nominales (Óptimas): {nominal_samples} ({(nominal_samples/total_samples)*100:.2f}%)")
        print(f"Muestras Recuperables:       {reparable_samples} ({(reparable_samples/total_samples)*100:.2f}%)")
        print(f"Sugeridas para Descarte:     {discard_samples} ({(discard_samples/total_samples)*100:.2f}%)")
        print("-" * 40)
        print(f"Total de Muestras Utilizables: {nominal_samples + reparable_samples}")
        print("="*40)

if __name__ == "__main__":
    run_integrity_diagnostics()