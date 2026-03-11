import h5py
import matplotlib.pyplot as plt
import numpy as np

H5_PATH = 'data/dataset_30fps.h5'
VIDEO_ID = '--6bmFM9wT4' # El primero de tu lista de sospechosos

def visualizar_sospechoso(vid_id):
    with h5py.File(H5_PATH, 'r') as f:
        if vid_id not in f:
            print("ID no encontrado")
            return
        
        data = f[vid_id][:]
        nariz_x = data[:, 0]
        nariz_y = data[:, 1]
        
        plt.figure(figsize=(10, 5))
        plt.plot(nariz_x, label='Movimiento X (Nariz)')
        plt.plot(nariz_y, label='Movimiento Y (Nariz)')
        plt.title(f"Trayectoria de la Nariz - Video: {vid_id}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    visualizar_sospechoso(VIDEO_ID)