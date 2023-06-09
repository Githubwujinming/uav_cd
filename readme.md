# dependency

pytorch >= 1.11
python >= 3.8
lightnint >= 2.0
CUDA >= 11.7
cv2 
scipy
yacs
jsonargparse
pytorch-lightning[extra]

# commands
python main.py fit --config configs/models/cd.yaml --config configs/data/sup_data_100.yaml --config configs/trainer/gpu0.yaml
