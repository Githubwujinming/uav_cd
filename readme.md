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
omegaconf
albumentations
pynecone

# commands
nohup python main.py fit --config configs/models/cd2.yaml --config configs/data/sup_data_100.yaml --config configs/trainer/gpu1.yaml -n levir_dsahrn  > nohuplogs/levir_dsahrn.log 2>&1 &
582861