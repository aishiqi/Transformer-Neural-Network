git clone https://github.com/aishiqi/Transformer-Neural-Network.git
cd Transformer-Neural-Network
pip install -r requirements.txt
python Transformer_Trainer.py
wandb_token_513......ba7

nohup python Transformer_Trainer.py &
nohup watch -n 100 wandb artifact cache cleanup 10GB &