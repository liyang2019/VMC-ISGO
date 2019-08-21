# Importance Sampling Gradient Optimization (ISGO) method for variational Monte Carlo (VMC) algorithm.

This is the Tensorflow implementation for the ISGO method for VMC algorithm. https://arxiv.org/abs/1905.10730

<div align="left">
<img src="figures/ISGO_flow_chart.png?raw=true" alt="ISGO flow chart" width="400"></img>
</div>

The VMC algorithm has two stages running alternately: a Markov-Chain Monte Carlo (MCMC) sampling stage and an optimization stage. In the MCMC sampling stage, ```N_sample``` states are generated, then in the optimization stage the parameters are updated using the ```N_sample``` states. The computation in the optimization stage can be parallelized and run on GPU. However, conventional VMC algorithm only allows updating parameters only once in the optimization stage. Inspired from the Off-Policy method in Reinforcement Learning (RL), we developed the ISGO method which allows updating parameters multiple times in the optimization stage.

### To run the code using command line:
```bash
OUTPUT_DIR="result/test"
python main.py \
  --output_dir="${OUTPUT_DIR}" \
  --system="sun_spin1d" \
  --mode="train" \
  --n_print=1 \
  --n_print_optimization=10 \
  --n_save=10 \
  --n_iter=100 \
  --n_sample=20000 \
  --n_sample_initial=1000 \
  --n_sample_warmup=10 \
  --n_optimize_1=100 \
  --n_optimize_2=10 \
  --change_n_optimize_at=50 \
  --learning_rate_1=1e-3 \
  --learning_rate_2=1e-4 \
  --change_learning_rate_at=30 \
  --n_sites=10 \
  --n_spin=2 \
  --layers=2 \
  --filters=4 \
  --kernel=3 \
  --network="CNN" \
  --state_encoding="one_hot"
```
The above command train the model for a ```N_site=10``` 1D SU(2) spin chain. 
A two layers CNN with one-hot state encoding is used. The number of filters is 4 and kernel size is 3, which are the same for all the hidden layers. The results will be saved in the folder ```result/test```.
You are encouraged to play with the flags in ```main.py```


### To measure the local energies:
```bash
python main.py \
  --output_dir="${OUTPUT_DIR}" \
  --system="sun_spin1d" \
  --mode="energy" \
  --n_sample=50000 \
  --n_batch=5000 \
  --n_sites=10 \
  --n_spin=2 \
  --layers=2 \
  --filters=4 \
  --kernel=3 \
  --network="CNN" \
  --state_encoding="one_hot" \
  --load_model_from="${OUTPUT_DIR}/models/latest_model.ckpt-100"
```
The above command load a previous checkpoint from ```result/test/models/latest_model.ckpt-100```.
The local energies for 50000 samples are calculated. 

### To measure the loop correlation functions:
```bash
for origin_site_index in 0 1 2 3 4 5 6 7 8 9
do
python main.py \
  --output_dir="${OUTPUT_DIR}" \
  --system="sun_spin1d" \
  --mode="loop_correlator" \
  --n_sample=50000 \
  --n_batch=5000 \
  --n_sites=10 \
  --n_spin=2 \
  --layers=2 \
  --filters=4 \
  --kernel=3 \
  --network="CNN" \
  --state_encoding="one_hot" \
  --load_model_from="${OUTPUT_DIR}/models/latest_model.ckpt-100" \
  --origin_site_index=${origin_site_index}
done
```
The above command load a previous checkpoint from ```result/test/models/latest_model.ckpt-100```.
The loop correlation functions with different origin site index for 50000 samples are calculated. 

### As of 2019-8-18, the code can be run on colab using GPU by the following way:
First upload the code to google drive to ```<VMC-ISGO FOLDER NAME>```, then mount the drive.
```
import os
from google.colab import drive
os.makedirs('drive', exist_ok=True)
drive.mount('/content/drive/', force_remount=True)
os.chdir('/content/drive/My Drive/<VMC-ISGO FOLDER NAME>')
```

And the code can be run in the following way in colab
```
OUTPUT_DIR="result/test12"
!python main.py \
  --output_dir={OUTPUT_DIR} \
  --system="sun_spin1d" \
  --mode="train" \
  --n_print=1 \
  --n_print_optimization=10 \
  --n_save=10 \
  --n_iter=100 \
  --n_sample=20000 \
  --n_sample_initial=1000 \
  --n_sample_warmup=10 \
  --n_optimize_1=100 \
  --n_optimize_2=10 \
  --change_n_optimize_at=50 \
  --learning_rate_1=1e-3 \
  --learning_rate_2=1e-4 \
  --change_learning_rate_at=30 \
  --n_sites=10 \
  --n_spin=2 \
  --layers=2 \
  --filters=4 \
  --kernel=3 \
  --network="CNN" \
  --state_encoding="one_hot"
```
Both CPU and GPU are supported. To use GPU, select GPU in the colab notebook setting under Edit/Notebook settings/Hardware accelerator.

