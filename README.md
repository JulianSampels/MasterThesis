# Code for Master Thesis

## PathE: Leveraging Entity-Agnostic Paths for Parameter-Efficient Knowledge Graph Embeddings
### Model overview
![pathe_v1](https://github.com/user-attachments/assets/c2cf52b1-8549-441a-9c0a-cd59aa1793af)

### Instructions:

1. Install cuda 11.8  
2. Make sure cuda 11.8 is used by calling  
   - `nvcc --version`
3. Install required dependencies by running `pip install -r PathE/pathe/requirements.txt` in the terminal
4. Generate data by running 
   - `./run.py`

#### Training 

1. FB15k-237
   - `python -m PathE.pathe.runner train pathe --log_dir ./logs --expname test1FB15k-237 --wandb_project fb15k237 --train_paths ./data/path_datasets/fb15k237/train/ --valid_paths ./data/path_datasets/fb15k237/val/ --test_paths ./data/path_datasets/fb15k237/test/ --max_ppt 8 --batch_size 4096 --nhead 2 --num_encoder_layers 1 --dim_feedforward 256 --embedding_dim 64 --num_negatives 99 --monitor valid_link_mrr --patience 5 --dropout 0.1 --lrate 1e-3 --device cuda --num_devices 1 --val_num_negatives 99 --num_workers 8 --ent_aggregation transformer --lp_loss_fn ce --accumulate_gradient 8 --label_smoothing 0.01 --node_projector dummy --path_setup 20_10 --num_agg_layers 1 --seed 7`

#### Full Evaluation

1. FB15k-237
   - `python runner.py full_eval pathe --log_dir [YOUR LOG DIR] --expname [YUR EXP NAME] --wandb_project fb15k237 --train_paths [PATH TO TRAIN DATA] --valid_paths [PATH TO VALID DATA] --test_paths [PATH TO TEST DATA] --max_ppt 8 --batch_size 4096 --nhead 2 --num_encoder_layers 1 --dim_feedforward 256 --embedding_dim 64 --num_negatives 99 --monitor valid_link_mrr --patience 5 --dropout 0.1 --lrate 1e-3 --device cuda --num_devices 1 --val_num_negatives 99 --num_workers 8 --ent_aggregation transformer --lp_loss_fn ce --accumulate_gradient 8 --label_smoothing 0.01 --node_projector dummy --path_setup 20_10 --num_agg_layers 1 --seed 7 --checkpoint [PATH TO MODEL CHECKPOINT]`

## Currently working commands wip
### train
1. train triples small-codex without wandb
   - `python -m PathE.pathe.runner train patheTuples --log_dir ./logs --expname test1Tuplescodex-small --train_paths ./data/path_datasets/codex-small/train/ --valid_paths ./data/path_datasets/codex-small/val/ --test_paths ./data/path_datasets/codex-small/test/ --max_ppt 8 --batch_size 2048 --nhead 2 --num_encoder_layers 1 --dim_feedforward 256 --embedding_dim 64 --num_negatives 0 --monitor valid_mrr --patience 5 --dropout 0.1 --lrate 1e-3 --device cuda --num_devices 1 --val_num_negatives 0 --num_workers 8 --ent_aggregation transformer --lp_loss_fn ce --accumulate_gradient 8 --label_smoothing 0.01 --node_projector dummy --path_setup 20_10 --num_agg_layers 1 --seed 7`
2. train tuples small-codex without wandb
   - `python -m PathE.pathe.runner train patheTuples --log_dir ./logs --expname test1Tuplescodex-small --train_paths ./data/path_datasets/codex-small/train/ --valid_paths ./data/path_datasets/codex-small/val/ --test_paths ./data/path_datasets/codex-small/test/ --max_ppt 8 --batch_size 4096 --nhead 2 --num_encoder_layers 1 --dim_feedforward 256 --embedding_dim 64 --num_negatives 0 --monitor valid_link_mrr --patience 5 --dropout 0.1 --lrate 1e-3 --device cuda --num_devices 1 --val_num_negatives 0 --num_workers 8 --ent_aggregation transformer --lp_loss_fn ce --accumulate_gradient 8 --label_smoothing 0.01 --node_projector dummy --path_setup 20_10 --num_agg_layers 1 --seed 7`

### test
   - `python -m PathE.pathe.runner test patheTuples --log_dir ./logs --expname tupleFb15k237negatives50 --train_paths ./data/path_datasets/fb15k237/train/ --valid_paths ./data/path_datasets/fb15k237/val/ --test_paths ./data/path_datasets/fb15k237/test/ --max_ppt 8 --batch_size 2048 --nhead 2 --num_encoder_layers 1 --dim_feedforward 256 --embedding_dim 64 --num_negatives 50 --monitor valid_mrr --patience 5 --dropout 0.1 --lrate 1e-3 --device cuda --num_devices 1 --val_num_negatives 50 --num_workers 16 --ent_aggregation transformer --lp_loss_fn ce --accumulate_gradient 8 --label_smoothing 0.01 --node_projector dummy --path_setup 20_10 --num_agg_layers 1 --seed 7 --version 1`