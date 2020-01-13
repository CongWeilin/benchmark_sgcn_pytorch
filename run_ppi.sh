python main_experiments.py --dataset ppi --batch_size 256 --samp_num 256 --cuda $1 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 > logs/train_log_ppi_256.txt
python main_experiments.py --dataset ppi --batch_size 512 --samp_num 512 --cuda $1 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 > logs/train_log_ppi_512.txt
python main_experiments.py --dataset ppi --batch_size 1024 --samp_num 1024 --cuda $1 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 > logs/train_log_ppi_1024.txt
python main_experiments.py --dataset ppi --batch_size 2048 --samp_num 2048 --cuda $1 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 > logs/train_log_ppi_2048.txt

# python main_experiments.py --dataset ppi-large --batch_size 256 --samp_num 256 --cuda 1 --is_ratio 0.2 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset ppi-large --batch_size 512 --samp_num 512 --cuda 1 --is_ratio 0.2 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset ppi-large --batch_size 1024 --samp_num 1024 --cuda 1 --is_ratio 0.2 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset ppi-large --batch_size 2048 --samp_num 2048 --cuda 1 --is_ratio 0.2 --batch_num 20 --n_stops 1000

# python main_experiments.py --dataset flickr --batch_size 256 --samp_num 256 --cuda 1 --is_ratio 0.5 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset flickr --batch_size 512 --samp_num 512 --cuda 1 --is_ratio 0.5 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset flickr --batch_size 1024 --samp_num 1024 --cuda 1 --is_ratio 0.5 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset flickr --batch_size 2048 --samp_num 2048 --cuda 1 --is_ratio 0.5 --batch_num 20 --n_stops 1000


# python main_experiments.py --dataset reddit --batch_size 256 --samp_num 256 --cuda 1 --is_ratio 0.05 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset reddit --batch_size 512 --samp_num 512 --cuda 1 --is_ratio 0.05 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset reddit --batch_size 1024 --samp_num 1024 --cuda 1 --is_ratio 0.05 --batch_num 20 --n_stops 1000
# python main_experiments.py --dataset reddit --batch_size 2048 --samp_num 2048 --cuda 1 --is_ratio 0.05 --batch_num 20 --n_stops 1000

# python main_experiments.py --dataset yelp --batch_size 256 --samp_num 256 --cuda -1 --is_ratio 0.01 --batch_num 20 --n_stops 1000 > train_log_yelp_256.txt
# python main_experiments.py --dataset yelp --batch_size 512 --samp_num 512 --cuda -1 --is_ratio 0.01 --batch_num 20 --n_stops 1000 > train_log_yelp_512.txt
# python main_experiments.py --dataset yelp --batch_size 1024 --samp_num 1024 --cuda -1 --is_ratio 0.01 --batch_num 20 --n_stops 1000 > train_log_yelp_1024.txt
# python main_experiments.py --dataset yelp --batch_size 2048 --samp_num 2048 --cuda -1 --is_ratio 0.01 --batch_num 20 --n_stops 1000 > train_log_yelp_2048å.txt