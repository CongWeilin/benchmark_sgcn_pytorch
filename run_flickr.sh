# python main_experiments.py --dataset reddit --batch_size 256 --samp_num 256 --cuda -1 --is_ratio 0.1 --batch_num 20 --n_stops 1000 > train_log_reddit_256.txt
# python main_experiments.py --dataset reddit --batch_size 512 --samp_num 512 --cuda -1 --is_ratio 0.1 --batch_num 20 --n_stops 1000 > train_log_reddit_512.txt
# python main_experiments.py --dataset reddit --batch_size 1024 --samp_num 1024 --cuda -1 --is_ratio 0.1 --batch_num 20 --n_stops 1000 > train_log_reddit_1024.txt
# python main_experiments.py --dataset reddit --batch_size 2048 --samp_num 2048 --cuda -1 --is_ratio 0.1 --batch_num 20 --n_stops 1000 > train_log_reddit_2048.txt

python main_experiments.py --dataset flickr --batch_size 256 --samp_num 256 --cuda $1 --is_ratio 0.5 --batch_num 20 --n_stops 1000 --show_grad_norm 1 > logs/train_log_flickr_256.txt
python main_experiments.py --dataset flickr --batch_size 512 --samp_num 512 --cuda $1 --is_ratio 0.5 --batch_num 20 --n_stops 1000 --show_grad_norm 1 > logs/train_log_flickr_512.txt
python main_experiments.py --dataset flickr --batch_size 1024 --samp_num 1024 --cuda $1 --is_ratio 0.5 --batch_num 20 --n_stops 1000 --show_grad_norm 1 > logs/train_log_flickr_1024.txt
python main_experiments.py --dataset flickr --batch_size 2048 --samp_num 2048 --cuda $1 --is_ratio 0.5 --batch_num 20 --n_stops 1000 --show_grad_norm 1 > logs/train_log_flickr_2048.txt
