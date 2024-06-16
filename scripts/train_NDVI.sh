cd ..
#nohup python -u train.py \
python -u train.py \
--model 'hablstm' \
--dataset 'NDVI' \
--data_root '/raid1/wpr21/ndvi/NDVI_5_9/' \
--lr 0.001 \
--batch_size 8 \
--total_epoch 400 \
--save_n_epoch 5 \
--epoch_size 650 \
--input_nc 1 \
--output_nc 1 \
--load_size 720 \
--image_width 256 \
--image_height 256 \
--patch_size 8 \
--rnn_size 64 \
--rnn_nlayer 4 \
--filter_size 3 \
--seq_len 6 \
--pre_len 6 \
--eval_len 6 \
--criterion 'MSE' \
--lr_policy 'cosine' \
--niter 5 \
--data_threads 0 \
--optimizer adamw \
--patience 2 \
--test_times 160 \
#> output.log 2>&1 &