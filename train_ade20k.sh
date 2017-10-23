
image_dir="/SSD_data/ADEChallengeData2016/images/training"
anno_dir="/home/s103062372/dataset/ade20k_combined_anno_5/training"
image_val_dir="/SSD_data/ADEChallengeData2016/images/validation"
anno_val_dir="/home/s103062372/dataset/ade20k_combined_anno_5/validation"

logdir="./weights/ade20k_combine_5/"

combine_list="./labels/ade20k_combine_list_5.mat"


echo "python train_enet.py --image_dir="$image_dir" --anno_dir="$anno_dir" --image_val_dir="$image_val_dir" --anno_val_dir="$anno_val_dir" --logdir="$logdir" --num_classes=5 --batch_size=10 --num_epochs=500 --num_epochs_before_decay=100 --combine_list="$combine_list" --use_gpu=$1"


python train_enet.py --image_dir="$image_dir" --anno_dir="$anno_dir" --image_val_dir="$image_val_dir" --anno_val_dir="$anno_val_dir" --logdir="$logdir" --num_classes=5 --batch_size=10 --num_epochs=300 --num_epochs_before_decay=100 --combine_list="$combine_list" --use_gpu=$1
