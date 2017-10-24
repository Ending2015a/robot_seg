
image_dir="/home/joehsiao/Dataset/ADEChallengeData2016/images/training"
anno_dir="/home/joehsiao/Dataset/ade20k_combined_anno_5/training"
image_val_dir="/home/joehsiao/Dataset/ADEChallengeData2016/images/validation"
anno_val_dir="/home/joehsiao/Dataset/ade20k_combined_anno_5/validation"

logdir="./weights/ade20k_combine_5_poly/"

combine_list="./labels/ade20k_combine_list_5.mat"
batch_size=8

#scp -r s103062372@140.114.75.144:/home/s103062372/dataset/ade20k_combined_anno_5 /home/joehsiao/Dataset/


echo "python train_enet.py --image_dir="$image_dir" --anno_dir="$anno_dir" --image_val_dir="$image_val_dir" --anno_val_dir="$anno_val_dir" --logdir="$logdir" --num_classes=5 --batch_size=$batch_size --num_epochs=500 --num_epochs_before_decay=100 --combine_list="$combine_list" --use_gpu=$1"


python train_enet.py --image_dir="$image_dir" --anno_dir="$anno_dir" --image_val_dir="$image_val_dir" --anno_val_dir="$anno_val_dir" --logdir="$logdir" --num_classes=5 --batch_size="$batch_size" --num_epochs=300 --num_epochs_before_decay=100 --combine_list="$combine_list" --use_gpu=$1
