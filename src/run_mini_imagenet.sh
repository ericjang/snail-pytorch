python train.py --exp mini_imagenet_5way_1shot_nonoverlap --dataset mini_imagenet --cuda --task_shuffling non_overlapping --batch_size 4 >> mini_imagenet_5way_1shot_nonoverlap.txt`
python train.py --exp mini_imagenet_5way_1shot_intratask --dataset mini_imagenet --cuda --task_shuffling intratask --batch_size 4 >> mini_imagenet_5way_1shot_intratask.txt`
python train.py --exp mini_imagenet_5way_1shot_intertask --dataset mini_imagenet --cuda --task_shuffling intertask --batch_size 4 >> mini_imagenet_5way_1shot_intertask.txt`
