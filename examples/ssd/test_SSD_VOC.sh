./build/tools/caffe train \
--solver models/VGGNet/VOC0712/SSD_300x300_score/solver.prototxt \
--weights ~/VGG_VOC0712_SSD_300x300_annodata_epoch0240.caffemodel --gpu 0
