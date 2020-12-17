# This script transforms the data set from Roboflow output to Pytorch-YOLOv3 input
# run the script from the frieburg directory

mkdir images
mkdir labels

touch train.txt
touch valid.txt

cp train/_darknet.labels classes.names

cp train/*.txt labels
cp train/*.jpg images

cp test/*.txt labels
cp test/*.jpg images

mkdir data
mkdir data/images
cp train/*.jpg data/images
find ./data/images/*.jpg > train.txt
rm -rf data

mkdir data
mkdir data/images
cp test/*.jpg data/images
find ./data/images/*.jpg > valid.txt
rm -rf data

# rm -rf train
# rm -rf test