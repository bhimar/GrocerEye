# This script transforms the data set from Roboflow output to Pytorch-YOLOv3 input
# run the script from the frieburg directory

mkdir images
mkdir labels

touch train.txt
touch valid.txt

cp train/_darknet.labels classes.names

cp train/*.txt labels
cp train/*.jpg images

cp train/*.txt labels
cp train/*.jpg images

mkdir data
mkdir data/custom
mkdir data/custom/images
cp train/*.jpg data/custom/images
find ./data/custom/images/*.jpg > train.txt
rm -rf data

mkdir data
mkdir data/custom
mkdir data/custom/images
cp test/*.jpg data/custom/images
find ./data/custom/images/*.jpg > valid.txt
rm -rf data

rm -rf train
rm -rf test