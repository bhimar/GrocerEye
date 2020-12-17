# This script transforms the data set from Roboflow output to Pytorch-YOLOv3 input
# run the script from the frieburg directory

mkdir images
mkdir labels

touch train.txt
touch test.txt

cp train/_darknet.labels classes.names

cp train/*.txt labels
cp train/*.jpg images

cp train/*.txt labels
cp train/*.jpg images

mkdir data
mkdir data/custom
cp train/*.jpg data/custom
find ./data/custom/*.jpg > train.txt
rm -rf data

mkdir data
mkdir data/custom
cp test/*.jpg data/custom
find ./data/custom/*.jpg > test.txt
rm -rf data

touch valid.txt