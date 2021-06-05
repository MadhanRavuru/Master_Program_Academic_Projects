# Get CIFAR10

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)

cd datasets
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/cifar10_train.zip -OL cifar10_train.zip
tar -xzvf cifar10_train.zip
rm cifar10_train.zip

cd $INITIAL_DIR
