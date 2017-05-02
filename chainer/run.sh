CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch alexnet   --batchsize 128     | tee out_alexnet.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch googlenet --batchsize 128     | tee out_googlenet.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgga      --batchsize 64      | tee out_vgga.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch overfeat  --batchsize 128     | tee out_overfeat.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgg16     --batchsize 64      | tee out_vgg16.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgg19     --batchsize 64      | tee out_vgg19.log

CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch alexnet   --batchsize 128 -g -1 | tee out_mkl-dnn_cpu_alexnet.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch googlenet --batchsize 128 -g -1 | tee out_mkl-dnn_cpu_googlenet.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgga      --batchsize 64  -g -1 | tee out_mkl-dnn_cpu_vgga.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch overfeat  --batchsize 128 -g -1 | tee out_mkl-dnn_cpu_overfeat.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgg16     --batchsize 64  -g -1 | tee out_mkl-dnn_cpu_vgg16.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgg19     --batchsize 64  -g -1 | tee out_mkl-dnn_cpu_vgg19.log

pip uninstall -y chainer
pip install chainer --pre

CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch alexnet   --batchsize 128 -g -1 | tee out_cpu_alexnet.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch googlenet --batchsize 128 -g -1 | tee out_cpu_googlenet.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgga      --batchsize 64  -g -1 | tee out_cpu_vgga.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch overfeat  --batchsize 128 -g -1 | tee out_cpu_overfeat.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgg16     --batchsize 64  -g -1 | tee out_cpu_vgg16.log
CHAINER_TYPE_CHECK=0 python -OO train_imagenet.py --arch vgg19     --batchsize 64  -g -1 | tee out_cpu_vgg19.log
