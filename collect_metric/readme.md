# Workloads Profiling

This part contains code for profiling metrics of multiple workloads.

## Directory
Note that `./result/` will be created when `main_co.py` or `main_single.py` is launched.

## Basic Usage
Run `main_co.py` will generate the colocated jobs' metrics under `./result/colocate`. Run `main_single.py` will generate single jobs' metrics under `./result/`. Some specific settings can be set in each workload's profiling file, e.g.`profile_cifar.py`. The output will be like this:
```
imagenet + imagenet
co-locate:
==> Training mobilenet_v3_small model with 32 batchsize, 0 mp..
==> Training mobilenet_v3_small model with 32 batchsize, 0 mp..
co-locate:
==> Training mobilenet_v3_small model with 32 batchsize, 0 mp..
==> Training mobilenet_v3_small model with 32 batchsize, 1 mp..
co-locate:
==> Training mobilenet_v3_small model with 32 batchsize, 1 mp..
==> Training mobilenet_v3_small model with 32 batchsize, 1 mp..
imagenet + cifar10
co-locate:
Files already downloaded and verified
==> Training ResNet18 model with 32 batchsize, 0 mp..
==> Training mobilenet_v3_small model with 32 batchsize, 0 mp..
...
```

## Datasets
The data path storing all datasets is specified in `./workloads/settings.py` as `data_dir`. You can also specify the total runtime of some workloads by changing `total_runtime`.


- CIFAR-10: The cifar10 dataset will be downloaded automatically(if not exist) when `./workloads/cifar/profile_cifar.py` is run.

- ImageNet: The dataset is generated automatically in `./workloads/imagenet/profile_imagenet.py`.

- LSUN: The dataset is generated automatically in `./workloads/dcgan/profile_dcgan.py`. You can change the custom image size of generated data via `--imageSize`. The default value is 64.

- ShapeNet: Use the following command to download dataset under directory `data_dir/shapenetcore/`:

    ```bash
    wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
    unzip shapenetcore_partanno_segmentation_benchmark_v0.zipcollect_metric/workloadspointnet.pytorch.

- SQuAD: The data can be downloaded with the following link and should be saved under `data_dir/SQUAD_DIR/` directory.

    [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)

- Wikitext2: The dataset can be downloaded from 

    [wikitext-2](https://github.com/pytorch/examples/tree/main/word_language_model/data/wikitext-2)

    File `test.txt`, `train.txt` and `valid.txt` should be saved in `data_dir/wikitext-2/` directory.

- Multi30k: First download the Moses tokenizer(http://www.statmt.org/moses/) for data preparation:
    ```bash
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
    sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
    ```
    These files should be downloaded in `./workloads/translation/`.

    Then download data in `data_dir/multi30k/`:
    ```bash
    mkdir -p data/multi30k
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C data/multi30k && rm mmt16_task1_test.tar.gz
    ```
    Preprocess the data:
    ```bash
    for l in en de; do for f in ~/data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
    for l in en de; do for f in ~/data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
    python preprocess.py -train_src ~/data/multi30k/train.en.atok -train_tgt ~/data/multi30k/train.de.atok -valid_src ~/data/multi30k/val.en.atok -valid_tgt ~/data/multi30k/val.de.atok -save_data ~/data/multi30k.atok.low.pt
    ```
    Referenced from: https://github.com/Eathoublu/attention-is-all-you-need-pytorch.

- MovieLens: Use the following command to download the dataset in `data_dir/ml-1m/`:
    ```bash
    wget https://github.com/hexiangnan/neural_collaborative_filtering/raw/master/Data/ml-1m.test.negative
    wget https://github.com/hexiangnan/neural_collaborative_filtering/raw/master/Data/ml-1m.test.rating
    wget https://github.com/hexiangnan/neural_collaborative_filtering/raw/master/Data/ml-1m.train.rating
    ```