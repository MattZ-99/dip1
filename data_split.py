import os
import random
import shutil
from tqdm import tqdm


def mkdirFordata(path):
    if os.path.exists(path):
        shutil.rmtree(path, True)

    os.mkdir(path)
    for name in ["train", "valid", "test"]:
        setpath = os.path.join(path, name)
        os.mkdir(setpath)
        os.mkdir(os.path.join(setpath, "data"))
        label_path = os.path.join(setpath, "label")
        os.mkdir(label_path)
        for lab in ["score", "sort"]:
            os.mkdir(os.path.join(label_path, lab))
    return 1


def Process_copyDir(data_path, output_path, list):
    for it in tqdm(list, desc="Process"):
        shutil.copytree(os.path.join(data_path, it), os.path.join(output_path, it))


def Process_copylabelfile(data_path, output_path, list):
    for dir in ["score", "sort"]:
        dp = os.path.join(data_path, dir)
        op = os.path.join(output_path, dir)
        for it in tqdm(list, desc="Process"):
            if dir == "score":
                file = "{}_score.csv".format(it)
            else:
                file = "{}.csv".format(it)

            shutil.copy(os.path.join(dp, file), op)


if __name__ == "__main__":
    data_path = "/home/zmt/work/QA4Camera/dataset"
    train_path = "Training/"
    test_path = "Test/"
    label = "score_and_sort/"
    output_path = "../data/"

    namelist = ["{:0>3d}".format(x + 1) for x in range(100)]
    random.shuffle(namelist)
    split_num = 80
    trainset = namelist[:split_num]
    valset = namelist[split_num:]
    testset = ["{:0>3d}".format(x + 1) for x in range(20)]

    mkdirFordata(output_path)

    Process_copyDir(os.path.join(data_path, train_path), os.path.join(output_path, "train/data"), trainset)
    Process_copylabelfile(os.path.join(os.path.join(data_path, label), "Training"),
                          os.path.join(output_path, "train/label"), trainset)

    Process_copyDir(os.path.join(data_path, train_path), os.path.join(output_path, "valid/data"), valset)
    Process_copylabelfile(os.path.join(os.path.join(data_path, label), "Training"),
                          os.path.join(output_path, "valid/label"), valset)

    Process_copyDir(os.path.join(data_path, test_path), os.path.join(output_path, "test/data"), testset)
