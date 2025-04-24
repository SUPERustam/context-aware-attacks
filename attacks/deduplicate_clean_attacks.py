import re
import argparse
from pathlib import Path
from typing import List
import sys
import os


def deduplicate_attacks(txt: List[str], verbose=False) -> List[str]:
    """Deduplicate the attack plans with same image id and ap from wb text file
    Args:
        txt (list of strs): the txt file
            for example:
            im000006_idx2_f8_t11_ap0_wb[1]_bb[0]
            im000006_idx2_f8_t11_ap1_wb[1]_bb[0]
            im000006_idx2_f8_t11_ap2_wb[1]_bb[1]
            im000006_idx2_f8_t11_ap3_wb[1]_bb[1]
            ...
    Returns:
        unique_items2write (list): the unique attack plans to write into the deduplicated txt file
    """

    unique_items = (
        dict()
    )  # Using dict as ordered set (Python 3.7+ maintains insertion order)
    unique_items2write = list()

    # Deduplicate the attack plans with same image id and ap
    for line in txt:
        if line.startswith("im"):
            line = line.strip()
            im_id = re.findall(r"im(.+?)\_", line)[0]
            ap = int(re.findall(r"ap(.+?)\_", line)[0])
            item = (im_id, ap)

            if item not in unique_items:
                unique_items[item] = True
                unique_items2write.append(line)
            elif verbose:
                print(f"Removed duplicate attack plan: {line}")

    return unique_items2write


def one_dataset_txt(txt: List[str], dataset: str, verbose=False) -> List[str]:
    """Change the txt file name to the dataset name
    Args:
        txt (list of strs): the txt file
        dataset (str): the dataset name
    Returns:
        dataset_items2write (list): the unique attack plans to write into the deduplicated txt file
    """
    dataset_items2write = list()

    for line in txt:
        im_id = re.findall(r"im(.+?)\_", line.strip())[0]

        if dataset == "voc":
            if len(im_id) == 6:
                dataset_items2write.append(line)
            elif verbose:
                print(f"Removed COCO attack plan: {line}")

        elif dataset == "coco":
            if len(im_id) == 12:
                dataset_items2write.append(line)
            elif verbose:
                print(f"Removed VOC attack plan: {line}")

    return dataset_items2write


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate attack plans and remove wrong attack plans."
    )

    parser.add_argument(
        "--eps", nargs="?", default=30, help="perturbation level: 10,20,30,40,50"
    )
    parser.add_argument(
        "--root", nargs="?", default="result", help="the folder name of result"
    )
    parser.add_argument("-bb", action="store_true", help="use bb txt file")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="voc",
        help="model dataset 'voc' or 'coco'. This will change txt file name",
    )

    args = parser.parse_args()
    eps = int(args.eps)
    result_folder = args.root
    use_bb_file = args.bb
    dataset = args.dataset

    # parse wb train txt and bb test txt
    exp = f"run_sequential_attack_eps{eps}"
    result_root = Path(result_folder) / exp
    if use_bb_file:
        file = open(result_root / f"{exp}_bb.txt", "r")
    else:
        file = open(result_root / f"{exp}.txt", "r")
    txt = file.readlines()
    print(file.name)


    # read files and count ap steps for each image
    deduplicated_txt = deduplicate_attacks(txt, verbose=True)

    attacks_plan_txt = one_dataset_txt(deduplicated_txt, dataset, verbose=True)
    
    print(f"Total number of attacks plan: {len(attacks_plan_txt)}, dataset: {dataset}, eps: {eps}, use_bb: {use_bb_file}, root: {result_root}")
    print(f"Example attacks plan: {attacks_plan_txt[0]}")
    if input("Do you want to save the attacks plan txt and remove the wrong attacks plan in 'pert' folder? (Y/n) ") != 'Y':
        print("Exiting...")
        sys.exit()

    with open(file.name, "w") as f:
        for line in attacks_plan_txt:
            f.write(line + "\n")

    if not use_bb_file:
        # Clean excess .npy in "pert" folder
        pert_root = result_root / "pert"
        attacks_plan_txt = set(attacks_plan_txt)
        for file in pert_root.glob("*.npy"):
            filename = os.path.basename(file)[:-4]
            if filename not in attacks_plan_txt:
                os.remove(file)
                print(f"Removed 'pert' file: {file}")


if __name__ == "__main__":
    main()
