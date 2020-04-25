import sys
import os
import argparse



def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Modify the file format for conlleval.py')
    parser.add_argument('-predict_file', '-p',
                        help='predict file path')
    parser.add_argument('-golden_file', '-g',
                        help='golden file path')
    parser.add_argument('-result_file', '-r',
                        help='Modified file path')
    return parser.parse_args(args)


def main(args):
    with open(args.predict_file, "r", encoding="utf-8") as pred_f, \
         open(args.golden_file, "r", encoding="utf-8") as gold_f,\
         open(args.result_file, "w", encoding="utf-8") as res_f:
        bad_pred_num = 0
        count = 0
        res = []
        for pred, gold in zip(pred_f, gold_f):
            count += 1
            gold_labels = gold.strip()#.split()
            pred_labels = pred.strip()#.split()
            if len(gold_labels) != len(pred_labels):
                bad_pred_num += 1
                # print(gold_labels)
                # print(pred_labels)
                # print('-'*40)
                item = (gold_labels + " " + pred_labels).split()
                res.append(" ".join(sorted(set(item), key=item.index)))
                continue
    print(bad_pred_num)
    print(count)
    print("Generate {} bad datas when evaluating {}".format(bad_pred_num, args.predict_file))
    print(res)
    with open(args.result_file, "w") as outfile:
        outfile.write("\n".join(res))


if __name__ == '__main__':
    main(parse_args())
