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
        for pred, gold in zip(pred_f, gold_f):
            count += 1
            gold_labels = gold.strip()
            pred_labels = pred.strip()
            if len(gold_labels) != len(pred_labels):
                bad_pred_num += 1
                # print(gold_labels)
                # print(pred_labels)
                # print('-'*40)
                continue
            item = (gold_labels + " " + pred_labels).split()
            # print(item)
            # print(len(item))
            if len(item) != 0:
                new_line = item[0] + " " + item[1] + " " + item[3] + "\n"
                # print(new_line)
                res_f.write(new_line)
        res_f.write("\n")
    # print(bad_pred_num)
    # print(count)
    print("Generate {}/{} bad datas when evaluating {}".format(bad_pred_num, count, args.predict_file))   
    os.system("./conlleval.pl < {}".format(args.result_file))
    os.remove(args.result_file)    


if __name__ == '__main__':
    main(parse_args())
