import numpy as np


def count_digits(num):
    count = 0
    copy_num = num
    while copy_num > 0:
        copy_num = int(copy_num /10)
        count += 1
    return count

def main():
    filename = "results_log"
    with open(file=filename, mode='r') as f:
        chunk_lines = []
        
        bin_mat_6 = np.zeros((6,12))
        di_mat_6 = np.zeros((6,12))
        trint_mat_6 = np.zeros((6,12))
        grad_mat_6 = np.zeros((6,12))
        rf_mat_6 = np.zeros((6,12))

        bin_mat_3 = np.zeros((3,3))
        di_mat_3 = np.zeros((3,3))
        trint_mat_3 = np.zeros((3,3))
        grad_mat_3 = np.zeros((3,3))
        rf_mat_3 = np.zeros((3,3))
        for line in f:
            chunk_lines.append(line.strip())
            if '-------' in line:
                approach = None
                class_type = None
                pre_start = -1
                pre_end = -1
                post_start = -1
                post_end = -1

                acc_val = -1
                for chunk_line in chunk_lines:
                    if 'pre_start' in chunk_line:
                        pre_start = int(''.join(filter(str.isdigit, chunk_line[:7])))
                        pre_end = int(''.join(filter(str.isdigit, chunk_line[7:7 + 13])))
                        post_start = int(''.join(filter(str.isdigit, chunk_line[7 + 13:7 + 13 + 13])))
                        post_end = int(''.join(filter(str.isdigit, chunk_line[7 + 13 + 13:7 + 13 + 13 + 13])))

                    if 'Classified' in chunk_line:
                        if 'acceptor_data' in chunk_line:
                            class_type = "acceptor_data"
                        elif 'donor_data' in chunk_line:
                            class_type = "donor_data"

                    if "APPROACH" in chunk_line:
                        approach = chunk_line.strip()

                    if "Accuracy" in chunk_line:
                        acc_val = chunk_line.split(',')[0]
                        if 'nan' in acc_val:
                            continue
                        acc_val = int(''.join(filter(str.isdigit, acc_val)))
                        acc_val = acc_val/10**(count_digits(acc_val) - 2)

                if pre_start == -1 and post_start -1 or acc_val == -1:
                    continue

                if post_end - post_start == 49:
                    base = 0 if class_type=="acceptor_data" else 6
                    i = int(pre_start/50)
                    j = int((post_start-300)/50)
                    if approach == "BINARY CLASSIFICATION APPROACH":
                        bin_mat_6[i,base+j] = acc_val
                    if approach == "DiProDB: BINARY CLASSIFICATION APPROACH":
                        di_mat_6[i,base+j] = acc_val
                    if approach == "TRINUCLEOTIDES: BINARY CLASSIFICATION APPROACH":
                        trint_mat_6[i,base+j] = acc_val
                    if approach == "GRADIENT BOOSTING APPROACH":
                        grad_mat_6[i,base+j] = acc_val
                    if approach == "RANDOM FOREST APPROACH":
                        rf_mat_6[i,base+j] = acc_val

                chunk_lines=[]

        '''
        print("bin_mat", np.array2string(bin_mat_6, precision=1, separator='&')
              .replace("[", "").replace("]]", "")
              .replace("]&", "\\\\"))
        print("di_mat", np.array2string(di_mat_6, precision=1, separator='&')
              .replace("[", "").replace("]]", "")
              .replace("]&", "\\\\"))
        print("trint_mat", np.array2string(trint_mat_6, precision=1, separator='&')
              .replace("[", "").replace("]]", "")
              .replace("]&", "\\\\"))
        print("grad_mat", np.array2string(grad_mat_6, precision=1, separator='&')
              .replace("[", "").replace("]]", "")
              .replace("]&", "\\\\"))
        print("rf_mat", np.array2string(rf_mat_6, precision=1, separator='&')
              .replace("[", "").replace("]]", "")
              .replace("]&", "\\\\"))
        '''




if __name__ == '__main__':
    main()