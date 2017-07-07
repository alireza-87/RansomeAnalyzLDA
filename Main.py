# Alireza.Karimi.67@gmail.com
from __future__ import division
import urllib
import urllib2
from os import walk
import os
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from DataBaseHandler import DatabaseHandler
from androdd import dump_all_method
import json
from collections import Counter
import unicodedata
import numpy as np
from shutil import copyfile
import xlsxwriter


class Main():
    def __init__(self):
        self.db = DatabaseHandler()

    def generat(self):
        f = []
        for (dirpath, dirnames, filenames) in walk(self.constant.getInputDir()):
            f.extend(filenames)
        return f


def get_all_files_in_directory(directory):
    f = []
    for (dirpath, dirnames, filenames) in walk(directory):
        f.extend(filenames)
    return f


def get_all_files_withpath_in_directory(directory):
    f = []
    for (dirpath, dirnames, filenames) in walk(directory):
        if filenames:
            for item in filenames:
                fillee = dirpath + '/' + item
                f.append(fillee)
    return f


def clean_up_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def func_weight_p_op1_op2(sample_mal, sample_mal_1, vector):
    cal_class = []
    for iii in range(0, len(sample_mal)):
        sample_vector = {}
        dict_y = Counter(sample_mal_1[iii])
        dict_x = Counter(sample_mal[iii])
        for op_seq in vector:
            print str(op_seq)
            spliter = op_seq.strip().split()
            x = 0
            y = 0
            if spliter[0] in dict_y:
                y = dict_y[spliter[0]]
            if op_seq in dict_x:
                x = dict_x[op_seq]
            if y != 0:
                p = x / y
            else:
                p = 0
            sample_vector[op_seq] = p
        cal_class.append(sample_vector)
    return cal_class


def write_arff(dataset, class1, class2):
    final_op_set = []
    opcode_bank = {}
    index_helper_x = 0
    seen = set()
    for item in class1:
        for key, value in item.iteritems():
            splitter = key.strip().split()
            if splitter[0] not in seen:
                final_op_set.append(splitter[0])
                opcode_bank[splitter[0]] = index_helper_x
                index_helper_x = index_helper_x + 1
                seen.add(splitter[0])
            if splitter[1] not in seen:
                final_op_set.append(splitter[1])
                opcode_bank[splitter[1]] = index_helper_x
                index_helper_x = index_helper_x + 1
                seen.add(splitter[1])
    for item in class2:
        for key, value in item.iteritems():
            splitter = key.strip().split()
            if splitter[0] not in seen:
                final_op_set.append(splitter[0])
                opcode_bank[splitter[0]] = index_helper_x
                index_helper_x = index_helper_x + 1
                seen.add(splitter[0])
            if splitter[1] not in seen:
                final_op_set.append(splitter[1])
                opcode_bank[splitter[1]] = index_helper_x
                index_helper_x = index_helper_x + 1
                seen.add(splitter[1])
    data_fp = open(dataset, "w")
    data_fp.write('''@RELATION OpcodeSequence
           ''')
    data_fp.write("\n")
    for opc_i in final_op_set:
        for opc_j in final_op_set:
            name = str(opc_i) + str(opc_j)
            data_fp.write("@ATTRIBUTE %s NUMERIC \n" % name)
    data_fp.write("@ATTRIBUTE Class1 {mal,bin} \n")
    data_fp.write("\n")
    data_fp.write("@DATA")
    data_fp.write("\n")

    for item in class1:
        image = np.array([[0.0 for j in range(len(final_op_set))] for i in range(len(final_op_set))])
        for opc_i in final_op_set:
            for opc_j in final_op_set:
                x = opcode_bank[opc_i]
                y = opcode_bank[opc_j]
                key = str(str(opc_i) + " " + str(opc_j))
                print key
                if key in item:
                    image[x][y] = item[str(opc_i) + " " + str(opc_j)]
                    data_fp.write(str(item[str(opc_i) + " " + str(opc_j)]) + ",")
                else:
                    data_fp.write("0" + ",")
        data_fp.write("mal")
        data_fp.write("\n")

    for item in class2:
        image = np.array([[0.0 for j in range(len(final_op_set))] for i in range(len(final_op_set))])
        for opc_i in final_op_set:
            for opc_j in final_op_set:
                x = opcode_bank[opc_i]
                y = opcode_bank[opc_j]
                key = str(str(opc_i) + " " + str(opc_j))
                print key
                if key in item:
                    image[x][y] = item[str(opc_i) + " " + str(opc_j)]
                    data_fp.write(str(item[str(opc_i) + " " + str(opc_j)]) + ",")
                else:
                    data_fp.write("0" + ",")
        data_fp.write("bin")
        data_fp.write("\n")


def opcode_sequence_generator4(repo, dumpMethodDir):
    db = DatabaseHandler()
    samples = db.select_sample_all()
    vector = []
    sample_mal = []
    sample_bin = []
    sample_mal_1 = []
    sample_bin_1 = []
    sample_bin_name = []
    sample_mal_name = []
    seen = set()
    for item in samples:
        try:
            # Generate Opcode Seq for every sample
            dump_all_method(repo + item[1], dumpMethodDir)
            opcode_sequence = check_opcode(dumpMethodDir)
            opcode_list1 = check_opcode2(dumpMethodDir)
            # Add opcode seq to class belong
            if item[1].startswith('bin_') and item[1].endswith(".apk"):
                sample_bin.append(opcode_sequence)
                sample_bin_1.append(opcode_list1)
                sample_bin_name.append(item[1])
            elif item[1].endswith(".apk"):
                sample_mal.append(opcode_sequence)
                sample_mal_1.append(opcode_list1)
                sample_mal_name.append(item[1])
            # Generate a Sequence banck
            for item in opcode_sequence:
                if item not in seen:
                    vector.append(item)
                    seen.add(item)
        except Exception as e:
            print e

    mal_class = []
    bin_class = []
    mal_class = func_weight_p_op1_op2(sample_mal, sample_mal_1, vector)
    bin_class = func_weight_p_op1_op2(sample_bin, sample_bin_1, vector)
    write_arff(repo + 'result.arff', mal_class, bin_class)

    output_filename = repo + 'resultLDA.txt'
    simple_result = repo + 'Expenses01.xlsx'
    fp_lda = open(output_filename, "w")
    workbook = xlsxwriter.Workbook(simple_result)
    worksheet = workbook.add_worksheet()

    n_fold = 10
    top_edge = []
    for i in range(2, 250):
        top_edge.append(i)
    row_index = 0
    for top in top_edge:
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        total_acc = 0
        total_tpr = 0
        total_fpr = 0
        total_final_set = 0
        name = "************** TOP" + str(top) + " **************"
        fp_lda.write(name)
        fp_lda.write('\n')
        test_count_mal = int(len(mal_class) / n_fold)
        test_count_bin = int(len(bin_class) / n_fold)
        p_bin = 0
        p_mal = 0
        for fold in range(1, n_fold + 1):
            train_mal_class = []
            train_bin_class = []
            test_mal_class = []
            test_bin_class = []
            test_mal_name = []
            test_bin_name = []
            for i in range(0, len(bin_class)):
                if i >= p_bin * test_count_bin and i < p_bin * test_count_bin + test_count_bin:
                    test_bin_class.append(bin_class[i])
                    test_bin_name.append(sample_bin_name[i])
                else:
                    train_bin_class.append(bin_class[i])
            p_bin = p_bin + 1

            for i in range(0, len(mal_class)):
                if i >= p_mal * test_count_mal and i < p_mal * test_count_mal + test_count_mal:
                    test_mal_class.append(mal_class[i])
                    test_mal_name.append(sample_mal_name[i])
                else:
                    train_mal_class.append(mal_class[i])
            p_mal = p_mal + 1

            # calculate MIN mal class for every feature
            MIN_total = {}
            total_len = len(train_mal_class) + len(train_bin_class)
            print "start Calculate Mean Malware Class"
            MIN_mal = {}
            for feature in vector:
                sum_feature = 0
                for item in train_mal_class:
                    if feature in item:
                        sum_feature = item[feature] + sum_feature
                MIN_mal[feature] = sum_feature / len(train_mal_class)
                MIN_total[feature] = sum_feature
            print "start Calculate Mean Bin Class"

            MIN_bin = {}
            for feature in vector:
                sum_feature = 0
                for item in train_bin_class:
                    if feature in item:
                        sum_feature = item[feature] + sum_feature
                MIN_bin[feature] = sum_feature / len(train_bin_class)
                MIN_total[feature] = (MIN_total[feature] + sum_feature) / total_len
            print "start Calculate SW"

            # Calculate SW
            SW = {}
            for feature in vector:
                sum_feature = 0
                for item in train_mal_class:
                    if feature in item and feature in MIN_mal:
                        X = item[feature] - MIN_mal[feature]
                    elif feature in item:
                        X = item[feature]
                    elif feature in MIN_mal:
                        X = MIN_mal[feature]
                    else:
                        X = 0
                    Y = X * X
                    sum_feature = sum_feature + Y

                for item in train_bin_class:
                    if feature in item and feature in MIN_bin:
                        X = item[feature] - MIN_bin[feature]
                    elif feature in item:
                        X = item[feature]
                    elif feature in MIN_bin:
                        X = MIN_bin[feature]
                    else:
                        X = 0
                    Y = X * X
                    sum_feature = sum_feature + Y

                SW[feature] = sum_feature

            # Calculate SB
            print "start Calculate Mean SB"
            malware_persentage = len(train_mal_class) * 100 / total_len
            binware_persentage = len(train_mal_class) * 100 / total_len
            SB = {}
            for features in vector:
                if feature in MIN_mal and feature in MIN_bin:
                    total_mean = MIN_total[features]
                    SB[features] = (malware_persentage * (MIN_mal[features] - total_mean) * (MIN_mal[features] - total_mean)) + (
                    binware_persentage * (MIN_bin[features] - total_mean) * (MIN_bin[features] - total_mean))
                elif feature in MIN_bin:
                    total_mean = MIN_total[features]
                    SB[features] = (malware_persentage * (0 - total_mean) * (0 - total_mean)) + (
                    binware_persentage * (MIN_bin[features] - total_mean) * (MIN_bin[features] - total_mean))
                elif feature in MIN_mal:
                    total_mean = MIN_total[features]
                    SB[features] = (malware_persentage * (MIN_mal[features] - total_mean) * (MIN_mal[features] - total_mean)) + (
                        binware_persentage * (0 - total_mean) * (0 - total_mean))
                else:
                    total_mean = 0
                    SB[features] = (malware_persentage * (0 - total_mean) * (0 - total_mean)) + (binware_persentage * (0 - total_mean) * (0 - total_mean))

            # Calculate ST
            print "start Calculate ST"
            ST = {}
            for item in vector:
                if SW[item] != 0:
                    ST[item] = (SB[item]) / SW[item]
                else:
                    ST[item] = 0
            select_top = sorted(ST.iteritems(), key=lambda x: -x[1], reverse=False)[: top]
            final_op_set = []
            opcode_bank = {}
            index_helper_x = 0
            seen = set()
            for key, value in select_top:
                splitter = key.strip().split()
                if splitter[0] not in seen:
                    final_op_set.append(splitter[0])
                    opcode_bank[splitter[0]] = index_helper_x
                    index_helper_x = index_helper_x + 1
                    seen.add(splitter[0])
                if splitter[1] not in seen:
                    final_op_set.append(splitter[1])
                    opcode_bank[splitter[1]] = index_helper_x
                    index_helper_x = index_helper_x + 1
                    seen.add(splitter[1])
            len_train = len(train_bin_class) + len(train_mal_class)
            test_set_mal = np.zeros((len(test_mal_class), len(final_op_set) * len(final_op_set)))
            test_set_bin = np.zeros((len(test_bin_class), len(final_op_set) * len(final_op_set)))
            train_set = np.zeros((len_train, len(final_op_set) * len(final_op_set)))
            train_lable = []
            index_train = 0

            for item in train_mal_class:
                image = np.array([[1.0 for j in range(len(final_op_set))] for i in range(len(final_op_set))])
                for opc_i in final_op_set:
                    for opc_j in final_op_set:
                        x = opcode_bank[opc_i]
                        y = opcode_bank[opc_j]
                        key = str(str(opc_i) + " " + str(opc_j))
                        if key in item:
                            image[x][y] = item[str(opc_i) + " " + str(opc_j)]
                        else:
                            image[x][y] = 0
                train_set[index_train] = image.flatten()
                train_lable.append(1)
                index_train = index_train + 1

            for item in train_bin_class:
                image = np.array([[1.0 for j in range(len(final_op_set))] for i in range(len(final_op_set))])
                for opc_i in final_op_set:
                    for opc_j in final_op_set:
                        x = opcode_bank[opc_i]
                        y = opcode_bank[opc_j]
                        key = str(str(opc_i) + " " + str(opc_j))
                        if key in item:
                            image[x][y] = item[str(opc_i) + " " + str(opc_j)]
                        else:
                            image[x][y] = 0
                train_set[index_train] = image.flatten()
                train_lable.append(0)
                index_train = index_train + 1

            index_test = 0
            for item in test_mal_class:
                image = np.array([[1.0 for j in range(len(final_op_set))] for i in range(len(final_op_set))])

                for opc_i in final_op_set:
                    for opc_j in final_op_set:
                        x = opcode_bank[opc_i]
                        y = opcode_bank[opc_j]
                        key = str(str(opc_i) + " " + str(opc_j))
                        if key in item:
                            image[x][y] = item[str(opc_i) + " " + str(opc_j)]
                        else:
                            image[x][y] = 0
                test_set_mal[index_test] = image.flatten()
                index_test = index_test + 1

            index_test = 0
            for item in test_bin_class:
                image = np.array([[1.0 for j in range(len(final_op_set))] for i in range(len(final_op_set))])
                for opc_i in final_op_set:
                    for opc_j in final_op_set:
                        x = opcode_bank[opc_i]
                        y = opcode_bank[opc_j]
                        key = str(str(opc_i) + " " + str(opc_j))
                        if key in item:
                            image[x][y] = item[str(opc_i) + " " + str(opc_j)]
                        else:
                            image[x][y] = 0
                test_set_bin[index_test] = image.flatten()
                index_test = index_test + 1

            clf = LinearDiscriminantAnalysis()

            clf.fit(train_set, train_lable)
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            fn_name = []
            fp_name = []
            index_name = 0
            for item in test_set_mal:
                result = clf.predict(item.reshape(1, -1))
                if result == 1:
                    tp = tp + 1
                else:
                    fn = fn + 1
                    fn_name.append(test_mal_name[index_name])
                index_name = index_name + 1
            index_name = 0
            for item in test_set_bin:
                result = clf.predict(item.reshape(1, -1))
                if result == 0:
                    tn = tn + 1
                else:
                    fp = fp + 1
                    fp_name.append(test_bin_name[index_name])
                index_name = index_name + 1
            acc = (tp + tn) / (tp + tn + fp + fn)
            tpr = (tp) / (tp + fn)
            fpr = (fp) / (fp + tn)
            fp_lda.write('\n')
            fp_lda.write('TP : ' + str(tp))
            fp_lda.write('\n')
            fp_lda.write('TN : ' + str(tn))
            fp_lda.write('\n')
            fp_lda.write('FP : ' + str(fp))
            fp_lda.write('\n')
            fp_lda.write('FN : ' + str(fn))
            fp_lda.write('\n')
            fp_lda.write('ACC : ' + str(acc))
            fp_lda.write('\n')
            fp_lda.write('LEN : ' + str(len(final_op_set)))
            fp_lda.write('\n')
            for item in fp_name:
                fp_lda.write('fp_name : ' + str(item))
                fp_lda.write('\n')
            for item in fn_name:
                fp_lda.write('fn_name : ' + str(item))
                fp_lda.write('\n')
            total_tp = total_tp + tp
            total_tn = total_tn + tn
            total_fp = total_fp + fp
            total_fn = total_fn + fn
            total_acc = total_acc + acc
            total_tpr = total_tpr + tpr
            total_fpr = total_fpr + fpr
            total_final_set = len(final_op_set) + total_final_set
        col_index = 0
        worksheet.write(row_index, col_index, total_tp / fold)
        col_index = col_index + 1
        worksheet.write(row_index, col_index, total_fp / fold)
        col_index = col_index + 1
        worksheet.write(row_index, col_index, total_tn / fold)
        col_index = col_index + 1
        worksheet.write(row_index, col_index, total_fn / fold)
        col_index = col_index + 1
        worksheet.write(row_index, col_index, total_tpr / fold)
        col_index = col_index + 1
        worksheet.write(row_index, col_index, total_fpr / fold)
        col_index = col_index + 1
        worksheet.write(row_index, col_index, total_acc / fold)
        col_index = col_index + 1
        worksheet.write(row_index, col_index, top)
        col_index = col_index + 1
        worksheet.write(row_index, col_index, total_final_set / fold)
        col_index = col_index + 1
        row_index = row_index + 1


def scan_with_virus_total(path, db=None):
    files = get_all_files_in_directory(path)
    for afile in files:
        try:
            if '.DS_Store' not in afile:
                make_virus_total_request(afile.split('.')[0])
        except Exception as e:
            print e


def make_virus_total_request(hash, db=None):
    try:
        params = {'apikey': 'YOUR_LEY', 'resource': hash}
        data = urllib.urlencode(params)
        result = urllib2.urlopen('https://www.virustotal.com/vtapi/v2/file/report', data)
        jdata = json.loads(result.read())
        return parse(jdata, hash)
    except Exception as e:
        print e
        return 'Forbidden'


def parse(it, md5, verbose=True, jsondump=True):
    if it['response_code'] == 0:
        print md5 + " -- Not Found in VT"
        return 0
    else:
        return it['positives']


def check_opcode(path_to_dir):
    full_address = (path_to_dir).strip('\n')

    list_files = get_all_files_withpath_in_directory(full_address)
    list_general = []
    for index in range(0, len(list_files)):
        temp_file = list_files[index]
        try:
            if temp_file.endswith('.ag'):
                list_opcode = []
                file_open = open(temp_file)
                print temp_file
                for m in file_open:
                    b = m.strip()
                    if b.startswith('1') or b.startswith('2') or b.startswith('3') or b.startswith('4') or b.startswith('5') or b.startswith('6') or b.startswith('7') or b.startswith('8') or b.startswith('9') or b.startswith('0'):
                        word = []
                        word = m.strip().split()
                        if len(word) >= 2:
                            list_opcode.append(word[2])
                            list_general.append(word[2])
                print list_opcode
        except Exception as e:
            print e

    list_opcode_sequence = []
    for item in range(0, len(list_general) - 1):
        list_opcode_sequence.append(list_general[item] + ' ' + list_general[item + 1])
    return list_opcode_sequence


def check_opcode2(path_to_dir):
    full_address = (path_to_dir).strip('\n')
    list_files = get_all_files_withpath_in_directory(full_address)
    list_general = []
    for index in range(0, len(list_files)):
        temp_file = list_files[index]
        try:
            if temp_file.endswith('.ag'):
                list_opcode = []
                file_open = open(temp_file)
                print temp_file
                for m in file_open:
                    b = m.strip()
                    if b.startswith('1') or b.startswith('2') or b.startswith('3') or b.startswith('4') or b.startswith('5') or b.startswith('6') or b.startswith('7') or b.startswith('8') or b.startswith('9') or b.startswith('0'):
                        word = []
                        word = m.strip().split()
                        if len(word) >= 2:
                            list_opcode.append(word[2])
                            list_general.append(word[2])
                print list_opcode
        except Exception as e:
            print e
    return list_general


def fill_samples_table(repo):
    db = DatabaseHandler()
    db.recreats_table_samples()
    files = get_all_files_in_directory(repo)
    for afile in files:
        try:
            if '.DS_Store' not in afile:
                db.insert_a_sample(afile, '')
        except Exception as e:
            print e


def update_samples_label(repo):
    db = DatabaseHandler()
    samples = db.select_sample_all()
    for item in samples:
        isSend = False
        while not isSend:
            lable = make_virus_total_request(item[1].split('.')[0])
            if 'Forbidden' != lable:
                shash = unicodedata.normalize('NFKD', item[1]).encode('ascii', 'ignore')
                rowcount = db.update_sample_lable(shash, lable)
                print item[0], ' -> ', item[1], " : ", lable, ' RowCount : ', str(rowcount)
                if (int(lable) == 0):
                    copyfile(repo + item[1], repo + "0/" + item[1])
                elif (int(lable) == 1):
                    copyfile(repo + item[1], repo + "1/" + item[1])
                elif int(lable) > 1 and int(lable) <= 5:
                    copyfile(repo + item[1], repo + "5/" + item[1])
                elif int(lable) > 5 and int(lable) <= 10:
                    copyfile(repo + item[1], repo + "10/" + item[1])
                else:
                    copyfile(repo + item[1], repo + "more/" + item[1])
                isSend = True
            else:
                print item[0], ' -> ', item[1], ' : Forbidden'
                time.sleep(120)


def run_whole_process(repo, dump_Method_dir):
    fill_samples_table(repo)
    opcode_sequence_generator4(repo, dump_Method_dir)


def menu_select():
    db = DatabaseHandler()
    repo = '/Users/midnightgeek/Repo/l14/'
    dump_Method_dir = '/Users/midnightgeek/Tools/test2'
    print '********* DataSet Generator *********'
    print 'Enter 1 For Run All Progress'
    print 'Enter 2 For Fill Samples Table'
    print 'Enter 3 For Lable Sample With VT Api'
    print 'Enter 4 For Clear Samples Table'
    menu = raw_input("Enter Number : ")
    if menu == '1':
        run_whole_process(repo, dump_Method_dir)
    elif menu == '2':
        fill_samples_table(repo, dump_Method_dir)
    elif menu == '3':
        update_samples_label(repo)
    elif menu == '4':
        db.clear_table_samples()
    else:
        print 'Wrong Number'


if __name__ == '__main__':
    menu_select()
