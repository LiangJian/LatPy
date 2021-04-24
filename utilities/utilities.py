import sys
import fileinput

def handle_inputs(inp):
    infile = fileinput.input()
    if len(sys.argv) == 2:
        infile = fileinput.input(sys.argv[1])
    for line_ in infile: 
        tmp_ = line_.split()
        if len(tmp_)<=1:
            continue
        if tmp_[1]=='False' or tmp_[1]=='false':  # hack for the bool type
            tmp_[1] = ''
        if tmp_[0] in inp.keys():
            if isinstance(inp[tmp_[0]], list):
                inp[tmp_[0]] = [type(inp[tmp_[0]][0])(i) for i in tmp_[1:]]
            else:
                inp[tmp_[0]] = type(inp[tmp_[0]])(tmp_[1])
    for i_ in inp.keys():
        print('%-20s'%(i_+': '),inp[i_])
    print('\n')

