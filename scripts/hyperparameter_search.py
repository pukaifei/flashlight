import os
import math
import random
import numpy
import uuid
import copy


batchsize = 8
archdir = "/mnt/vol/gfsai-east/ai-group/users/vitaliy888/arch/"
rundir = "/mnt/vol/gfsai-flash-east/ai-group/users/vitaliy888/chronos/prod"

# OMP_NUM_THREADS=1

def runCommand(cmd):
    filename = "/home/vitaliy888/temp/autoscripts/runexperiment{}.sh".format(uuid.uuid1())
    f = open(filename, "w+")
    f.write(cmd)
    f.close()
    os.system("/usr/local/chronos/scripts/crun --hostgroup fblearner_ash_bigbasin_fair --retries 20 -G " + str(batchsize) + " -M 50 " + filename)
    #--hostgroup fblearner_ash_bigbasin_fair
    #print(cmd)

def generateConvLayer(inputplanes, outputplanes, dropout, kw):
    outputplanes = outputplanes * 2
    s = "WN 3 C " + str(inputplanes) + " " + str(outputplanes) + " " + str(kw) + " 1 -1 \n"
    s = s + "GLU 2 " + "\n"
    if dropout > .0:
        s = s + "DO {}\n".format(dropout)
    return s



# layer kernels follow arithmetic progression, i.e. kw0, kw0 + dkw, kw0 + 2dkw etc, where dkw is computed from totalkw
# featratio is number of output features in the last non-output layer vs number of output features in the first layer
# number of output features at each layer follow geometric progression, same for dropout
def generateConvArch(totalkw, kw0, totalparams, ninput, nlabel, featratio, depth, dropout0, dropoutN):
    tag = "totalkw_{}_kw0_{}_params_{}x{}_depth{}_drop{}x{}".format(totalkw, kw0, totalparams, featratio, depth, dropout0, dropoutN)
    nconv = depth - 2 # number of convolution layers assuming that last two layers are linear
    #compute difference in kw, dkw assuming that layer kernels follow and arhitmetic progression
    dkw = 2.0 * (totalkw + nconv - nconv * kw0 - 1) / (nconv * nconv - nconv)
    # kw of i-th layer
    def f_kw_i(ilayer):
        assert(ilayer <= depth)
        # the last two layers are linear
        if ilayer >= depth - 1:
            return 1
        else:
            return int(kw0 + dkw * (ilayer - 1))

    #compute ratio in outputs geometric progression
    rfeat = featratio ** (1.0 / (depth - 2))
    def f_input_i(input0, ilayer):
        assert(ilayer <= depth + 1)
        if ilayer == 1:
            return ninput
        elif ilayer == depth + 1:
            return nlabel
        elif input0 == 0.0:
            return rfeat ** (ilayer - 2)
        else:
            return int(input0 * (rfeat ** (ilayer - 2)))

    def f_param_i(input0, ilayer):
        return f_kw_i(ilayer) * f_input_i(input0, ilayer) * f_input_i(input0, ilayer + 1) * (2 if ilayer < depth else 1)
    # since we have totalparams constraint, we need to solve quadratic equation to get number of output features at each layer
    b = f_param_i(0.0, 1) + f_param_i(0.0, depth)
    a = 0
    for i in range(2, depth):
        a += f_param_i(0.0, i)
    feat0 = ((b * b + 4 * a * totalparams) ** 0.5 - b) / (2.0 * a)

    # dropout is also geometric progression
    rdrop = (dropoutN / dropout0) ** (1.0 / (depth - 1))
    #generate arch file with defined progressions
    #print("Coefficients: dkw {}, rfeat {}, feat0 {}, rdprop {}, featratio {}, feat0 {} \n".format(dkw, rfeat, feat0, rdrop, f_input_i(feat0, depth)/f_input_i(feat0, 2), f_input_i(feat0, 2)))
    arch = "V -1 1 NFEAT 0\n"
    tp = 0
    for i in range(1, depth):
        tp += f_param_i(feat0, i)
        drop_i = dropout0 * (rdrop ** (i - 1))
        #print("Layer {}: kw {}, param {}, manual param {}, output {} \n".format(i, f_kw_i(i), f_param_i(feat0, i), f_kw_i(i) * feat0 * feat0 * 2 * (rfeat ** (2*i - 3)), f_input_i(feat0, i + 1)))
        arch += generateConvLayer(f_input_i(feat0, i) if i > 1 else "NFEAT", f_input_i(feat0, i + 1), drop_i, f_kw_i(i))
    # generate last layer
    tp += f_param_i(feat0, depth)
    #print("Layer {}: kw {}, param {}, output {} \n".format(depth, f_kw_i(depth), f_param_i(feat0, depth), f_input_i(feat0, depth + 1)))
    arch += "WN 3 C " + str(f_input_i(feat0, depth)) + " NLABEL 1 1 -1 \n"
    arch += "RO 2 0 3 1\n"
    #print("Total number of parameters: {}".format(tp))
    return arch, tag


def gridSearch(paramList, templatestr, parameterstr, arch, tag):
    if len(paramList) == 0:
        runCommand(templatestr % {'tag': tag, 'arch': arch, 'batchsize': batchsize,
                                  'parameters': parameterstr, 'rundir':rundir, 'archdir':archdir})
    else:
        key = paramList.keys()[0]
        values = paramList[key]
        newParamList = copy.deepcopy(paramList)
        del newParamList[key]
        for paramval in values:
            newparamstr = parameterstr + ' ' + ('' if key[0] == '%' else key) + ' ' + str(paramval)
            gridSearch(newParamList, templatestr, newparamstr, arch, tag)

def scheduleExperiment(parameters, template, totalkw, kw0, totalparams, ninput, nlabel, featratio, depth, dropout0, dropoutN):
    arch, archname = generateConvArch(totalkw, kw0, totalparams, ninput, nlabel, featratio, depth, dropout0, dropoutN)
    filenameArch = "arch2_" + archname
    filenameArch = filenameArch
    farch = open(archdir + filenameArch, "w+")
    farch.write(arch)
    farch.close()
    gridSearch(copy.deepcopy(parameters), template, "", filenameArch, filenameArch)


def scheduleTrain():
    template =("if [ -f \"%(rundir)s/%(tag)s/001_model_last.bin\" ]; then \n "
               "/usr/local/fbcode/gcc-5-glibc-2.23/bin/mpirun -n %(batchsize)d /mnt/vol/gfsai-east/ai-group/users/vitaliy888/train_cpp "
               "continue %(rundir)s/%(tag)s " +
               " --linseg=0 "
               " --valid=/mnt/vol/gfsai-oregon/langtech/users/jaym/tasks/wav2letter/datasets/video_en_2018_08_30/val_30sec.metadata "
               " --enable_distributed \n"
               " else  \n"
               "/usr/local/fbcode/gcc-5-glibc-2.23/bin/mpirun -n %(batchsize)d /mnt/vol/gfsai-east/ai-group/users/vitaliy888/train_cpp "
               "train --lr 1.0 --lrcrit=0.001 --target=ltr --mfsc --archdir %(archdir)s "
               "--tokensdir=/mnt/vol/gfsai-east/ai-group/teams/wav2letter/examples/dict "
               "--rundir=%(rundir)s --runname=%(tag)s "
               "--train=/mnt/vol/gfsai-oregon/langtech/users/jaym/tasks/wav2letter/datasets/video_en_2018_08_30/max_possible_segment/train_30s_3w.metadata "
               "--valid=/mnt/vol/gfsai-oregon/langtech/users/jaym/tasks/wav2letter/datasets/video_en_2018_08_30/val_30sec.metadata "
               "--arch=%(arch)s --criterion=asg --tokens=letters.lst "
               "--melfloor=1.0 --linseg=1 --momentum=0.9 --maxgradnorm=0.1 --replabel=2 --surround=\"|\" --onorm=target --sqnorm --nthread=6 "
               "--batchsize=8 --reportiters=554 --everstoredb --skipoov --localnrmlleftctx 108 "
               "--enable_distributed "
               " %(parameters)s \n"
               "fi")

    parameters = {}
    parameters['--filterbanks'] = [80]

    for param_ratio in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        for depth in [15, 18, 21]:
            for kw0 in [3, 8]:
                scheduleExperiment(parameters, template, totalkw=218, kw0=kw0, totalparams=80000000, ninput=80, nlabel=30, featratio=param_ratio, depth=depth, dropout0=0.2, dropoutN=0.5)


scheduleTrain()
