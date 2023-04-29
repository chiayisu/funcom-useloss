import os
import sys
import traceback
import pickle
import h5py
import argparse
import collections
#from keras import metrics
import random
import tensorflow as tf
import numpy as np

import tensorflow.keras.backend as K

#import matplotlib as plt

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import multiprocessing
from itertools import product

from multiprocessing import Pool

from timeit import default_timer as timer

# example command to run:
# time python3 predict_saliency.py outdir/models/attendgru_E04_1672929668.h5 --gpu=0 --data=/nfs/projects/funcom/data/java/ --batch-size=10

def plot_saliency(model, batch, tdatstok, comstok):
    #text_class = 'Positive' if text_class==1 else 'Negative'
    #pred_labels = 'Positive' if pred_labels==1 else 'Negative'
    
    fun = 6 * 13
    inpos = 1 + fun
    
    #inp0 = batch[0]
    inp0 = [batch[0][0][inpos]], [batch[0][1][inpos]] # tdat input sequence (for attendgru and most models)
    #inp0 = batch[0]
    #inp0 = batch[0][0] # all inputs (for attendgru and most models)
    oup0 = batch[1][inpos] # word in com output gold/oracle word

    #inp0 = np.asarray(inp0)
    #inp0 = tf.Variable(inp0)
    print(inp0)
    #inp0 = tf.ragged.constant(inp0)
    inp0 = (tf.convert_to_tensor(inp0[0], dtype=tf.float32), tf.convert_to_tensor(inp0[1], dtype=tf.float32))
    #sys.exit()
    
    oup0 = (tf.convert_to_tensor(oup0, dtype=tf.float32))
    
    print(oup0)
    
    #print(batch[0][1][0])
    #sys.exit()
    #print(np.asarray(batch[0]).shape)

    #input_tensors = [model.input, K.learning_phase()]
    #model_input = model.layers[0].input # tdat input layer for attendgru
    #model_output = model.output[0][1]
    
    #tf.gradients(model_input, model_output)
    with tf.GradientTape() as tape:
        pred = model(inp0)#, training=False)
        #loss = tf.reduce_mean(pred-oup0)
        #loss = tf.reduce_mean(pred)
        #loss = tf.reduce_mean(oup0)
        pred = tf.reshape(pred, (10908))
        loss = keras.losses.categorical_crossentropy(oup0, pred)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    #tf.print(gradients[1], summarize=-1)
    for gradient in gradients:
        print(gradient.shape)
    print(gradients[11])
    saliency = K.sum(gradients[0], axis=1)
    #print(np.argmax(saliency))
    
    #words = set(batch[0][0][0])
    #words = [int(w) for w in words]
    #print(words)
    
    for w in batch[0][0][inpos]:
        w = int(w)
        print(tdatstok.i2w[w], w, float(saliency[w]))
        
    pw = np.argmax(batch[1][inpos])
    print(comstok.i2w[pw])
    
    pw = np.argmax(pred)
    print(comstok.i2w[pw])
    
    #sal = saliency[saliency > 0]
    #print(sal)
    
    sys.exit()
    
    #gradients = model.optimizer.get_gradients(model_output, model_input)
    
    #compute_gradients = K.function(inputs=input_tensors, outputs=gradients)
    #matrix = compute_gradients([inp0, oup0])[0][0]
    
    #print(matrix)
    
    #matrix = matrix[:len(pure_txt),:]

    #matrix_magnify=np.zeros((matrix.shape[0]*10,matrix.shape[1]))
    #for i in range(matrix.shape[0]):
    #    for j in range(10):
    #        matrix_magnify[i*10+j,:]=matrix[i,:]

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.imshow(normalize_array(np.absolute(matrix_magnify)), interpolation='nearest', cmap=plt.cm.Blues)
    #plt.yticks(np.arange(5, matrix.shape[0]*10, 10), pure_txt, weight='bold',fontsize=24)
    #plt.xticks(np.arange(0, matrix.shape[1], 50), weight='bold',fontsize=24)
    #plt.title('True Label: "{}" Predicted Label: "{}"'.format(text_class,pred_labels), weight='bold',fontsize=24)
    #plt.colorbar()
    #plt.show()
    #sys.quit()

def gendescr(model, batch, badfids, comseqpos, comstok, batchsize, config):
    
    comlen = config['comlen']
    
    fiddats = list(zip(*batch.values()))
    #tdats = np.array(tdats)
    #coms = np.array(coms)
    #fiddats = [ tdats, coms ]
    nfiddats = list()
    
    for fd in fiddats:
        fd = np.array(fd)
        nfiddats.append(fd)
    
    #print(comlen)

    for i in range(1, comlen):
        #fiddats[comseqpos] = coms
        results = model.predict(nfiddats, batch_size=batchsize)
        if(len(results)<=3):
            results = results[2]
        #print(len(results))
        #print(np.asarray(results).shape)
        #quit()
        for c, s in enumerate(results):
            nfiddats[comseqpos][c][i] = np.argmax(s)
            #print(c, i, np.argmax(s))

    final_data = {}
    for fid, com in zip(batch.keys(), nfiddats[comseqpos]):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def load_model_from_weights(modelpath, modeltype, datvocabsize, comvocabsize, smlvocabsize, datlen, comlen, smllen):
    config = dict()
    config['datvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['datlen'] = datlen # length of the data
    config['comlen'] = comlen # comlen sent us in workunits
    config['smlvocabsize'] = smlvocabsize
    config['smllen'] = smllen

    model = create_model(modeltype, config)
    model.load_weights(modelpath)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modelfile', type=str, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/funcom/data/javastmt/q90')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--with-graph', dest='withgraph', action='store_true', default=False)
    parser.add_argument('--with-calls', dest='withcalls', action='store_true', default=False)
    #parser.add_argument('--with-biodats', dest='withbiodats', action='store_true', default=False)
    parser.add_argument('--with-biodats', dest='withbiodats', type=str , default='vanilla')
    parser.add_argument('--with-simmats', dest='withsimmats', action='store_true', default=False)
    parser.add_argument('--simmat-file', dest='simmatfile', type=str, default='softmax_usec.pkl')
    parser.add_argument('--loss-type', dest='losstype', type=str, default='cce')
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset.pkl')
    parser.add_argument('--testval', dest='testval', type=str, default='test')
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)

    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modelfile = args.modelfile
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    modeltype = args.modeltype
    outfile = args.outfile
    datfile = args.datfile
    testval = args.testval
    withgraph = args.withgraph
    withcalls = args.withcalls
    withbiodats = False
    withsimmat = args.withsimmats
    simmatfile = args.simmatfile
    losstype = args.losstype
    vmemlimit = args.vmemlimit
    
    if args.withbiodats != 'vanilla':
        withbiodats = True
        biodatfile = args.withbiodats
    
    if outfile is None:
        outfile = modelfile.split('/')[-1]

    #K.set_floatx(args.dtype)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                if(vmemlimit > 0):
                    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vmemlimit)])
        except RuntimeError as e:
            print(e)

    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    
    from model import create_model
    from myutils import prep, drop, batch_gen, seq2sent

    from custom.graphlayers import GCNLayer

    prep('loading sequences... ')
    extradata = pickle.load(open('%s/dataset_short.pkl' % (dataprep), 'rb'))
    h5data = h5py.File('%s/dataset_seqs.h5' % (dataprep), 'r')
    
    seqdata = dict()
    seqdata['dt%s' % testval] = h5data.get('/dt%s' % testval)
    seqdata['ds%s' % testval] = h5data.get('/ds%s' % testval)
    seqdata['s%s' % testval] = h5data.get('/s%s' % testval)
    seqdata['c%s' % testval] = h5data.get('/c%s' % testval)
    
    drop()

    if withgraph:
        prep('loading graph data... ')
        graphdata = pickle.load(open('%s/dataset_graph.pkl' % (dataprep), 'rb'))
        for k, v in extradata.items():
            graphdata[k] = v
        extradata = graphdata
        drop()

    if withcalls:
        prep('loading call data... ')
        callnodes = pickle.load(open('%s/callsnodes.pkl' % (dataprep), 'rb'))
        calledges = pickle.load(open('%s/callsedges.pkl' % (dataprep), 'rb'))
        callnodesdata = pickle.load(open('%s/callsnodedata.pkl' % (dataprep), 'rb'))
        extradata['callnodes'] = callnodes
        extradata['calledges'] = calledges
        extradata['callnodedata'] = callnodesdata
        drop()

    if withbiodats:
        prep('loading biomodel results... ')
        biodats = pickle.load(open(biodatfile, 'rb'))
        #biodats = pickle.load(open('/nfs/home/cmc/dev/funcom_bio/funcom_mod1/funcom_bio/outdir/scratch/biodats.pkl', 'rb'))
        extradata['biodats'] = biodats
        drop()

    if withsimmat:
        prep('loading target comwords distribution')
        softmax_usemat = pickle.load(open('%s/useopt/%s' % (dataprep, simmatfile), 'rb'))
        extradata['target_dist'] = softmax_usemat
        drop()
        
    prep('loading tokenizers... ')
    comstok = extradata['comstok']
    tdatstok = extradata['tdatstok']
    sdatstok = tdatstok
    smlstok = extradata['smlstok']
    if withgraph:
        graphtok = extradata['graphtok']
    drop()

    prep('loading config... ')
    (modeltype, mid, timestart) = modelfile.split('_')
    (timestart, ext) = timestart.split('.')
    modeltype = modeltype.split('/')[-1]
    config = pickle.load(open(outdir+'/histories/'+modeltype+'_conf_'+timestart+'.pkl', 'rb'))

    comlen = config['comlen']
    #fid2loc = config['fidloc']['c'+testval] # fid2loc[fid] = loc
    loc2fid = config['locfid']['c'+testval] # loc2fid[loc] = fid
    #allfids = list(fid2loc.keys())
    allfidlocs = list(loc2fid.keys())

    drop()

    prep('loading model... ')
    #config['no-final-softmax'] = True
    config, model = create_model(modeltype, config)
    model.load_weights(modelfile)
    #model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras, "GCNLayer":GCNLayer)
    print(model.summary())
    drop()

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    #outfn = outdir+"/predictions/predict-{}.txt".format(outfile.split('.')[0])
    #outf = open(outfn, 'w')
    #print("writing to file: " + outfn)
    batch_sets = [allfidlocs[i:i+batchsize] for i in range(0, len(allfidlocs), batchsize)]
 
    prep("computing predictions...\n")
    for c, fidloc_set in enumerate(batch_sets):
        st = timer()
        
        #pcomseqs = list()
        #for fidloc in fidloc_set:
        #    pcomseqs.append(comstart)
        #    seqdata['c'+testval][fidloc] = comstart
            
        bg = batch_gen(h5data, extradata, testval, config, training=True)
        batch = bg.make_batch(fidloc_set)

        plot_saliency(model, batch, tdatstok, comstok)

        #batch_results = gendescr(model, batch, badfids, comseqpos, comstok, batchsize, config)

        #for key, val in batch_results.items():
            #print("{}\t{}\n".format(key, val))
            #quit()
        #    outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    #outf.close()        
    drop()
