from datautil import SparseVector,TrainingData
import numpy as np
class Data:
    def __init__(self):
        self.vids = []
        self.title = []
        self.img = []

    def parse(self,line,index={}):
        pars = line.strip('\n').split('\t')
        if 'vid' in index:
            vids = pars[index['vid']]
            self.vids = [int(x) for x in vids.split(' ')]
        if 'title' in index:
            titles = pars[index['title']].split(',')
            self.title = [SparseVector().parse(x) for x in titles]
        if 'img' in index:
            self.img = [float(x) for x in pars[index['img']].split(' ')]

class DataSet:
    def __init__(self):
        self.X = []
        self.Y = []

    def size(self):
        assert len(self.X)==len(self.Y)
        return len(self.X)

    def load(self,path):
        for line in open(path):
            line = line.strip('\n')
            pars = line.split('\t')
            x = Data()
            x.parse("\t".join(pars[0:2]),{"vid":0,"title":1})
            y = Data()
            y.parse("\t".join(pars[2:]),{"vid":0,"title":1})
            self.X.append(x)
            self.Y.append(y)

    def get_data(self,begin,batch_size,dssmDim):
        input_vids = [x.vids for x in self.X[begin:begin+batch_size]]
        input_title = sum([x.title for x in self.X[begin:begin+batch_size]],[])
        input_title = TrainingData.toSparseTensorValue(input_title,dssmDim)
        output_vids = [y.vids for y in self.Y[begin:begin+batch_size]]
        output_title = sum([y.title for y in self.Y[begin:begin+batch_size]],[])
        #print 'output_title size ', len(output_title)
        #print 'output_title size ',np.shape(output_title)
        output_title = TrainingData.toSparseTensorValue(output_title,dssmDim)
        return (input_vids,input_title,output_vids,output_title)



