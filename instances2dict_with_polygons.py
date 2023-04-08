
from __future__ import print_function, absolute_import, division
import os, sys
from collections import namedtuple
import cv2
import numpy as np

class Instance(object):
    instID     = 0
    labelID    = 0
    pixelCount = 0
    medDist    = -1
    distConf   = 0.0

    def __init__(self, imgNp,imgNp_seg, instID):
        if (instID == -1):
            return
        self.instID     = int(instID)
        self.labelID    = int(self.getLabelID(imgNp,imgNp_seg,instID))
        self.pixelCount = int(self.getInstancePixels(imgNp, instID))

    def getLabelID(self,imgNp,imgNp_seg, instID): # function to pick semantic labels
        cls_id = np.unique(imgNp_seg[imgNp == instID])
        dcls_id = cls_id[0]
        c_id  = dcls_id 

        return c_id

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def toDict(self):
        buildDict = {}
        buildDict["instID"]     = self.instID
        buildDict["labelID"]    = self.labelID
        buildDict["pixelCount"] = self.pixelCount
        buildDict["medDist"]    = self.medDist
        buildDict["distConf"]   = self.distConf
        return buildDict

    def fromJSON(self, data):
        self.instID     = int(data["instID"])
        self.labelID    = int(data["labelID"])
        self.pixelCount = int(data["pixelCount"])
        if ("medDist" in data):
            self.medDist    = float(data["medDist"])
            self.distConf   = float(data["distConf"])

    def __str__(self):
        return "("+str(self.instID)+")"

Label = namedtuple( 'Label', ['name','id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color', 'm_color'])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color          multiplied color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0      ),
    Label(  'ship'                 ,  1 ,        0 , 'transport'       , 1       , True         , False        , (  0,  0, 63) , 4128768),
    Label(  'storage_tank'         ,  2 ,        1 , 'transport'       , 1       , True         , False        , (  0, 63, 63) , 4144896),
    Label(  'baseball_diamond'     ,  3 ,        2 , 'land'            , 2       , True         , False        , (  0, 63,  0) , 16128  ),
    Label(  'tennis_court'         ,  4 ,        3 , 'land'            , 2       , True         , False        , (  0, 63,127) , 8339200),
    Label(  'basketball_court'     ,  5 ,        4 , 'land'            , 2       , True         , False        , (  0, 63,191) , 12533504),
    Label(  'Ground_Track_Field'   ,  6 ,        5 , 'land'            , 2       , True         , False        , (  0, 63,255) , 16727808),
    Label(  'Bridge'               ,  7 ,        6 , 'land'            , 2       , True         , False        , (  0,127, 63) , 4161280),
    Label(  'Large_Vehicle'        ,  8 ,        7 , 'transport'       , 1       , True         , False        , (  0,127,127) , 8355584),
    Label(  'Small_Vehicle'        ,  9 ,        8 , 'transport'       , 1       , True         , False        , (  0,  0,127) , 8323072),
    Label(  'Helicopter'           , 10 ,        9 , 'transport'       , 1       , True         , False        , (  0,  0,191) , 12517376),
    Label(  'Swimming_pool'        , 11 ,       10 , 'land'            , 2       , True         , False        , (  0,  0,255) , 16711680),
    Label(  'Roundabout'           , 12 ,       11 , 'land'            , 2       , True         , False        , (  0,191,127) , 8371968),
    Label(  'Soccer_ball_field'    , 13 ,       12 , 'land'            , 2       , True         , False        , (  0,127,191) , 12549888),
    Label(  'plane'                , 14 ,       13 , 'transport'       , 1       , True         , False        , (  0,127,255) , 16744192),
    Label(  'Harbor'               , 15 ,       14 , 'transport'       , 1       , True         , False        , (  0,100,155) , 10183680),
]

m2label        = { label.m_color : label for label in labels           }
label2id = { label.name : label.id for label in labels }

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def instances2dict_with_polygons(seg_imageFileList,ins_imageFileList,verbose=False):
    imgCount     = 0
    instanceDict = {}
    #import pdb;pdb.set_trace()
    if not isinstance(seg_imageFileList, list):
        seg_imageFileList = [seg_imageFileList]

    if verbose:
        print("Processing {} images...".format(len(seg_imageFileList)))
        print("Processing {} images...".format(len(ins_imageFileList)))

    for imageFileName_seg,imageFileName_ins in zip(seg_imageFileList,ins_imageFileList):
        print("Segment file:",imageFileName_seg)
        print("Instance files:",imageFileName_ins)
        img = cv2.imread(imageFileName_ins) # (1738, 1956, 3) instance file
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_seg = cv2.imread(imageFileName_seg, cv2.IMREAD_COLOR) # (1738, 1956, 3) segmentation file
        img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)

        # Image as numpy array
        imgNp = np.array(img) # Gives h * w * 3 matrix
        imgNp_seg = np.array(img_seg)
        if not (imgNp.ndim and imgNp_seg.ndim) == 3:
            import pdb;pdb.set_trace(); 
        imgNp = imgNp[:,:,0] + 256 * imgNp[:,:,1] + 256*256*imgNp[:,:,2]
        imgNp_seg = imgNp_seg[:,:,0] + 256 * imgNp_seg[:,:,1] + 256*256*imgNp_seg[:,:,2]


        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp): #np.unique(imgNp)
        #for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp, imgNp_seg, instanceId)
            instanceObj_dict = instanceObj.toDict()

            #instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
            #if id2label[instanceObj.labelID].hasInstances:
            if m2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                #contour, hier = cv2_util.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour, hier = findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict['contours'] = polygons

            instances[m2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey= os.path.abspath(imageFileName_ins)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict
