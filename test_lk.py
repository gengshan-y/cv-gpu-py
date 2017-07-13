import pbcvt
pbcvt.set_dev(4)
import cv2
import numpy as np
import pdb
import sys
sys.path.insert(0,'/home/gengshan/wapr/mdp_track')
from mot_tracking.utils import bb_points,draw_flow
import time

lk_params = dict( winSize  = (4, 4),
       maxLevel = 1,
       criteria = (cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS, 20, 0.03),
       flags = cv2.OPTFLOW_USE_INITIAL_FLOW)



pdb.set_trace()
 
size = 1000

for b in range(50,1000,50):
  #_bb = np.asarray([0, b, 0,b])
  _bb = np.asarray([375, 500, 475,600])
  p0 = bb_points(_bb, 4, 4, [5,2])
  p1 = p0.copy()
  p2 = p0.copy()

  totalt=0.
  for i in range(100):
    im0 = cv2.cvtColor(cv2.imread('/data01/gsy/mdp_test/images/%06d.jpg'%(i+5000)),\
                    cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(cv2.imread('/data01/gsy/mdp_test/images/%06d.jpg'%(i+5001)),\
                    cv2.COLOR_BGR2GRAY)
    im0 = cv2.resize(im0,(size,size))
    im1 = cv2.resize(im1,(size,size))
    beg = time.time()
    st1,p1 = pbcvt.pyrlk_optflow_spr(im0,im1,np.expand_dims(p0,0),\
                                     np.expand_dims(p1,0))
    p1 = p1[0]
    st2,p2 = pbcvt.pyrlk_optflow_spr(im1,im0,np.expand_dims(p1,0),\
                                     np.expand_dims(p0,0))
    p2 = p2[0]
    st = st1 * st2
    #t0,t1 = pbcvt.optflow_tvl1_gpu(im0,im1)
    #p1,st,_=cv2.calcOpticalFlowPyrLK(im0, im1, p0, p1,**lk_params)
    #p2,st,_=cv2.calcOpticalFlowPyrLK(im1, im0, p1.copy(), p2,**lk_params)
    totalt += (time.time() - beg)

  print '%d \t %f' % (b, totalt)

pbcvt.rel_mem()
