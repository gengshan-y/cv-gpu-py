import numpy as np
import pbcvt
import cv2
import pdb
import time

lk_params = dict( winSize  = (10, 10),
       maxLevel = 1,
       criteria = (cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS, 20, 0.03),
       flags = cv2.OPTFLOW_USE_INITIAL_FLOW)

pbcvt.set_params(4,10,1)

def draw_flow(im00,im01,p0,p1):
  im0 = cv2.cvtColor( im00, cv2.COLOR_GRAY2RGB )
  im1 = cv2.cvtColor( im01, cv2.COLOR_GRAY2RGB )
  for p in p0:
    cv2.circle(im0, (p[0], p[1]), 1, (0, 255, 0), -1)
  for p in p1:
    cv2.circle(im1, (p[0], p[1]), 1, (0, 255, 0), -1)

  im0 = cv2.resize(im0,(600,600))
  im1 = cv2.resize(im1,(600,600))
  cv2.imshow('0',im0)
  cv2.imshow('1',im1)
  cv2.waitKey(0)


def bb_points(tmp, n0, n1, margin):
  bb = tmp.copy()
  bb[:2] += margin
  bb[2:] -= margin

  x = np.linspace(bb[0], bb[2], num=n0)
  y = np.linspace(bb[1], bb[3], num=n1)

  xv,yv = np.meshgrid(x,y)
  xv = xv.reshape(-1)
  yv = yv.reshape(-1)
  return np.vstack((xv,yv)).astype(np.float32).transpose()


for size in range(50,1000,50):
  _bb = np.asarray([375, 500, 475,600])
  p0 = bb_points(_bb, 4, 4, [5,2])  # 4x4 grid and margin (5,2)
  p1 = p0.copy()
  p2 = p0.copy()

  totalt=0.
  for i in range(100):
    im0 = cv2.imread('/data01/gsy/mdp_test/images/%06d.jpg'%(i+5000))
    #im0 = cv2.cvtColor(cv2.imread('/data01/gsy/mdp_test/images/%06d.jpg'%(i+5000)),\
                    #cv2.COLOR_BGR2GRAY)
    im1 = cv2.imread('/data01/gsy/mdp_test/images/%06d.jpg'%(i+5001))
    #im1 = cv2.cvtColor(cv2.imread('/data01/gsy/mdp_test/images/%06d.jpg'%(i+5001)),\
                    #cv2.COLOR_BGR2GRAY)
    im0 = cv2.resize(im0,(size,size))
    im1 = cv2.resize(im1,(size,size))
    beg = time.time()
    # gpu pyrflow
    st1,p1 = pbcvt.pyrlk_optflow_spr(im0,im1,np.expand_dims(p0,0),\
                                     np.expand_dims(p1,0))
    p1 = p1[0]
    st2,p2 = pbcvt.pyrlk_optflow_spr(im1,im0,np.expand_dims(p1,0),\
                                     np.expand_dims(p0,0))
    p2 = p2[0]

    # tvl1
    #t0,t1 = pbcvt.optflow_tvl1_gpu(im0,im1)

    # cpu pyrflow
    #p1,st,_=cv2.calcOpticalFlowPyrLK(im0, im1, p0, p1,**lk_params)
    #p2,st,_=cv2.calcOpticalFlowPyrLK(im1, im0, p1.copy(), p2,**lk_params)
    totalt += (time.time() - beg)

  print '%d \t %f' % (size, totalt)

pbcvt.rel_mem()
