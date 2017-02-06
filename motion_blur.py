# -*- coding: utf-8 -*-
"""Testing experimental motion blur feature"""

import sys
import os
import cv2
import numpy as np
from cv2 import calcOpticalFlowFarneback as sw_farneback_optical_flow
from butterflow.core import FlowMagnitudeDirectionInfo, ImageProcess, MotionBlur
from butterflow import motion


# debugging opts for motion blur
FRS_TO_INTERPOLATE          = 2
# MASK_MAGNITUDE_LIMIT        = 10     # if it moves more than px, make 1=white (opaque)
USE_SW                      = False
MOTION_BLUR_KERNEL_SIZE     = 50     # bigger -> blurrier
N_SAMPLES                   = 1600   # must be a perfect square: 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936, 2025, 2116, 2209, 2304, 2401, 2500


# so we dont have to split the flows later on if USE_SW=True
def optical_flow_fn(*args):
    fu = None   # flow(y,x)[0], flowx, col, horizontal component
    fv = None   # flow(y,x)[1], flowy, row, vertical
    if USE_SW:
        flow = sw_farneback_optical_flow(*args)  # flow(row,col)=flow(y,x)
        fu = flow[:,:,0]
        fv = flow[:,:,1]
    else:
        fu, fv = motion.ocl_farneback_optical_flow(*args)
    return fu, fv


def write_mask(md):
    s = 'hw'
    if USE_SW:
        s = 'sw'
    s = 'mask.%s.%d.jpg' % (s, FlowMagnitudeDirectionInfo.masks_written)
    cv2.imwrite(s, (md.M_scaled_mag*255.0).astype(np.uint8))
    print('Wrote: %s' % s)
    FlowMagnitudeDirectionInfo.masks_written += 1


desktopdir = ''
if sys.platform.startswith('darwin'):
    desktopdir = os.path.join(os.path.expanduser('~'), 'Desktop')
# fr dimensions must be divisible by 2, and if w>256, it must be divisible by 4
a = os.path.join(desktopdir, 'mpv-shot0004.jpg')
b = os.path.join(desktopdir, 'mpv-shot0005.jpg')

fr_1 = cv2.imread(a)
fr_2 = cv2.imread(b)

# uncomment to test blurring:
# kernel = MotionBlur.kernel(25, 90*4+45*(2*5+1), half=False)
# kernel = MotionBlur.kernel(51, 0, half=False)  # full=way more streaky, half=subdued blurring
# MotionBlur.visualize_kernel(kernel)
# blurred_fr = cv2.filter2D(fr_1, -1, kernel)
# Display.show(blurred_fr)
# exit(0)
# import pdb; pdb.set_trace()

r = fr_1.shape[0]
c = fr_1.shape[1]

fr_1_gr = cv2.cvtColor(fr_1, cv2.COLOR_BGR2GRAY)    # optflow_fn needs 1ch M
fr_2_gr = cv2.cvtColor(fr_2, cv2.COLOR_BGR2GRAY)

# optical_flow_fn(prev, next, fbargs...):
# returns displacement field, "flow", such that prev(y,x) ~ next(y+flow(y,x)[1],x+flow(y,x)[0])
fn = lambda x, y: optical_flow_fn(x,y,0.5,3,25,3,5,0.8,False,0)
fu, fv = fn(fr_1_gr,fr_2_gr)  # forward  displacement, apply flow to fr_2 -> fr_1
bu, bv = fn(fr_2_gr,fr_1_gr)  # backward displacement, apply flow to fr_1 -> fr_2


f_md = FlowMagnitudeDirectionInfo(fu, fv, maglimit=-1)  # apply flow+fr_2 -> fr_1
b_md = FlowMagnitudeDirectionInfo(bu, bv, maglimit=-1)  # apply flow+fr_1 -> fr_2

max_magnitude = np.fmax(f_md.max_magnitude, b_md.max_magnitude)
b_md.rescale_magnitude_data(max_magnitude)
write_mask(b_md)
f_md.rescale_magnitude_data(max_magnitude)
write_mask(f_md)

# Display.show((b_md.M_scaled_mag*255.0).astype(np.uint8))

sampler = MatrixSampler(r, c, N_SAMPLES)

def visualize_flows(md, fr, tag=0):  # `md`: FlowMagnitudeDirectionInfo, `fr`: fr to draw on
    sampler.draw(fr, center_points=True, rect_outlines=False, point_text=False)

    # fvisual_fr = np.zeros((r,c,3), dtype=np.uint8)
    fvisual_fr = fr.copy()
    fvisual = FlowVisualizer(sampler, md)
    fvisual.draw(fvisual_fr)

    fvisual_fr = ImageProcess.alpha_blend(fvisual_fr, fr, 0.4)  # blend: fr+centers+rects+arrows
    # Display.show(fvisual_fr)
    s = 'direction.%d.jpg' % tag
    cv2.imwrite(s, fvisual_fr)
    print('Wrote: %s' % s)


# b_md.M[:,:,1] = b_md.M[:,:,1] * f_md.M[:,:,1]
# b_md.M[:,:,2] = b_md.M[:,:,2] + f_md.M[:,:,2]
visualize_flows(b_md, fr_1, 0)    # arrows point to where fr is heading, bu+bv: (fr1->fr2), fu+fv: (fr2->fr1)
visualize_flows(f_md, fr_2, 1)
exit(0)

# TODO:
# (1) get general direction of the flow, by quadrant
# (2) construct a blur kernel going in the same direction as the flow, by quadrant
# (3) selectively blur based on magnitude, may have to construct a Graph
kernel = MotionBlur.kernel(97, 0, half=False)  # full=way more streaky, half=subdued blurring


# fr_1_32 = np.float32(fr_1) * 1/255.0
# fr_2_32 = np.float32(fr_2) * 1/255.0

# cv2.imwrite('motionblur.0.jpg', fr_1)
i = 1

print('Blurring...')
blurred_fr_1 = cv2.filter2D(fr_1, -1, kernel)
blurred_fr_2 = cv2.filter2D(fr_2, -1, kernel)

print('Blending...')
fr_1_32 = np.float32(ImageProcess.apply_mask(blurred_fr_1, fr_1, f_md.M_scaled_mag)) * 1/255.0
fr_2_32 = np.float32(ImageProcess.apply_mask(blurred_fr_2, fr_2, b_md.M_scaled_mag)) * 1/255.0

print('Interpolating...')
for x in motion.ocl_interpolate_flow(fr_1_32,fr_2_32,fu,fv,bu,bv,FRS_TO_INTERPOLATE):
    # blurred_fr = cv2.filter2D(x, -1, kernel)
    # result = ImageProcess.apply_mask(blurred_fr, x, b_md.M_scaled_mag)
    s = 'motionblur.%d.jpg' % i
    # cv2.imwrite(s, result)
    cv2.imwrite(s, x)
    print('Wrote: %s' % s)
    i += 1

# cv2.imwrite('motionblur.%d.jpg' % i, fr_2)

cv2.destroyAllWindows()

print('Done')
