#import os #Uncomment when runing on local system
#os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources" #This solves the problem of not finding R #Uncomment when runing on local system

import rpy2.robjects as robjects
r = robjects.r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr


def r_sample_distb(data_arr,nums):
    r_data_arr = robjects.IntVector(data_arr)
    r['source']('scripts/r_kernel_estimate.R')
    k_val = robjects.globalenv['kestimate']
    k_result_r = k_val(r_data_arr, nums)
    # convert result to python
    k_result = list(k_result_r)
    return list(map(int, k_result))

