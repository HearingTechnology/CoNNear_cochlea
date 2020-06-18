import numpy as np
from .solve_one_cochlea import solve_one_cochlea
from .cochlear_model import *
import multiprocessing as mp


def tl_vbm_and_oae(stim, L):
    '''
    DEFINE ALL THE PARAMTERS HERE
    ''' 
    #Define all the variables for the model
    sheraPdat = 'StartingPoles.dat'
    poles = []
    for line in open(sheraPdat, 'r'):
        poles.append(float(line.rstrip()))
    sheraPo = np.array(poles)
    irregularities = 1 #adaptIRR0! 
    # Create the opts file for passing the variables
    opts={}
    opts['sheraPo'] = sheraPo
    opts ['storeflag'] = 've'
    opts ['probe_points'] = 'abr'
    opts ['Fs'] = 100e3
    opts ['channels'] = np.min(stim.shape)
    opts ['subjectNo'] = 1
    opts ['sectionsNo'] = int(1e3)
    opts ['output_folder'] = os.getcwd() + "/"
    opts ['numH'] = 13.
    opts ['numM'] = 3.
    opts ['numL'] = 3.
    opts ['IrrPct'] = 0.05
    opts ['nl'] = 'vel'
    opts ['L'] = L

    irr_on = irregularities * np.ones((1, opts ['channels'])).astype('int')
    cochlear_list=[[cochlea_model(), stim[i], irr_on[0][i], i, opts] for i in range(opts ['channels'])]

    print("running human auditory model 2018: Verhulst, Altoe, Vasilkov")
    p=mp.Pool(mp.cpu_count(),maxtasksperchild=1)
    output = p.map(solve_one_cochlea,cochlear_list)
    p.close()
    p.join()

    print("cochlear simulation: done")

    return output