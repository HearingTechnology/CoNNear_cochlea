# for running the Transmision line model using python

from .cochlear_model import *
import warnings

#this relates to python 3.6 on ubuntu
#there is one future warning related to "scipy.signal.decimate" in this file
#there is one runtime warning related to firwin "scipy.signal.decimate" in ic_cn2017.py (not important)
#so we suppress these warnings here
warnings.filterwarnings("ignore")

def solve_one_cochlea(model): #definition here, to have all the parameter implicit
    ii=model[3]
    coch=model[0]
    opts = model[4]
    
    sheraPo = opts['sheraPo']
    storeflag = opts ['storeflag']
    probe_points = opts ['probe_points']
    Fs = opts ['Fs']
    subjectNo = opts ['subjectNo']
    sectionsNo = opts ['sectionsNo']
    output_folder = opts ['output_folder']
    numH = opts ['numH'] 
    numM = opts ['numM'] 
    numL = opts ['numL']
    IrrPct = opts ['IrrPct']
    nl = opts ['nl']
    L = opts ['L']
    
    coch.init_model(model[1],Fs,sectionsNo,probe_points,Zweig_irregularities=model[2],sheraPo=sheraPo,subject=subjectNo,IrrPct=IrrPct,non_linearity_type=nl)
    coch.solve()
    matcontent = {}
    matcontent [u'fs_bm'] = Fs    
    matcontent[u'v'] = coch.Vsolution
    matcontent[u'e'] = coch.oto_emission
    matcontent[u'cf'] = coch.cf

    return matcontent