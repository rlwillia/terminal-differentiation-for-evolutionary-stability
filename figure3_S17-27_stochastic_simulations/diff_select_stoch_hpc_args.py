import sys
import numpy as np
from differentiation_functions_ODE_integrase_mutations_abx_cheating_additive_burden_CIIE_20211129 import *
import numpy as np
import pandas as pd
import pickle
import time

muN = float(sys.argv[1])
burden = int(sys.argv[2])
n_cassettes = int(sys.argv[3])
n_int_cassettes = int(sys.argv[4])
cassettes_equal = bool(int(sys.argv[5]))
K = int(sys.argv[6])
Vmax_x = int(sys.argv[7])
abx_in = float(sys.argv[8])
kPL = float(sys.argv[9])
selection = sys.argv[10]
kdiff = float(sys.argv[11])
split_cassettes = bool(int(sys.argv[12]))
CIIE = bool(int(sys.argv[13]))
max_growths = int(sys.argv[14])
n_div = int(sys.argv[15])
n_sim = int(sys.argv[16])

muP = muN*round((100 - int(burden))/100,2)
Vmax = Vmax_x*1e6/6.022e23*422.36*1e6*3600

#Global Variables

# from bioscrape.simulator import py_simulate_model
# from bioscrape.simulator import py_simulate_model
# from bioscrape.simulator import ModelCSimInterface
# from bioscrape.simulator import DeterministicSimulator, SSASimulator
# from bioscrape.types import Model

    
# arguments for function
kMB = 1e-6
kMD = 1e-6
kMI = 1e-6
V_act = K/1e9
Km = 6.7
MIC = 1.1
dt = 0.01
t_max = 8
plotting = True
fixed_end = True
complete_growths = True
summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection','CIIE','n_div',
                                 'muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                 'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                 'n_growths','prod_integral','tot_integral','prod_frac_median',
                                 'prod_frac_avg','total_production','production_rate','genotypes',
                                 'species_ind_dict','results','production_array','time_array'])
if split_cassettes == False:
    circuit = 'diff_select'
else:
    circuit = 'diff_split_select'
if cassettes_equal:
    n_int_cassettes = n_cassettes
    
args_list = []
res_list = []
start_time = time.time()
args = (muN, muP, n_cassettes, n_int_cassettes, kdiff, kMB, kMD, kMI, K, kPL,V_act,Vmax,Km,MIC,abx_in,selection,
        t_max, dt, summary_cols, n_div,split_cassettes, max_growths, n_sim, plotting, fixed_end, complete_growths,CIIE)
res_list = run_diff_select_stoch_MP_batch(args)
    
stop_time = time.time()
df_results = pd.DataFrame(columns=summary_cols,data=res_list)
year = str(time.localtime().tm_year)
month = str(time.localtime().tm_mon)
if len(month) == 1: month = '0' + month
day = str(time.localtime().tm_mday)
if len(day) == 1: day = '0' + day
date = f'{year}{month}{day}'
if kPL==0:
    kPLexp=0
else:
    kPLexp = str(-1*int(np.log10(kPL)))
if CIIE:
    fname = f'{date}_{circuit}_{burden}burden_{int(K)}K_{n_cassettes}cassettes_{n_int_cassettes}int_{n_div}div_{int(kdiff*100)}kdiff_{Vmax_x}xabxdeg_{int(abx_in)}abx_{kPLexp}kPL_{selection}_CIIE.pkl'
else:
    fname = f'{date}_{circuit}_{burden}burden_{int(K)}K_{n_cassettes}cassettes_{n_int_cassettes}int_{n_div}div__{int(kdiff*100)}kdiff_{Vmax_x}xabxdeg_{int(abx_in)}abx_{kPLexp}kPL_{selection}.pkl'
df_results.to_pickle(fname)
print(f'{fname} took {(stop_time-start_time)/60} minutes to complete {n_sim} simulations')
   
          
    
          
    
    
        
