import numpy as np
import pandas as pd
import numba
import itertools
import multiprocessing as MP
import scipy.integrate
import scipy.signal
import time
#asdf
"""
Functions below are for the simulation of the naive circuit
"""
def naive_multicassette_mut_list_fixed_step(kMB, kPL, muN, muP, beta_p, n_cassettes,
                                   selection='recessive',muP_single_cassette=False, return_gens=True):
    
    # create unique genotypes
    states=np.array(['P','N'])
    x = itertools.combinations_with_replacement(states,n_cassettes)
    num_gens = sum(1 for i in x)
    gens = np.empty((num_gens,n_cassettes+1),'<U2')
    
    # Resistant and Sensitive cells
    abx_states = ['R','S']
    x = itertools.combinations_with_replacement(states,n_cassettes)
    for i, gen in enumerate(x):
        for j in range(n_cassettes):
            gens[i,j] = gen[j]
    gens[:,n_cassettes] = abx_states[0]
    gens_S = gens.copy()
    gens_S[:,n_cassettes] = abx_states[1]
    gens = np.concatenate([gens,gens_S])
     # production rate and growth rate arrays
    beta_ps = np.zeros(len(gens))
    mus = np.ones(len(gens))*muN
    abx_gammas = np.ones(len(gens))

    # create and fill out lists of source/dest/rates      
    source_list = []
    dest_list = []
    rate_list = []
    ps = np.zeros(len(gens),int)
    ns = np.zeros(len(gens),int)
    abxrs = np.empty(len(gens),'<U2')
    for i, c1 in enumerate(gens):
        ps[i] = np.sum(c1=='P')
        ns[i] = np.sum(c1=='N')
        abxrs[i] = c1[-1]
        if ps[i]>0: # if at least one producer cassette, producer growth growth and production is on
            if selection == 'additive':
                if muP_single_cassette:
                    mus[i] = muN*(muP/muN)**ps[i]
    #                 mus[i] = muN - ps[i]/n_cassettes*(muN-muP)
                    beta_ps[i] = beta_p * (muN-mus[i])/(muN-muP)
                else:
                    mus[i] = muN*(muP/muN)**(ps[i]/n_cassettes)
    #                 mus[i] = muN - ps[i]/n_cassettes*(muN-muP)
                    beta_ps[i] = beta_p * (muN-mus[i])/(muN-muP)
    #                 beta_ps[i] = beta_p*ps[i]/n_cassettes    
            elif selection == 'recessive':
                mus[i] = muP
                beta_ps[i] = beta_p
            else:
                print('defaulting to recessive selection')
                mus[i] = muP
                beta_ps[i] = beta_p
        if abxrs[i] == 'S':
            mus[i] = muN
            beta_ps[i] = 0
            abx_gammas[i] = 0
       

    for i in range(num_gens):
        if ps[i]>0:
            mutb_to = np.argwhere(((ps==ps[i]-1)&\
                                   (ns==ns[i]+1)&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutb_to]
            rate_list += [ps[i]*kMB]
        if abxrs[i] == 'R':
            plos_to = np.argwhere(((ps==ps[i])&\
                                   (ns==ns[i])&\
                                   (abxrs=='S'))).flatten()[0]
            source_list += [i]
            dest_list += [plos_to]
            rate_list += [kPL]

    # make full list of genotypes
    species = ['',]*len(gens)
    for i in range(len(gens)):
        for c in gens[i]: species[i] += c

    # make dictionary of producer, progenitor producer, mutant, and non-producer genotypes
    producer_gens = []
    producer_gens_ind = []
    nonproducer_gens = []
    nonproducer_gens_ind = []
    degrader_gens = []
    degrader_gens_ind = []
    cheater_gens = []
    cheater_gens_ind = []
    for i in range(len(gens)):
        if (ps[i] > 0) and (abxrs[i]=='R'): 
            producer_gens += [species[i]]
            producer_gens_ind += [i]
        else: 
            nonproducer_gens += [species[i]]
            nonproducer_gens_ind += [i]
        if abxrs[i] == 'R':
            degrader_gens += [species[i]]
            degrader_gens_ind += [i]
        else:
            cheater_gens += [species[i]]
            cheater_gens_ind += [i]
            
    pn_dict = {'producers':producer_gens,'mutants':nonproducer_gens,'degraders':degrader_gens,'cheaters':cheater_gens}
    pn_ind_dict = {'producers':producer_gens_ind,'mutants':nonproducer_gens_ind,
                   'degraders':degrader_gens_ind,'cheaters':cheater_gens_ind}
    
    if return_gens:
        return np.array(source_list), np.array(dest_list),\
               np.array(rate_list), mus, beta_ps, abx_gammas, pn_dict, pn_ind_dict, species
    else:
        return np.array(source_list), np.array(dest_list),\
               np.array(rate_list), mus, beta_ps, abx_gammas, pn_dict, pn_ind_dict
    
@numba.jit(nopython=True)
def multicassette_naive_fixed_step_update(
    concs,K,D,V,Vmax,Km,MIC,abx_in,source,dest,rates,mus,beta_ps,abx_gammas, dt,dt_mut,stochastic,mut
    ):
    
    
    # allocate array for ddts
    ddts = np.zeros(len(concs))

    # total cell count
    ctot = concs[0:-2].sum()

    # carrying capacity limited growth and production
    MOD = (K-ctot)/K
    
    # growth and washout of cells, abx sensitive cells do not grow if [abx]>MIC
    ddts[0:-2] += mus*MOD*concs[0:-2]*(1-Heaviside_array(1-abx_gammas,0)*Heaviside(concs[-2]-MIC,1))*dt - D*concs[0:-2]*dt
    
    # antibiotic degradation and dilution
    ddts[-2] += -np.sum(concs[0:-2]*abx_gammas)*(Vmax/V)*concs[-2]/(concs[-2]+Km)*dt + D*(abx_in-concs[-2])*dt
    
    # production
    ddts[-1] += np.sum(concs[0:-2]*MOD*beta_ps)*dt
    
    # mutation
    if mut:
        if stochastic:
#             concs[0:-1] += multicassette_naive_fs_mut_stochastic(source,dest,rates*dt_mut,concs[0:-1])
            ddts[0:-2] += multicassette_naive_fs_mut_stochastic(source,dest,mus[source]*MOD*rates*dt_mut,concs[0:-2])
        else:
#             concs[0:-1] += multicassette_naive_fs_mut_determ(source,dest,rates*dt_mut,concs[0:-1])
            ddts[0:-2] += multicassette_naive_fs_mut_determ(source,dest,mus[source]*MOD*rates*dt_mut,concs[0:-2])
    
    return concs + ddts

@numba.jit(nopython=True)
def multicassette_naive_fs_mut_stochastic(source,dest,rates,concs):
    mut_update = np.zeros(len(concs))
    for i in range(len(source)):
        if concs[source[i]]>1:
            n_mut = np.random.binomial(int(concs[source[i]]),rates[i])
            mut_update[source[i]] -= n_mut
            mut_update[dest[i]] += n_mut
    return mut_update

@numba.jit(nopython=True)
def multicassette_naive_fs_mut_determ(source,dest,rates,concs):
    mut_update = np.zeros(len(concs))
    for i in range(len(source)):
        if concs[source[i]]>0:
            n_mut = concs[source[i]]*rates[i]
            mut_update[source[i]] -= n_mut
            mut_update[dest[i]] += n_mut
    return mut_update

    
### pick up here ###
                                       
@numba.jit(nopython=True)
def multicassette_naive_fixed_step_growth(out,K,V,Vmax,Km,MIC,source,dest,rates,mus,beta_ps,abx_gammas,dt,dt_mut,stochastic,
                                         frac_at_dilution,fixed_end=False):
    for i in range(out.shape[0]-1):
        if round(i*dt/dt_mut,6) == int(round(i*dt/dt_mut,6)):    
            i_1 = multicassette_naive_fixed_step_update(out[i,:],K,0,V,Vmax,Km,MIC,0,source,dest,rates,mus,
                                                    beta_ps,abx_gammas,dt,dt_mut,stochastic,True)
        else:
            i_1 = multicassette_naive_fixed_step_update(out[i,:],K,0,V,Vmax,Km,MIC,0,source,dest,rates,mus,
                                                    beta_ps,abx_gammas,dt,dt_mut,stochastic,False)
        i_1[i_1<0] = 0
        out[i+1,:] = i_1
    
    pop_tots = out[:,0:-2].sum(axis=1)
    
    if fixed_end == False:
        if pop_tots[-1] < (frac_at_dilution*K):
            return out
        else:
            end_loc = do_argwhere(pop_tots,K,frac_at_dilution)
            return out[0:end_loc,:]
    else:
        return out
    
@numba.jit(nopython=True)
def Heaviside_array(x1,x2):
    out = np.zeros(len(x1))
    for i, x in enumerate(x1):
        if x < 0:
            out[i] = 0
        elif x == 0:
            out[i] = x2
        else:
            out[i] = 1
    return out
@numba.jit(nopython=True)
def Heaviside(x1,x2) :
    if x1 < 0:
        return 0
    elif x1 == 0:
        return x2
    else:
        return 1
        
@numba.jit(nopython=True)
def do_argwhere(pop_tots,K,frac_at_dilution):
    for i in range(len(pop_tots)):
        if pop_tots[i] > frac_at_dilution*K:
            return i
        
def multicassette_naive_fixed_step_batch(t, dt, dt_mut, cell_init, n_cassettes, muN, muP,
                                   beta_p, kMB, kPL, K, V, Vmax, Km, MIC, abx_in, dilution_factor=50, 
                                   frac_at_dilution=0.95,prod_frac_thresh=1e-3,
                                   stochastic=False,max_growths=None,selection='recessive',
                                   plotting=False,fixed_end=False,complete_growths=False, muP_single_cassette=False):
    if int(dt_mut/dt) != (dt_mut/dt):
        print('dt_mut must be an integer multiple of dt')
        return
                                       

    source, dest, rates, mus, beta_ps, abx_gammas, pn_dict, pn_ind_dict, species = \
            naive_multicassette_mut_list_fixed_step(kMB, kPL, muN, muP, beta_p, 
                                            n_cassettes, selection, muP_single_cassette, return_gens=True)
    
    
    concs = np.zeros(len(mus)+2)
    concs[0] = cell_init
    concs[-2] = abx_in
    production = 0
    dilute_again = True
    producer_locs = pn_ind_dict['producers']
    nonproducer_locs = pn_ind_dict['mutants']
    n=0
    if plotting:
        concs_endpoint = []
        times_endpoint = []
    while dilute_again:
        out = np.zeros((int(t/dt)+1,len(concs)))
        out[0,:]=concs
        out = multicassette_naive_fixed_step_growth(out,K,V,Vmax,Km,MIC,source,dest,rates,mus,
                                                    beta_ps,abx_gammas,dt,dt_mut, stochastic,frac_at_dilution,fixed_end)
        end_concs = out[-1,:]
        if plotting:
            concs_endpoint.append(list(end_concs))
            times_endpoint.append(len(out)*dt - dt)
        if n == 0:
            out_concat = out
        else:
            out_concat = np.concatenate((out_concat,out))
        if type(max_growths) is int:
            if n+1 >= max_growths:
                dilute_again = False
        if complete_growths == False:
            if end_concs[0:-2].sum() < 2*dilution_factor:
                dilute_again = False
            elif end_concs[0:-2].sum() < 0.02*K:
                dilute_again = False
            elif (end_concs[producer_locs].sum()/end_concs[0:-2].sum()) < prod_frac_thresh:
                dilute_again = False
        
        if dilute_again:
            concs = end_concs
            if np.any(concs<0):
                concs[concs<0] = 0
            if np.any(np.isnan(concs)):
                print('nan')
            if stochastic:
                concs[0:-2] = np.random.binomial(list(concs[0:-2]),1/dilution_factor)
            else:
                concs[0:-2] /= dilution_factor
            concs[-2] = concs[-2]/dilution_factor + (1-1/dilution_factor)*abx_in
        n +=1
    
    if plotting:
        return out_concat, np.array(concs_endpoint), np.array(times_endpoint), pn_dict, pn_ind_dict, n, species
    else:
        return out_concat, pn_dict, pn_ind_dict, n, species

# def multicassette_naive_fixed_step_batch_plotting(t, dt, dt_mut, cell_init, n_cassettes, muN, muP,
#                                    beta_p, kMB, K, dilution_factor=50, 
#                                    frac_at_dilution=0.95,prod_frac_thresh=1e-3,
#                                    stochastic=False,max_growths=None,selection='recessive'):
#     if int(dt_mut/dt) != (dt_mut/dt):
#         print('dt_mut must be an integer multiple of dt')
#         return

#     source, dest, rates, mus, beta_ps, pn_dict, pn_ind_dict, species = \
#             naive_multicassette_mut_list_fixed_step(kMB, muN, muP, beta_p, 
#                                             n_cassettes, selection, return_gens=True)
    
    
#     concs = np.zeros(len(mus)+1)
#     concs[0] = cell_init
#     production = 0
#     dilute_again = True
#     producer_locs = pn_ind_dict['producers']
#     nonproducer_locs = pn_ind_dict['mutants']
#     n=0
#     endpoint_concs = []
#     while dilute_again:
#         out = np.zeros((int(t/dt)+1,len(concs)))
#         out[0,:]=concs
#         out = multicassette_naive_fixed_step_growth(out,K,source,dest,rates,mus,
#                                                     beta_ps,dt,dt_mut, stochastic,frac_at_dilution)
        
#         end_concs = out[-1,:]
#         endpoint_concs.append(list(end_concs))
#         if type(max_growths) is int:
#             if n+1 >= max_growths:
#                 dilute_again = False
#         if end_concs[0:-1].sum() < 2*dilution_factor:
#             dilute_again = False
#         elif end_concs[0:-1].sum() < 0.02*K:
#             dilute_again = False
#         elif (end_concs[producer_locs].sum()/end_concs[0:-1].sum()) < prod_frac_thresh:
#             dilute_again = False
       
#         else:
#             concs = end_concs
#             if np.any(concs<0):
#                 concs[concs<0] = 0
#             if np.any(np.isnan(concs)):
#                 print('nan')
#             if stochastic:
#                 concs[0:-1] = np.random.binomial(list(concs[0:-1]),1/dilution_factor)
#             else:
#                 concs[0:-1] /= dilution_factor
#         n +=1
#     return np.array([endpoint_concs]), pn_dict, pn_ind_dict, n, species

"""
Functions below are for the simulation of the differentiation architecture without selection
"""
def diff_multicassette_mut_list_fixed_step(kdiff, kMB, kMD, kMI, kPL, muN, muP, beta_p,
                                           n_cassettes,n_int_cassettes,selection='recessive',CIIE=False, return_gens=False):

    #CIIE is copy-number indepedendent integrase expression
    states=np.array(['PP','PN','DP','DN','M-'])
    x = itertools.combinations_with_replacement(states,n_cassettes)
    num_gens = sum(1 for i in x)
    gens = np.empty((num_gens*(n_int_cassettes+1),n_cassettes+2),'<U2')
    
    # Resistant and Sensitive cells
    abx_states = ['R','S']
    x = itertools.combinations_with_replacement(states,n_cassettes)
#     diff_states=np.array(['+','-'])
#     y = itertools.combinations_with_replacement(diff_states,n_int_cassettes)
    for i, gen in enumerate(x):
        for k in range(n_int_cassettes+1):
            gens[i+k*num_gens,-2] = str(int(n_int_cassettes-k)) 
            for j in range(n_cassettes):
                gens[i+k*num_gens,j] = gen[j]
    
    gens[:,-1] = abx_states[0]
    gens_S = gens.copy()
    gens_S[:,-1] = abx_states[1]
    gens = np.concatenate([gens,gens_S])
    
    pps = np.zeros(len(gens),int)
    pns = np.zeros(len(gens),int)
    dps = np.zeros(len(gens),int)
    dns = np.zeros(len(gens),int)
    m_s = np.zeros(len(gens),int)
    integrases = np.zeros(len(gens),int)
    beta_ps = np.zeros(len(gens))
    mus = np.ones(len(gens))*muN
    abx_gammas = np.ones(len(gens))
    abxrs = np.empty(len(gens),'<U2')

    # create and fill out matrix with rates
    source_list = []
    dest_list = []
    rate_list = []
    diff_source_list =[]
    diff_dest_list = []
    diff_rate_list = []
    for i, c1 in enumerate(gens):
        pps[i] = np.sum(c1=='PP')
        pns[i] = np.sum(c1=='PN')
        dps[i] = np.sum(c1=='DP')
        dns[i] = np.sum(c1=='DN')
        m_s[i] = np.sum(c1=='M-')
        integrases[i] = int(c1[-2])
        abxrs[i] = c1[-1]
        if dps[i]>0:
            if selection == 'additive':
                mus[i] = muN*(muP/muN)**dps[i]
#                 new_mu = muN
#                 for j in range(dps[i]):
#                     new_mu -= new_mu*((muN-muP)/muN)                
                beta_ps[i] = beta_p * (muN-mus[i])/(muN-muP)
                              
            elif selection == 'recessive':
                beta_ps[i] = beta_p
                mus[i] = muP
        if abxrs[i] == 'S':
            mus[i] = muN
            beta_ps[i] = 0
            abx_gammas[i] = 0
    
    for i in range(len(pps)):
        if pps[i]>0:
            if integrases[i]>0:
                # Differentiation
                diff_to = np.argwhere(((pps==pps[i]-1)&\
                                       (pns==pns[i])&\
                                       (dps==dps[i]+1)&\
                                       (dns==dns[i])&\
                                       (m_s==m_s[i])&\
                                       (integrases==integrases[i])&\
                                       (abxrs==abxrs[i]))).flatten()[0]
                diff_source_list += [i]
                diff_dest_list += [diff_to]
                if CIIE:
                    diff_rate_list += [pps[i]/n_cassettes*kdiff]
                else:
                    diff_rate_list += [pps[i]/n_cassettes*integrases[i]/n_int_cassettes*kdiff]
    #             rate_list += [kdiff]

            # Burden mutation
            mutb_to = np.argwhere(((pps==pps[i]-1)&\
                                   (pns==pns[i]+1)&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutb_to]
            rate_list += [pps[i]*kMB]

            # Differentiation mutation
            mutd_to = np.argwhere(((pps==pps[i]-1)&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i]+1)&\
                                   (integrases==integrases[i]))&\
                                   (abxrs==abxrs[i])).flatten()[0]
            source_list += [i]
            dest_list += [mutd_to]
            rate_list += [pps[i]*kMD]

        if pns[i]>0:
            # Differentiation
            if integrases[i]>0:
                diff_to = np.argwhere(((pps==pps[i])&\
                                       (pns==pns[i]-1)&\
                                       (dps==dps[i])&\
                                       (dns==dns[i]+1)&\
                                       (m_s==m_s[i])&\
                                       (integrases==integrases[i])&\
                                       (abxrs==abxrs[i]))).flatten()[0]
                diff_source_list += [i]
                diff_dest_list += [diff_to]
                if CIIE:
                    diff_rate_list += [pns[i]/n_cassettes*kdiff]
                else:
                    diff_rate_list += [pns[i]/n_cassettes*integrases[i]/n_int_cassettes*kdiff]
    #             rate_list += [kdiff]

            # Differentiation mutation
            mutd_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i]-1)&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i]+1)&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutd_to]
            rate_list += [pns[i]*kMD]
        if dps[i]>0:
            # Burden mutation
            mutb_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i]-1)&\
                                   (dns==dns[i]+1)&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutb_to]
            rate_list += [dps[i]*kMB]
            
        if integrases[i] > 0:
            # integrase expression differentiation mutation
            mutint_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i]-1)&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutint_to]
            rate_list += [integrases[i]*kMI]
        if abxrs[i] == 'R':
            plos_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs=='S'))).flatten()[0]
            source_list += [i]
            dest_list += [plos_to]
            rate_list += [kPL]
            
    
    # make full list of genotypes
    species = ['',]*len(gens)
    for i in range(len(gens)):
        for c in gens[i]: species[i] += c
            
    # make dictionary of producer, progenitor producer, mutant, and non-producer genotypes
    producer_gens = []
    progenitor_gens = []
    mutant_gens = []
    producer_gens_ind = []
    progenitor_gens_ind = []
    mutant_gens_ind = []
    degrader_gens = []
    degrader_gens_ind = []
    cheater_gens = []
    cheater_gens_ind = []
    for i in range(len(gens)):
        if (dps[i] > 0) and (abxrs[i]=='R'): 
            producer_gens += [species[i]]
            producer_gens_ind += [i]
        elif (pps[i] > 0) and (abxrs[i]=='R'): 
            if integrases[i] > 0:
                progenitor_gens += [species[i]]
                progenitor_gens_ind += [i]
            else:
                mutant_gens += [species[i]]
                mutant_gens_ind += [i]
        else: 
            mutant_gens += [species[i]]
            mutant_gens_ind += [i]
        if abxrs[i] == 'R':
            degrader_gens += [species[i]]
            degrader_gens_ind += [i]
        else:
            cheater_gens += [species[i]]
            cheater_gens_ind += [i]
    ppm_dict = {'producers':producer_gens,'progenitors':progenitor_gens,'mutants':mutant_gens,
                'degraders':degrader_gens,'cheaters':cheater_gens}
    ppm_ind_dict = {'producers':producer_gens_ind,'progenitors':progenitor_gens_ind,'mutants':mutant_gens_ind,
                    'degraders':degrader_gens_ind,'cheaters':cheater_gens_ind}
    if return_gens:
        return np.array(source_list), np.array(dest_list),np.array(rate_list),\
               np.array(diff_source_list), np.array(diff_dest_list),np.array(diff_rate_list),\
               mus, beta_ps, abx_gammas, ppm_dict, ppm_ind_dict, species
    else:
        return np.array(source_list), np.array(dest_list),np.array(rate_list),\
               np.array(diff_source_list), np.array(diff_dest_list),np.array(diff_rate_list),\
               mus, beta_ps, abx_gammas, ppm_dict, ppm_ind_dict

@numba.jit(nopython=True)
def multicassette_diff_fixed_step_update(concs,K,D,V,Vmax,Km,MIC,abx_in,source,dest,rates,
                                         diff_source,diff_dest,diff_rates,mus,
                                         beta_ps,abx_gammas,dt,dt_mut,stochastic,mut):
    # allocate array for ddts
    ddts = np.zeros(len(concs))

    # total cell count
    ctot = concs[0:-2].sum()

    # fraction to modify growth rates and production rates
    MOD = (K-ctot)/K
    
     # growth and washout of cells, abx sensitive cells do not grow if [abx]>MIC
    ddts[0:-2] += mus*MOD*concs[0:-2]*(1-Heaviside_array(1-abx_gammas,0)*Heaviside(concs[-2]-MIC,1))*dt - D*concs[0:-2]*dt
    
    # antibiotic degradation and dilution
    ddts[-2] += -np.sum(concs[0:-2]*abx_gammas)*(Vmax/V)*concs[-2]/(concs[-2]+Km)*dt + D*(abx_in-concs[-2])*dt
    
    # production
    ddts[-1] += np.sum(concs[0:-2]*MOD*beta_ps)*dt
    
    # differentiation: deterministic
    ddts[0:-2] += multicassette_diff_fs_mut_determ(diff_source,diff_dest,diff_rates*dt_mut,concs[0:-2])
    if mut:
        if stochastic:
#             concs[0:-1] += multicassette_diff_fs_mut_stochastic(source,dest,rates*dt_mut,concs[0:-1])
            ddts[0:-2] += multicassette_diff_fs_mut_stochastic(source,dest,mus[source]*MOD*rates*dt_mut,concs[0:-2])
        else:
#             concs[0:-1] += multicassette_diff_fs_mut_determ(source,dest,rates*dt_mut,concs[0:-1])
            ddts[0:-2] += multicassette_diff_fs_mut_determ(source,dest,mus[source]*MOD*rates*dt_mut,concs[0:-2])

    return concs + ddts

@numba.jit(nopython=True)
def multicassette_diff_fs_mut_stochastic(source,dest,rates,concs):
    mut_update = np.zeros(len(concs))
    for i in range(len(source)):
        if concs[source[i]]>1:
            n_mut = np.random.binomial(int(concs[source[i]]),rates[i])
            mut_update[source[i]] -= n_mut
            mut_update[dest[i]] += n_mut
    return mut_update

@numba.jit(nopython=True)
def multicassette_diff_fs_mut_determ(source,dest,rates,concs):
    mut_update = np.zeros(len(concs))
    for i in range(len(source)):
        if concs[source[i]]>0:
            n_mut = concs[source[i]]*rates[i]
            mut_update[source[i]] -= n_mut
            mut_update[dest[i]] += n_mut
    return mut_update

@numba.jit(nopython=True)
def multicassette_diff_fixed_step_growth(out,K,V,Vmax,Km,MIC,source,dest,rates,diff_source,diff_dest,diff_rates,
                                         mus,beta_ps,abx_gammas,dt,dt_mut,stochastic,
                                         frac_at_dilution,fixed_end=False):
    for i in range(out.shape[0]-1):
        if round(i*dt/dt_mut,6) == int(round(i*dt/dt_mut,6)):  
            i_1 = multicassette_diff_fixed_step_update(out[i,:],K,0,V,Vmax,Km,MIC,0,source,dest,rates,
                                                       diff_source,diff_dest,diff_rates,mus,
                                                       beta_ps,abx_gammas,dt,dt_mut,stochastic,True)
        else:
            i_1 = multicassette_diff_fixed_step_update(out[i,:],K,0,V,Vmax,Km,MIC,0,source,dest,rates,
                                                       diff_source,diff_dest,diff_rates,mus,
                                                       beta_ps,abx_gammas,dt,dt_mut,stochastic,False)
        i_1[i_1<0] = 0
        out[i+1,:] = i_1
    pop_tots = out[:,0:-2].sum(axis=1)
    
    if fixed_end == False:
        if pop_tots[-1] < (frac_at_dilution*K):
            return out
        else:
            end_loc = do_argwhere(pop_tots,K,frac_at_dilution)
            return out[0:end_loc,:]
    else:
        return out
    
def multicassette_diff_fixed_step_batch(t, dt, dt_mut,cell_init, n_cassettes, n_int_cassettes, muN, muP,
                                        beta_p, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in, selection='recessive', 
                                        dilution_factor=50, frac_at_dilution=0.95,prod_frac_thresh=1e-3,
                                        stochastic=False, max_growths=None,plotting=False,fixed_end=False,
                                        complete_growths=False, CIIE=False):
    if int(dt_mut/dt) != (dt_mut/dt):
        print('dt_mut must be an integer multiple of dt')
        return
    source, dest, rates, diff_source, diff_dest, diff_rates, mus, beta_ps, abx_gammas, ppm_dict, ppm_ind_dict, species = \
            diff_multicassette_mut_list_fixed_step(kdiff,kMB, kMD, kMI, kPL, muN, muP, beta_p, 
                                            n_cassettes, n_int_cassettes, selection, CIIE=CIIE, return_gens=True)

    concs = np.zeros(len(mus)+2)
    concs[0] = cell_init
    concs[-2] = abx_in
    production = 0
    dilute_again = True
    producer_locs = np.argwhere(beta_ps==beta_p).flatten()
    nonproducer_locs = np.argwhere(beta_ps==0).flatten()
    
    n=0
    if plotting:
        concs_endpoint = []
        times_endpoint = []

    while dilute_again:
        out = np.zeros((int(t/dt)+1,len(concs)))
        out[0,:]=concs
        out = multicassette_diff_fixed_step_growth(out,K,V,Vmax,Km,MIC,source,dest,rates,
                                                   diff_source,diff_dest,diff_rates,mus,
                                                    beta_ps,abx_gammas,dt,dt_mut,stochastic,frac_at_dilution,fixed_end)
        
        end_concs = out[-1,:]
        if plotting:
            concs_endpoint.append(list(end_concs))
            times_endpoint.append(len(out)*dt - dt)
    
        if n == 0:
            out_concat = out
        else:
            out_concat = np.concatenate((out_concat,out))
        
        if type(max_growths) is int:
            if n+1 >= max_growths:
                dilute_again = False
        if complete_growths == False:
            if end_concs[0:-2].sum() < 2*dilution_factor:
                dilute_again = False
            elif end_concs[0:-2].sum() < 0.02*K:
                dilute_again = False
            elif (end_concs[producer_locs].sum()/end_concs[0:-2].sum()) < prod_frac_thresh:
                dilute_again = False
            
        if dilute_again:
            concs = end_concs
            if np.any(concs<0):
                concs[concs<0] = 0
            if stochastic:
                concs[0:-2] = np.random.binomial(list(concs[0:-2]),1/dilution_factor)
            else:
                concs[0:-2] /= dilution_factor
            
            # fresh antibiotic
            concs[-2] = concs[-2]/dilution_factor + (1-1/dilution_factor)*abx_in
        n +=1
        
    if plotting:
        return out_concat, np.array(concs_endpoint), np.array(times_endpoint), ppm_dict, ppm_ind_dict, n, species
    else:
        return out_concat, ppm_dict, ppm_ind_dict, n, species

"""
Functions below are for the simulation of the differentiation with selection architecture (both the identical cassette and split cassette versions)
"""
def diff_select_multicassette_mut_list_fixed_step(kdiff, kMB, kMD, kMI, kPL, muN, muP, beta_p,
                                           n_cassettes,n_int_cassettes, n_div, selection='recessive',
                                                  CIIE=False,return_gens=True):

    states=np.array(['PP','PN','DP','DN','M-'])
    x = itertools.combinations_with_replacement(states,n_cassettes)
    num_gens = sum(1 for i in x)
    gens = np.empty((num_gens*(n_int_cassettes+1),n_cassettes+2),'<U2')
    x = itertools.combinations_with_replacement(states,n_cassettes)

    
    for i, gen in enumerate(x):
        for k in range(n_int_cassettes+1):
            gens[i+k*num_gens,-2] = str(int(n_int_cassettes-k))
            for j in range(n_cassettes):
                gens[i+k*num_gens,j] = gen[j]
#     print(gens)
    # make abx Resistant and Sensitive genotypes
    abx_states = ['R','S']
    gens[:,-1] = abx_states[0]
#     print(gens)
    gens_S = gens.copy()
    gens_S[:,-1] = abx_states[1]
    gens = np.concatenate([gens,gens_S])
    
    
    pps = np.zeros(len(gens),int)
    pns = np.zeros(len(gens),int)
    dps = np.zeros(len(gens),int)
    dns = np.zeros(len(gens),int)
    m_s = np.zeros(len(gens),int)
    integrases = np.zeros(len(gens),int)
    abxrs = np.empty(len(gens),'<U2')
    
#     print(gens)
    for i, c1 in enumerate(gens):
        pps[i] = np.sum(c1=='PP')
        pns[i] = np.sum(c1=='PN')
        dps[i] = np.sum(c1=='DP')
        dns[i] = np.sum(c1=='DN')
        m_s[i] = np.sum(c1=='M-')
        integrases[i] = int(c1[-2])
        abxrs[i] = c1[-1]
        
    
    lim_div_locs = np.argwhere(dps+dns==n_cassettes).flatten()
    lim_div_gens = gens[lim_div_locs]
    lim_div_dps = dps[lim_div_locs]
    lim_div_dns = dns[lim_div_locs]
    lim_div_integrases = integrases[lim_div_locs]
    lim_div_abxrs = abxrs[lim_div_locs]
#     print(lim_div_gens)
    
    # create and fill out matrix with rates        
    source_list = []
    dest_list = []
    rate_list = []
    diff_source_list =[]
    diff_dest_list = []
    diff_rate_list = []
    beta_ps = np.zeros(len(gens)+n_div*len(lim_div_gens))
    mus = np.ones(len(gens)+n_div*len(lim_div_gens))*muN
    abx_gammas = np.ones(len(gens)+n_div*len(lim_div_gens))
    
    
    for i, c1 in enumerate(gens):
        if dps[i]>0:
            if selection == 'additive':
                mus[i] = muN*(muP/muN)**dps[i]
#                 new_mu = muN
#                 for j in range(dps[i]):
#                     new_mu -= new_mu*((muN-muP)/muN)                
                beta_ps[i] = beta_p * (muN-mus[i])/(muN-muP)                                
            elif selection == 'recessive':
                beta_ps[i] = beta_p
                mus[i] = muP
            else:
                raise Exception("Selection must be 'additive' or 'recessive'")
        if abxrs[i] == 'S':
            mus[i] = muN
            beta_ps[i] = 0
            abx_gammas[i] = 0
        if i in lim_div_locs:
            mus[i] *= -1
    
    for i in range(len(pps)):
        if pps[i]>0:
            if integrases[i]>0:
                # Differentiation
                diff_to = np.argwhere(((pps==pps[i]-1)&\
                                       (pns==pns[i])&\
                                       (dps==dps[i]+1)&\
                                       (dns==dns[i])&\
                                       (m_s==m_s[i])&\
                                       (integrases==integrases[i])&\
                                       (abxrs==abxrs[i]))).flatten()[0]
                diff_source_list += [i]
                diff_dest_list += [diff_to]
                if CIIE:
                    diff_rate_list += [pps[i]/n_cassettes*kdiff]
                else:
                    diff_rate_list += [pps[i]/n_cassettes*integrases[i]/n_int_cassettes*kdiff]
    #             rate_list += [pps[i]*kdiff]

            # Burden mutation
            mutb_to = np.argwhere(((pps==pps[i]-1)&\
                                   (pns==pns[i]+1)&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutb_to]
            rate_list += [pps[i]*kMB]
            
            # Differentiation mutation
            mutd_to = np.argwhere(((pps==pps[i]-1)&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i]+1)&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutd_to]
            rate_list += [pps[i]*kMD]

        if pns[i]>0:
            if integrases[i]>0:
                # Differentiation
                diff_to = np.argwhere(((pps==pps[i])&\
                                       (pns==pns[i]-1)&\
                                       (dps==dps[i])&\
                                       (dns==dns[i]+1)&\
                                       (m_s==m_s[i])&\
                                       (integrases==integrases[i])&\
                                       (abxrs==abxrs[i]))).flatten()[0]
                diff_source_list += [i]
                diff_dest_list += [diff_to]
                if CIIE:
                    diff_rate_list += [pns[i]/n_cassettes*kdiff]
                else:
                    diff_rate_list += [pns[i]/n_cassettes*integrases[i]/n_int_cassettes*kdiff]
    #             rate_list += [pns[i]*kdiff]
            
            # Differentiation mutation
            mutd_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i]-1)&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i]+1)&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutd_to]
            rate_list += [pns[i]*kMD]

        if dps[i]>0:
            # Burden mutation
            mutb_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i]-1)&\
                                   (dns==dns[i]+1)&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutb_to]
            rate_list += [dps[i]*kMB]
        if integrases[i] > 0:
            # integrase expression differentiation mutation
            mutint_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i]-1)&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutint_to]
            rate_list += [integrases[i]*kMI]
        if abxrs[i] == 'R':
            plos_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs=='S'))).flatten()[0]
            source_list += [i]
            dest_list += [plos_to]
            rate_list += [kPL]
            
    for i, c1 in enumerate(lim_div_gens):
        if dps[lim_div_locs[i]] > 0:
            mutb_to = np.argwhere(((lim_div_dps==lim_div_dps[i]-1)&\
                                   (lim_div_dns==lim_div_dns[i]+1)&\
                                   (lim_div_integrases==lim_div_integrases[i])&\
                                   (lim_div_abxrs==lim_div_abxrs[i]))).flatten()[0]
            if beta_ps[lim_div_locs[i]] == 0:
                mus[len(gens)+i::len(lim_div_gens)] = -muN
                beta_ps[len(gens)+i::len(lim_div_gens)] = 0
            else:
                if selection == 'additive':
                    new_mu = muN*(muP/muN)**dps[lim_div_locs[i]]   
                    mus[len(gens)+i::len(lim_div_gens)] = -new_mu             
                    beta_ps[len(gens)+i::len(lim_div_gens)] = beta_p * (muN-new_mu)/(muN-muP)   
                elif selection == 'recessive':
                    mus[len(gens)+i::len(lim_div_gens)] = -muP
                    beta_ps[len(gens)+i::len(lim_div_gens)] = beta_p
                else:
                    raise Exception("Selection must be additive or recessive")
                
            for j in range(n_div):
                source_list += [len(gens) + j*len(lim_div_gens) + i]
                dest_list += [len(gens) + j*len(lim_div_gens) + mutb_to]
                rate_list += [lim_div_dps[i]*kMB]
        else:
            mus[len(gens)+i::len(lim_div_gens)] = -muN
            beta_ps[len(gens)+i::len(lim_div_gens)] = 0
        if abxrs[lim_div_locs[i]] == 'R':
            abx_gammas[len(gens)+i::len(lim_div_gens)] = 1
            plos_to = np.argwhere(((lim_div_dps==lim_div_dps[i])&\
                                   (lim_div_dns==lim_div_dns[i])&\
                                   (lim_div_integrases==lim_div_integrases[i])&\
                                   (lim_div_abxrs=='S'))).flatten()[0]
            for j in range(n_div):
                source_list += [len(gens) + j*len(lim_div_gens) + i]
                dest_list += [len(gens) + j*len(lim_div_gens) + plos_to]
                rate_list += [kPL]
        else:
            abx_gammas[len(gens)+i::len(lim_div_gens)] = 0
            


    n_ldg = len(lim_div_gens) # number of limited division genotypes            
    diff_div_loss_locs = np.array(list(lim_div_locs) +\
                                  list(np.arange(len(gens),len(gens)+n_ldg*(n_div-1))))
    diff_div_gain_locs = np.arange(len(gens),len(gens)+n_ldg*(n_div))
    
    # make full list of genotypes
    species = ['',]*(len(gens)+n_div*n_ldg)
    for i in range(gens.shape[0]):
        for c in gens[i]: species[i] += c
    for i in range(n_div):
        for j in range(n_ldg):
            species[len(gens)+i*n_ldg+j] = species[lim_div_locs[j]] + '_' + str(i+1)
#     print(species)

    producer_gens = []
    progenitor_gens = []
    mutant_gens = []
    nonproducer_gens = []
    producer_gens_ind = []
    progenitor_gens_ind = []
    mutant_gens_ind = []
    nonproducer_gens_ind = []
    degrader_gens = []
    degrader_gens_ind = []
    cheater_gens = []
    cheater_gens_ind = []
    
    for i in range(len(gens)):
        if i in lim_div_locs:
            add_list = [species[i],]
            add_list_ind = [i,]
            for j in range(n_div): 
                add_list += [species[i] + '_' + str(j+1)]
                add_list_ind += [np.argwhere(np.array(species)==add_list[-1]).flatten()[0]]
        else: 
            add_list = [species[i]]
            add_list_ind = [i]
        if (dps[i] > 0) and (abxrs[i]=='R'):
            producer_gens += add_list
            producer_gens_ind += add_list_ind
        elif (pps[i] > 0) and (abxrs[i]=='R'):
            if integrases[i]>0:
                progenitor_gens += add_list
                progenitor_gens_ind += add_list_ind
            else:
                mutant_gens += add_list
                mutant_gens_ind += add_list_ind
        elif m_s[i] > 0: 
            mutant_gens += add_list
            mutant_gens_ind += add_list_ind
        else: 
            nonproducer_gens += add_list
            nonproducer_gens_ind += add_list_ind
        if abxrs[i] == 'R':
            degrader_gens += add_list
            degrader_gens_ind += add_list_ind
        else:
            cheater_gens += add_list
            cheater_gens_ind += add_list_ind

    ppmn_dict = {'producers':producer_gens,'progenitors':progenitor_gens,
                 'mutants':mutant_gens,'nonproducers':nonproducer_gens,
                 'degraders':degrader_gens,'cheaters':cheater_gens}
    ppmn_ind_dict = {'producers':producer_gens_ind,'progenitors':progenitor_gens_ind,
                     'mutants':mutant_gens_ind,'nonproducers':nonproducer_gens_ind,
                     'degraders':degrader_gens_ind,'cheaters':cheater_gens_ind}
        
    if return_gens:
        return np.array(source_list), np.array(dest_list), np.array(rate_list), \
               np.array(diff_source_list), np.array(diff_dest_list), np.array(diff_rate_list), mus,\
               beta_ps, abx_gammas, diff_div_loss_locs, diff_div_gain_locs, ppmn_dict, ppmn_ind_dict, species 
    else:
        return np.array(source_list), np.array(dest_list), np.array(rate_list), \
               np.array(diff_source_list), np.array(diff_dest_list), np.array(diff_rate_list), mus,\
           beta_ps, abx_gammas, diff_div_loss_locs, diff_div_gain_locs, ppmn_dict, ppmn_ind_dict

def diff_split_select_multicassette_mut_list_fixed_step(kdiff, kMB, kMD, kMI, kPL, muN, muP, beta_p,
                                           n_cassettes,n_int_cassettes, n_div, selection='recessive',
                                                        CIIE=False,return_gens=True):


    states=np.array(['PP','PN','DP','DN','M-'])
    x = itertools.combinations_with_replacement(states,n_cassettes)
    num_gens = sum(1 for i in x)
    gens = np.empty((num_gens*(n_int_cassettes+1),n_cassettes+2),'<U2')
    x = itertools.combinations_with_replacement(states,n_cassettes)
    for i, gen in enumerate(x):
        for k in range(n_int_cassettes+1):
            gens[i+k*num_gens,-2] = str(int(n_int_cassettes-k))
            for j in range(n_cassettes):
                gens[i+k*num_gens,j] = gen[j]
                
    # make abx Resistant and Sensitive genotypes
    abx_states = ['R','S']
    gens[:,-1] = abx_states[0]
#     print(gens)
    gens_S = gens.copy()
    gens_S[:,-1] = abx_states[1]
    gens = np.concatenate([gens,gens_S])
#     print(gens)
    
    pps = np.zeros(len(gens),int)
    pns = np.zeros(len(gens),int)
    dps = np.zeros(len(gens),int)
    dns = np.zeros(len(gens),int)
    m_s = np.zeros(len(gens),int)
    integrases = np.zeros(len(gens),int)
    abxrs = np.empty(len(gens),'<U2')
    
    for i, c1 in enumerate(gens):
        pps[i] = np.sum(c1=='PP')
        pns[i] = np.sum(c1=='PN')
        dps[i] = np.sum(c1=='DP')
        dns[i] = np.sum(c1=='DN')
        m_s[i] = np.sum(c1=='M-')
        integrases[i] = int(c1[-2])
        abxrs[i] = c1[-1]
        
    
    lim_div_locs = np.argwhere((dps+dns)>0).flatten()
    lim_div_gens = gens[lim_div_locs]
    lim_div_pps = pps[lim_div_locs]
    lim_div_pns = pns[lim_div_locs]
    lim_div_dps = dps[lim_div_locs]
    lim_div_dns = dns[lim_div_locs]
    lim_div_m_s = m_s[lim_div_locs]
    lim_div_integrases = integrases[lim_div_locs]
    lim_div_abxrs = abxrs[lim_div_locs]
    
    
    # create and fill out matrix with rates        
    source_list = []
    dest_list = []
    rate_list = []
    diff_source_list =[]
    diff_dest_list = []
    diff_rate_list = []
    beta_ps = np.zeros(len(gens)+n_div*len(lim_div_gens))
    mus = np.ones(len(gens)+n_div*len(lim_div_gens))*muN
    abx_gammas = np.ones(len(gens)+n_div*len(lim_div_gens))
    
    for i, c1 in enumerate(gens):
        if dps[i]>0:
            if selection == 'additive':
                mus[i] = muN*(muP/muN)**dps[i]
#                 new_mu = muN
#                 for j in range(dps[i]):
#                     new_mu -= new_mu*((muN-muP)/muN)                
                beta_ps[i] = beta_p * (muN-mus[i])/(muN-muP)               
            elif selection == 'recessive':
                beta_ps[i] = beta_p
                mus[i] = muP
            else:
                raise Exception("Selection must be 'additive' or 'recessive'")
        if abxrs[i] == 'S':
            mus[i] = muN
            beta_ps[i] = 0
            abx_gammas[i] = 0
        if i in lim_div_locs:
            mus[i] *= -1
    
    for i in range(len(pps)):
        if pps[i]>0:
            # Differentiation
            if integrases[i]>0:
                diff_to = np.argwhere(((pps==pps[i]-1)&\
                                       (pns==pns[i])&\
                                       (dps==dps[i]+1)&\
                                       (dns==dns[i])&\
                                       (m_s==m_s[i])&\
                                       (integrases==integrases[i])&\
                                       (abxrs==abxrs[i]))).flatten()[0]
                diff_source_list += [i]
                diff_dest_list += [diff_to]
                if CIIE:
                    diff_rate_list += [pps[i]/n_cassettes*kdiff]
                else:
                    diff_rate_list += [pps[i]/n_cassettes*integrases[i]/n_int_cassettes*kdiff]
    #             rate_list += [pps[i]*kdiff]

            # Burden mutation
            mutb_to = np.argwhere(((pps==pps[i]-1)&\
                                   (pns==pns[i]+1)&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutb_to]
            rate_list += [pps[i]*kMB]
            
            # Differentiation mutation
            mutd_to = np.argwhere(((pps==pps[i]-1)&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i]+1)&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutd_to]
            rate_list += [pps[i]*kMD]

        if pns[i]>0:
            if integrases[i]>0:
                # Differentiation
                diff_to = np.argwhere(((pps==pps[i])&\
                                       (pns==pns[i]-1)&\
                                       (dps==dps[i])&\
                                       (dns==dns[i]+1)&\
                                       (m_s==m_s[i])&\
                                       (integrases==integrases[i])&\
                                       (abxrs==abxrs[i]))).flatten()[0]
                diff_source_list += [i]
                diff_dest_list += [diff_to]
                if CIIE:
                    diff_rate_list += [pns[i]/n_cassettes*kdiff]
                else:
                    diff_rate_list += [pns[i]/n_cassettes*integrases[i]/n_int_cassettes*kdiff]
    #             rate_list += [pns[i]*kdiff]
            
            # Differentiation mutation
            mutd_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i]-1)&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i]+1)&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutd_to]
            rate_list += [pns[i]*kMD]

        if dps[i]>0:
            # Burden mutation
            mutb_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i]-1)&\
                                   (dns==dns[i]+1)&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutb_to]
            rate_list += [dps[i]*kMB]
        if integrases[i] > 0:
            # integrase expression differentiation mutation
            mutint_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i]-1)&\
                                   (abxrs==abxrs[i]))).flatten()[0]
            source_list += [i]
            dest_list += [mutint_to]
            rate_list += [integrases[i]*kMI]
        if abxrs[i] == 'R':
            plos_to = np.argwhere(((pps==pps[i])&\
                                   (pns==pns[i])&\
                                   (dps==dps[i])&\
                                   (dns==dns[i])&\
                                   (m_s==m_s[i])&\
                                   (integrases==integrases[i])&\
                                   (abxrs=='S'))).flatten()[0]
            source_list += [i]
            dest_list += [plos_to]
            rate_list += [kPL]
            
    for i, c1 in enumerate(lim_div_gens):
        if dps[lim_div_locs[i]] > 0:
            mutb_to = np.argwhere(((lim_div_pps==lim_div_pps[i])&\
                                   (lim_div_pns==lim_div_pns[i])&\
                                   (lim_div_dps==lim_div_dps[i]-1)&\
                                   (lim_div_dns==lim_div_dns[i]+1)&\
                                   (lim_div_m_s==lim_div_m_s[i])&\
                                   (lim_div_integrases==lim_div_integrases[i])&\
                                   (lim_div_abxrs==lim_div_abxrs[i]))).flatten()[0]
            if beta_ps[lim_div_locs[i]] == 0:
                mus[len(gens)+i::len(lim_div_gens)] = -muN
                beta_ps[len(gens)+i::len(lim_div_gens)] = 0
            else:
                if selection == 'additive':
                    new_mu = muN*(muP/muN)**dps[lim_div_locs[i]]   
                    mus[len(gens)+i::len(lim_div_gens)] = -new_mu             
                    beta_ps[len(gens)+i::len(lim_div_gens)] = beta_p * (muN-new_mu)/(muN-muP)
                elif selection == 'recessive':
                    mus[len(gens)+i::len(lim_div_gens)] = -muP
                    beta_ps[len(gens)+i::len(lim_div_gens)] = beta_p
                else:
                    raise Exception("Selection must be additive or recessive")
            for j in range(n_div):
                source_list += [len(gens) + j*len(lim_div_gens) + i]
                dest_list += [len(gens) + j*len(lim_div_gens) + mutb_to]
                rate_list += [lim_div_dps[i]*kMB]
        else:
            mus[len(gens)+i::len(lim_div_gens)] = -muN
            beta_ps[len(gens)+i::len(lim_div_gens)] = 0
        
        if pps[lim_div_locs[i]] > 0:
            if integrases[lim_div_locs[i]]>0:
                diff_to = np.argwhere(((lim_div_pps==lim_div_pps[i]-1)&\
                                       (lim_div_pns==lim_div_pns[i])&\
                                       (lim_div_dps==lim_div_dps[i]+1)&\
                                       (lim_div_dns==lim_div_dns[i])&\
                                       (lim_div_m_s==lim_div_m_s[i])&\
                                       (lim_div_integrases==lim_div_integrases[i])&\
                                       (lim_div_abxrs==lim_div_abxrs[i]))).flatten()[0]
                for j in range(n_div):
                    diff_source_list += [len(gens) + j*len(lim_div_gens) + i]
                    diff_dest_list += [len(gens) + j*len(lim_div_gens) + diff_to]
                    if CIIE:
                        diff_rate_list += [lim_div_pps[i]/n_cassettes*kdiff]
                    else:
                        diff_rate_list += [lim_div_pps[i]/n_cassettes*lim_div_integrases[i]/n_int_cassettes*kdiff]
                
            mutb_to = np.argwhere(((lim_div_pps==lim_div_pps[i]-1)&\
                                   (lim_div_pns==lim_div_pns[i]+1)&\
                                   (lim_div_dps==lim_div_dps[i])&\
                                   (lim_div_dns==lim_div_dns[i])&\
                                   (lim_div_m_s==lim_div_m_s[i])&\
                                   (lim_div_integrases==lim_div_integrases[i])&\
                                   (lim_div_abxrs==lim_div_abxrs[i]))).flatten()[0]
            for j in range(n_div):
                source_list += [len(gens) + j*len(lim_div_gens) + i]
                dest_list += [len(gens) + j*len(lim_div_gens) + mutb_to]
                rate_list += [lim_div_pps[i]*kMB]
                
            mutd_to = np.argwhere(((lim_div_pps==lim_div_pps[i]-1)&\
                                   (lim_div_pns==lim_div_pns[i])&\
                                   (lim_div_dps==lim_div_dps[i])&\
                                   (lim_div_dns==lim_div_dns[i])&\
                                   (lim_div_m_s==lim_div_m_s[i]+1)&\
                                   (lim_div_integrases==lim_div_integrases[i])&\
                                   (lim_div_abxrs==lim_div_abxrs[i]))).flatten()[0]
            for j in range(n_div):
                source_list += [len(gens) + j*len(lim_div_gens) + i]
                dest_list += [len(gens) + j*len(lim_div_gens) + mutd_to]
                rate_list += [lim_div_pps[i]*kMD]
            if integrases[lim_div_locs[i]] > 0:
                muti_to = np.argwhere(((lim_div_pps==lim_div_pps[i])&\
                                       (lim_div_pns==lim_div_pns[i])&\
                                       (lim_div_dps==lim_div_dps[i])&\
                                       (lim_div_dns==lim_div_dns[i])&\
                                       (lim_div_m_s==lim_div_m_s[i])&\
                                       (lim_div_integrases==lim_div_integrases[i]-1)&\
                                       (lim_div_abxrs==lim_div_abxrs[i]))).flatten()[0]
                for j in range(n_div):
                    source_list += [len(gens) + j*len(lim_div_gens) + i]
                    dest_list += [len(gens) + j*len(lim_div_gens) + muti_to]
                    rate_list += [lim_div_integrases[i]*kMI]
        if abxrs[lim_div_locs[i]] == 'R':
            abx_gammas[len(gens)+i::len(lim_div_gens)] = 1
            plos_to = np.argwhere(((lim_div_pps==lim_div_pps[i])&\
                                   (lim_div_pns==lim_div_pns[i])&\
                                   (lim_div_dps==lim_div_dps[i])&\
                                   (lim_div_dns==lim_div_dns[i])&\
                                   (lim_div_m_s==lim_div_m_s[i])&\
                                   (lim_div_integrases==lim_div_integrases[i])&\
                                   (lim_div_abxrs=='S'))).flatten()[0]
            for j in range(n_div):
                source_list += [len(gens) + j*len(lim_div_gens) + i]
                dest_list += [len(gens) + j*len(lim_div_gens) + plos_to]
                rate_list += [kPL]
        else:
            abx_gammas[len(gens)+i::len(lim_div_gens)] = 0

    n_ldg = len(lim_div_gens)            
    diff_div_loss_locs = np.array(list(lim_div_locs) +\
                                  list(np.arange(len(gens),len(gens)+n_ldg*(n_div-1))))
    diff_div_gain_locs = np.arange(len(gens),len(gens)+n_ldg*(n_div))
    
    # make full list of genotypes
    species = ['',]*(len(gens)+n_div*n_ldg)
    for i in range(gens.shape[0]):
        for c in gens[i]: species[i] += c
    for i in range(n_div):
        for j in range(n_ldg):
            species[len(gens)+i*n_ldg+j] = species[lim_div_locs[j]] + '_' + str(i+1)
#     print(species)
    producer_gens = []
    progenitor_gens = []
    mutant_gens = []
    nonproducer_gens = []
    producer_gens_ind = []
    progenitor_gens_ind = []
    mutant_gens_ind = []
    nonproducer_gens_ind = []
    degrader_gens = []
    degrader_gens_ind = []
    cheater_gens = []
    cheater_gens_ind = []
    
    for i in range(len(gens)):
        if i in lim_div_locs:
            add_list = [species[i],]
            add_list_ind = [i,]
            for j in range(n_div): 
                add_list += [species[i] + '_' + str(j+1)]
                add_list_ind += [np.argwhere(np.array(species)==add_list[-1]).flatten()[0]]
        else: 
            add_list = [species[i]]
            add_list_ind = [i]
        if (dps[i] > 0) and (abxrs[i]=='R'):
            producer_gens += add_list
            producer_gens_ind += add_list_ind
        elif (pps[i] > 0) and (abxrs[i]=='R'): 
            if integrases[i]>0:
                progenitor_gens += add_list
                progenitor_gens_ind += add_list_ind
            elif dns[i]>0:
                nonproducer_gens += add_list
                nonproducer_gens_ind += add_list_ind
            else:
                mutant_gens += add_list
                mutant_gens_ind += add_list_ind
        elif m_s[i] == n_cassettes: 
            mutant_gens += add_list
            mutant_gens_ind += add_list_ind
        else: 
            nonproducer_gens += add_list
            nonproducer_gens_ind += add_list_ind
        if abxrs[i] == 'R':
            degrader_gens += add_list
            degrader_gens_ind += add_list_ind
        else:
            cheater_gens += add_list
            cheater_gens_ind += add_list_ind
    ppmn_dict = {'producers':producer_gens,'progenitors':progenitor_gens,
                 'mutants':mutant_gens,'nonproducers':nonproducer_gens,
                 'degraders':degrader_gens,'cheaters':cheater_gens}
    ppmn_ind_dict = {'producers':producer_gens_ind,'progenitors':progenitor_gens_ind,
                     'mutants':mutant_gens_ind,'nonproducers':nonproducer_gens_ind,
                     'degraders':degrader_gens_ind,'cheaters':cheater_gens_ind}
    if return_gens:
        return np.array(source_list), np.array(dest_list), np.array(rate_list), \
               np.array(diff_source_list), np.array(diff_dest_list), np.array(diff_rate_list), mus,\
               beta_ps, abx_gammas, diff_div_loss_locs, diff_div_gain_locs, ppmn_dict, ppmn_ind_dict,species
    else:
        return np.array(source_list), np.array(dest_list), np.array(rate_list), \
               np.array(diff_source_list), np.array(diff_dest_list), np.array(diff_rate_list), mus,\
           beta_ps, abx_gammas, diff_div_loss_locs, diff_div_gain_locs, ppmn_dict, ppmn_ind_dict 
@numba.jit(nopython=True)
def multicassette_diff_select_fixed_step_update(concs, K,D,V,Vmax,Km,MIC,abx_in,
                                                source,dest,rates,diff_source,diff_dest,diff_rates,
                                                mus,beta_ps,abx_gammas,dt,dt_mut,stochastic,
                                                diff_div_loss_locs, diff_div_gain_locs, mut):

    # allocate array for ddts
    ddts = np.zeros(len(concs))
    
    # total cell count
    ctot = concs[0:-2].sum()
    
    # fraction to modify growth rates and production rates
    MOD = (K-ctot)/K
    
    # growth and washout of cells, abx sensitive cells do not grow if [abx]>MIC
    ddts[0:-2] += mus*MOD*concs[0:-2]*(1-Heaviside_array(1-abx_gammas,0)*Heaviside(concs[-2]-MIC,1))*dt - D*concs[0:-2]*dt
    ddts[diff_div_gain_locs] -= 2*mus[diff_div_loss_locs]*\
                                (1-Heaviside_array(1-abx_gammas[diff_div_loss_locs],0)*Heaviside(concs[-2]-MIC,1))*\
                                MOD*concs[diff_div_loss_locs]*dt
    
    # antibiotic degradation and dilution
    ddts[-2] += -np.sum(concs[0:-2]*abx_gammas)*(Vmax/V)*concs[-2]/(concs[-2]+Km)*dt + D*(abx_in-concs[-2])*dt
    
    
    # production
    ddts[-1] = np.sum(concs[0:-2]*MOD*beta_ps)*dt
    
    # differentiation: deterministic
    ddts[0:-2] += multicassette_diff_fs_mut_determ(diff_source,diff_dest,diff_rates*dt_mut,concs[0:-2])
    
    if mut:
        if stochastic:
            ddts[0:-2] += multicassette_diff_fs_mut_stochastic(source,dest,np.abs(mus[source])*MOD*rates*dt_mut,concs[0:-2])
        else:
#             concs[0:-1] += multicassette_diff_fs_mut_determ(source,dest,rates*dt_mut,concs[0:-1])
            ddts[0:-2] += multicassette_diff_fs_mut_determ(source,dest,np.abs(mus[source])*MOD*rates*dt_mut,concs[0:-2])
    
    return concs + ddts

@numba.jit(nopython=True)
def multicassette_diff_select_fixed_step_growth(out,K,V,Vmax,Km,MIC,source,dest,rates,
                                                diff_source,diff_dest,diff_rates,mus,
                                                beta_ps,abx_gammas,dt,dt_mut,stochastic,
                                                frac_at_dilution,diff_div_loss_locs, diff_div_gain_locs,fixed_end=False):
    for i in range(out.shape[0]-1):
        if round(i*dt/dt_mut,6) == int(round(i*dt/dt_mut,6)):  
            i_1 = multicassette_diff_select_fixed_step_update(out[i,:],K,0,V,Vmax,Km,MIC,0,source,dest,rates,
                                                              diff_source,diff_dest,diff_rates,mus,
                                                              beta_ps,abx_gammas,dt,dt_mut,stochastic,
                                                              diff_div_loss_locs, diff_div_gain_locs,True)
            
        else:
            i_1 = multicassette_diff_select_fixed_step_update(out[i,:],K,0,V,Vmax,Km,MIC,0,source,dest,rates,
                                                              diff_source,diff_dest,diff_rates,mus,
                                                              beta_ps,abx_gammas,dt,dt_mut,stochastic,
                                                              diff_div_loss_locs, diff_div_gain_locs,True)
        i_1[i_1<0] = 0
        out[i+1,:] = i_1
    pop_tots = out[:,0:-2].sum(axis=1)
    if fixed_end == False:
        if pop_tots[-1] < (frac_at_dilution*K):
            return out
        else:
            end_loc = do_argwhere(pop_tots,K,frac_at_dilution)
            return out[0:end_loc,:]
    else:
        return out
        

# @numba.jit(nopython=True)
def multicassette_diff_select_fixed_step_batch(t, dt, dt_mut, cell_init, n_cassettes, n_int_cassettes, n_div, muN, muP,
                                               beta_p, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in, 
                                               selection='recessive',dilution_factor=50, 
                                               frac_at_dilution=0.95,prod_frac_thresh=1e-3,split_cassettes=False,
                                               stochastic=False, max_growths=None,plotting=False,fixed_end=False, 
                                               complete_growths=False, CIIE=False):
    
    if split_cassettes:
        source, dest, rates, diff_source,diff_dest,diff_rates, mus, beta_ps, abx_gammas, diff_div_loss_locs, \
        diff_div_gain_locs,ppmn_dict, ppmn_ind_dict, species = \
                diff_split_select_multicassette_mut_list_fixed_step(kdiff, kMB, kMD, kMI, kPL, muN, muP, beta_p,
                                           n_cassettes,n_int_cassettes, n_div, selection,CIIE=CIIE,return_gens=True)
                
    else: 
        source, dest, rates, diff_source,diff_dest,diff_rates, mus, beta_ps, abx_gammas, diff_div_loss_locs, \
        diff_div_gain_locs,ppmn_dict, ppmn_ind_dict, species = \
                diff_select_multicassette_mut_list_fixed_step(kdiff, kMB, kMD, kMI, kPL, muN, muP, beta_p,
                                           n_cassettes,n_int_cassettes, n_div, selection, CIIE=CIIE, return_gens=True)
    
    concs = np.zeros(len(mus)+2,'float64')
    concs[0] = cell_init
    production = 0
    dilute_again = True
    producer_locs = np.argwhere(beta_ps==beta_p).flatten()
    nonproducer_locs = np.argwhere(beta_ps==0).flatten()
    n=0
    if plotting:
        concs_endpoint = []
        times_endpoint = []
    while dilute_again:
        out = np.zeros((int(t/dt)+1,len(concs)))
        out[0,:]=concs
        out = multicassette_diff_select_fixed_step_growth(out,K,V,Vmax,Km,MIC,source,dest,rates,
                                                          diff_source,diff_dest,diff_rates,mus,
                                                          beta_ps,abx_gammas,dt,dt_mut,stochastic,
                                                          frac_at_dilution,diff_div_loss_locs, diff_div_gain_locs,fixed_end)
        end_concs = out[-1,:]
        if plotting:
            concs_endpoint.append(list(end_concs))
            times_endpoint.append(len(out)*dt - dt)
    
        if n == 0:
            out_concat = out
        else:
            out_concat = np.concatenate((out_concat,out))
        if type(max_growths) is int:
            if n+1 >= max_growths:
                dilute_again = False
        if complete_growths == False:
            if end_concs[0:-2].sum() < 2*dilution_factor:
                dilute_again = False
            elif end_concs[0:-2].sum() < 0.02*K:
                dilute_again = False
            elif (end_concs[producer_locs].sum()/end_concs[0:-2].sum()) < prod_frac_thresh:
                dilute_again = False
        if dilute_again:
            concs = end_concs
            if np.any(concs<0):
                concs[concs<0] = 0
            if stochastic:
                concs[0:-2] = np.random.binomial(list(concs[0:-2]),1/dilution_factor)
            else:
                concs[0:-2] /= dilution_factor
            # fresh antibiotic
            concs[-2] = concs[-2]/dilution_factor + (1-1/dilution_factor)*abx_in
        n +=1
    if plotting:
        return out_concat, np.array(concs_endpoint), np.array(times_endpoint), ppmn_dict, ppmn_ind_dict, n, species
    else:
        return out_concat, ppmn_dict, ppmn_ind_dict, n, species

"""
The functions below are for running multiprocessed deterministic and stochastic simulations of batch dilutions for the naive architecture, and calculating summary statistics to return
"""
                                       

                                       
"""pick up here"""
def run_naive_stoch_batch(muN=1, 
                          n_cassette_array=np.array([1,2,3]),
                          muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
                          kMBs=np.array([1e-6]),
                          kPL=1e-4,
                          Ks=np.array([1e4,1e5,1e6,1e7]),
                          V=1, #mL
                          Vmax = 1e6/6.022e23*422.36*1e6*3600, #(molecules/cell/s)*(1 mol/6.022e23 molec)*(422.36g/mol)*(1e6ug/g)*(3600s/h)
                          Km=6.7, #ug/mL
                          MIC=1.1, #ug/mL
                          abx_in=100, #ug/mL
                          selections=np.array(['recessive','additive']),
                          t_max = 'variable',
                          dt = 0.01,
                          summary_cols = np.array(['circuit','selection','muP_single_cassette','stochastic','n_cassettes',
                                                     'muN','muP','Kpop','kMB','kPL','Vmax','Km','MIC','abx','V','rep',
                                                     't_dilute','t_1M','t_half','t_99M','t_tot','n_growths','prod_integral',
                                                     'tot_integral','prod_frac_median','prod_frac_avg','total_production',
                                                     'production_rate','genotypes','final_pop']),
                          n_stoch=8,
                          max_growths=1000,
                          fstring=None,
                          plotting=False,
                          fixed_end=False,
                          complete_growths=False,
                          muP_single_cassette=False):
    

    circuit = 'naive'
    stochastic = True
    if plotting:
        summary_cols = np.array(['circuit','selection','muP_single_cassette','stochastic','n_cassettes',
                                 'muN','muP','Kpop','kMB','kPL','Vmax','Km','MIC','abx','V','rep',
                                 't_dilute','t_1M','t_half','t_99M','t_tot','n_growths','prod_integral',
                                 'tot_integral','prod_frac_median','prod_frac_avg','total_production',
                                 'production_rate','genotypes','pn_ind_dict','results',
                                 'production_array','time_array'])
    summary = []
    for n_cassettes in n_cassette_array:
        print(f'{n_cassettes} cassettes')
        args_list = []
        for muP in muPs:
            for K in Ks:
                V_act = V*K/1e9 # assume carrying capacity of 1e9/mL
                for selection in selections:
                    if t_max == 'variable':
                        args_list.append((muN, muP, n_cassettes, selection, kMBs, K,
                                          kPL,V_act,Vmax,Km,MIC,abx_in,
                                         12*(np.log(2)/muP), dt, summary_cols, n_stoch, 
                                          max_growths, plotting,fixed_end,complete_growths, muP_single_cassette))
                    else:
                        args_list.append((muN, muP, n_cassettes, selection, kMBs, K, 
                                          kPL,V_act,Vmax,Km,MIC,abx_in,
                                         t_max, dt, summary_cols, n_stoch, max_growths, 
                                          plotting,fixed_end,complete_growths, muP_single_cassette))
        Nprocessors = min(7,len(args_list))
        MPPool = MP.Pool(Nprocessors)
        results = MPPool.map(run_naive_stoch_MP_batch, args_list)
        i = 0
        for batch in results:
            if i == 0 and n_cassettes == n_cassette_array[0]:
                results_concat = batch
                i = 1
            else:
                for result in batch:
                    results_concat.append(result) 
        MPPool.terminate()
    return pd.DataFrame(columns=summary_cols,data=results_concat)

def run_naive_stoch_MP_batch(args):                             
    muN = args[0]
    muP = args[1]
    n_cassettes = args[2]
    selection = args[3]
    kMBs = args[4]
    K = args[5]
    kPL = args[6]                                   
    V = args[7]
    Vmax = args[8]
    Km = args[9]
    MIC = args[10]
    abx_in = args[11]                  
    t_max = args[12]
    dt = args[13]
    summary_cols = args[14]
    n_stoch = args[15]
    max_growths = args[16]
    plotting = args[17]
    fixed_end = args[18]
    complete_growths = args[19]
    muP_single_cassette = args[20]
    beta_p = 1
                                       
    dt_mut = dt
    cell_init = K/50
    
    stochastic = True
    circuit = 'naive'
    summary = []
    
    for kMB in kMBs:
        for i in range(n_stoch):
            if plotting:
                results, concs_endpoint, times_endpoint, pn_dict, pn_ind_dict, n_growths, species = \
                multicassette_naive_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, muN, muP,
                                               beta_p, kMB, kPL, K, V, Vmax, Km, MIC, abx_in, dilution_factor=50, 
                                               frac_at_dilution=0.95,prod_frac_thresh=1e-3,
                                               stochastic=True,max_growths=max_growths,selection=selection,
                                               plotting=plotting,fixed_end=fixed_end,complete_growths=complete_growths,
                                               muP_single_cassette=muP_single_cassette)
                results_summary = naive_summary_plotting(results,concs_endpoint, times_endpoint,pn_ind_dict, n_growths, muN, 
                                                         muP, selection, n_cassettes, kMB, kPL, K, V, Vmax, Km, MIC, abx_in, 
                                                         dt, stochastic, i+1, 
                                                         summary_cols, species,muP_single_cassette)
            else:
                results, pn_dict, pn_ind_dict, n_growths, species = \
                    multicassette_naive_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, muN, muP,
                                               beta_p, kMB, kPL, K, V, Vmax, Km, MIC, abx_in, dilution_factor=50, 
                                               frac_at_dilution=0.95,prod_frac_thresh=1e-3,
                                               stochastic=True,max_growths=max_growths,selection=selection,
                                               plotting=plotting,fixed_end=fixed_end,complete_growths=complete_growths,
                                               muP_single_cassette=muP_single_cassette)
                results_summary = naive_summary(results,pn_ind_dict, n_growths, muN, muP, selection, n_cassettes, 
                                                kMB, kPL, K, V, Vmax, Km, MIC, abx_in,
                                                dt, stochastic, i+1, summary_cols, species,muP_single_cassette)

            summary.append(results_summary)
    return summary

def run_naive_det_batch(muN=1, 
                          n_cassette_array=np.array([1,2,3]),
                          muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
                          kMBs=np.array([1e-6]),
                          kPL=1e-4,
                          Ks=np.array([1e4,1e5,1e6,1e7]),
                          V=1, #mL
                          Vmax = 1e6/6.022e23*422.36*1e6*3600, #(molecules/cell/s)*(1 mol/6.022e23 molec)*(422.36g/mol)*(1e6ug/g)*(3600s/h)
                          Km=6.7, #ug/mL
                          MIC=1.1, #ug/mL
                          abx_in=100, #ug/mL
                          selections=np.array(['recessive','additive']),
                          t_max = 'variable',
                          dt = 0.01,
                          summary_cols = np.array(['circuit','selection','muP_single_cassette','stochastic','n_cassettes',
                                                     'muN','muP','Kpop','kMB','kPL','Vmax','Km','MIC','abx','V','rep',
                                                     't_dilute','t_1M','t_half','t_99M','t_tot','n_growths','prod_integral',
                                                     'tot_integral','prod_frac_median','prod_frac_avg','total_production',
                                                     'production_rate','genotypes','final_pop']),
                          max_growths=1000,
                          plotting=False,
                          fixed_end=False,
                          complete_growths=False,
                          muP_single_cassette=False):
    
    circuit = 'naive'
    stochastic = False
    summary = []
    if plotting:
        summary_cols = np.array(['circuit','selection','muP_single_cassette','stochastic','n_cassettes',
                                 'muN','muP','Kpop','kMB','kPL','Vmax','Km','MIC','abx','V','rep',
                                 't_dilute','t_1M','t_half','t_99M','t_tot','n_growths','prod_integral',
                                 'tot_integral','prod_frac_median','prod_frac_avg','total_production',
                                 'production_rate','genotypes','pn_ind_dict','results',
                                 'production_array','time_array'])
    for n_cassettes in n_cassette_array:
        print(f'{n_cassettes} cassettes')
        args_list = []
        for muP in muPs:
            for K in Ks:
                V_act = V*K/1e9 # assume carrying capacity of 1e9/mL
                for selection in selections:
                    if t_max == 'variable':
                        args_list.append((muN, muP, n_cassettes, selection, kMBs, K,
                                          kPL,V_act,Vmax,Km,MIC,abx_in,
                                         12*(np.log(2)/muP), dt, summary_cols, 
                                          max_growths, plotting,fixed_end,complete_growths,muP_single_cassette))
                    else:
                        args_list.append((muN, muP, n_cassettes, selection, kMBs, K, 
                                          kPL,V_act,Vmax,Km,MIC,abx_in,
                                         t_max, dt, summary_cols, max_growths, 
                                          plotting,fixed_end,complete_growths,muP_single_cassette))
                    
        Nprocessors = min(7,len(args_list))
        MPPool = MP.Pool(Nprocessors)
        results = MPPool.map(run_naive_det_MP_batch, args_list)
        i = 0
        for batch in results:
            if i == 0 and n_cassettes == n_cassette_array[0]:
                results_concat = batch
                i = 1
            else:
                for result in batch:
                    results_concat.append(result) 
        MPPool.terminate()
    return pd.DataFrame(columns=summary_cols,data=results_concat)

def run_naive_det_MP_batch(args):
    muN = args[0]
    muP = args[1]
    n_cassettes = args[2]
    selection = args[3]
    kMBs = args[4]
    K = args[5]
    kPL = args[6]                                   
    V = args[7]
    Vmax = args[8]
    Km = args[9]
    MIC = args[10]
    abx_in = args[11]                  
    t_max = args[12]
    dt = args[13]
    summary_cols = args[14]
    max_growths = args[15]
    plotting = args[16]
    fixed_end = args[17]
    complete_growths = args[18]
    muP_single_cassette = args[19]
    
    dt_mut = dt
    cell_init = K/50
    beta_p = 1
    
    stochastic = False
    circuit = 'naive'
    summary = []
    for kMB in kMBs:
        if plotting:
            results, concs_endpoint, times_endpoint, pn_dict, pn_ind_dict, n_growths, species = \
            multicassette_naive_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, muN, muP,
                                           beta_p, kMB, kPL, K, V, Vmax, Km, MIC, abx_in, dilution_factor=50, 
                                           frac_at_dilution=0.95,prod_frac_thresh=1e-3,
                                           stochastic=True,max_growths=max_growths,selection=selection,
                                           plotting=plotting,fixed_end=fixed_end,complete_growths=complete_growths,
                                           muP_single_cassette=muP_single_cassette)
            results_summary = naive_summary_plotting(results,concs_endpoint, times_endpoint,pn_ind_dict, n_growths, muN, 
                                                     muP, selection, n_cassettes, kMB, kPL, K, V, Vmax, Km, MIC, abx_in, 
                                                     dt, stochastic, 0, 
                                                     summary_cols, species,muP_single_cassette)
        else:
            results, pn_dict, pn_ind_dict, n_growths, species = \
                multicassette_naive_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, muN, muP,
                                           beta_p, kMB, kPL, K, V, Vmax, Km, MIC, abx_in, dilution_factor=50, 
                                           frac_at_dilution=0.95,prod_frac_thresh=1e-3,
                                           stochastic=True,max_growths=max_growths,selection=selection,
                                           plotting=plotting,fixed_end=fixed_end,complete_growths=complete_growths,
                                           muP_single_cassette=muP_single_cassette)
            results_summary = naive_summary(results,t_max, pn_ind_dict, n_growths, muN, muP, selection, n_cassettes, 
                                            kMB, kPL, K, V, Vmax, Km, MIC, abx_in,
                                            dt, stochastic, 0, summary_cols, species,muP_single_cassette)

        summary.append(results_summary)
    return summary

def naive_summary(results, t_max, species_ind_dict, n_growths, muN, muP, selection, n_cassettes, 
                  kMB, kPL, K, V, Vmax, Km, MIC, abx_in, dt, stochastic, rep, summary_cols, species,muP_single_cassette): 
    

    summary_dict = {}
    summary_dict['circuit'] = 'naive'
    summary_dict['selection'] = selection
    summary_dict['muP_single_cassette'] = muP_single_cassette
    summary_dict['stochastic'] = stochastic
    summary_dict['muN'] = muN
    summary_dict['muP'] = muP
    summary_dict['t_dilute'] = t_max
    summary_dict['n_cassettes'] = n_cassettes
    summary_dict['kMB'] = kMB
    summary_dict['kPL'] = kPL
    summary_dict['Kpop'] = K
    summary_dict['V'] = V
    summary_dict['Vmax'] = Vmax
    summary_dict['Km'] = Km
    summary_dict['MIC'] = MIC
    summary_dict['abx'] = abx_in   
    summary_dict['rep'] = rep
    summary_dict['n_growths'] = n_growths
    summary_dict['genotypes'] = species
    summary_dict['results'] = results
    summary_dict['time'] = np.arange(1,n_growths+1)*t_max
    summary_dict['production'] = results[:,-1]
    
    pop_tots = {}
    for label in species_ind_dict:
        pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()
    pop_tots['total'] = results[:,0:-1].sum(axis=1).flatten()
    
    # time at first dilution
    summary_dict['t_dilute'] = scipy.signal.find_peaks(pop_tots['total'])[0][0]*dt
    
    # check if washout happened
#     if np.any(pop_tots['total'] < summary_dict['Kpop']*0.01):
#         t_washout_ind = np.argwhere(pop_tots['total']<summary_dict['Kpop']*0.01).min()
#         summary_dict['t_washout'] = timepoints[t_washout_ind]
# #         if summary_dict['t_end'] == -1:
# #             summary_dict['t_end'] = summary_dict['t_washout']
#         summary_dict['washout'] = True
#     else:
#         t_washout_ind = -1
#         summary_dict['t_washout'] = np.inf
#         summary_dict['washout'] = False
        
#         summary.append(M.get_param_value(param))
    
    # time till one mutant
    if np.any(pop_tots['mutants']>=1):
        summary_dict['t_1M'] = np.argwhere(pop_tots['mutants']>=1).min()*dt
    else:
        summary_dict['t_1M'] = -1
        
    # time till half mutants
    mutant_frac = pop_tots['mutants'] / pop_tots['total']
    if np.any(mutant_frac>0.5):
        summary_dict['t_half'] = np.argwhere(mutant_frac>0.5).min()*dt
    else:
        summary_dict['t_half'] = -1
    
    # time till 99% mutants
    if np.any(mutant_frac>=0.99):
        summary_dict['t_99M'] = np.argwhere(mutant_frac>=0.99).min()*dt
        summary_dict['t_tot'] = np.argwhere(mutant_frac>=0.99).min()*dt
    else:
        t_99M_ind = -1
        summary_dict['t_99M'] = -1
    t_end_ind = -1    
    
    if summary_dict['t_99M'] == -1:
        summary_dict['t_tot'] = len(pop_tots['mutants'])*dt
        t_end_ind = -1   
    else:
        summary_dict['t_tot'] = summary_dict['t_99M']
        t_end_ind = np.argwhere(mutant_frac>=0.99).min()
    
#     if t_99M_ind != -1 and t_washout_ind != -1:
#         t_end_ind = min(t_99M_ind, t_washout_ind)
#     elif t_99M_ind != -1:
#         t_end_ind = t_99M_ind
#     elif t_washout_ind != -1:
#         t_end_ind = t_washout_ind
#     summary_dict['t_tot'] = timepoints[t_end_ind]
    
    # final population state
    summary_dict['final_pop'] = results[t_end_ind,0:-1]
    # integral of producers
    summary_dict['prod_integral'] = dt*np.sum(pop_tots['producers'][0:t_end_ind])
    # total integral
    summary_dict['tot_integral'] = dt*np.sum(pop_tots['total'][0:t_end_ind])
    # median fraction of producers (should be ~the steady state)
    summary_dict['prod_frac_median'] = np.median(pop_tots['producers'][0:t_end_ind]/pop_tots['total'][0:t_end_ind])
    # average fraction of producers
    summary_dict['prod_frac_avg'] = summary_dict['prod_integral']/summary_dict['tot_integral']
    summary_dict['tot_pop_avg'] = np.mean(pop_tots['total'][0:t_end_ind])
    summary_dict['prod_pop_avg'] = np.mean(pop_tots['producers'][0:t_end_ind])
    summary_dict['total_production'] = results[t_end_ind,-1]
    summary_dict['production_rate'] = summary_dict['total_production']/summary_dict['t_tot']
    
    summary = []
    for col in summary_cols:
        summary.append(summary_dict[col])
        
    return summary


def naive_summary_plotting(results, concs_endpoint, times_endpoint, species_ind_dict, n_growths, muN, muP, selection, 
                           n_cassettes, kMB, kPL, K, V, Vmax, Km, MIC, abx_in, dt, stochastic, rep, summary_cols, species,
                           muP_single_cassette): 
    
    summary_dict = {}
    summary_dict['circuit'] = 'naive'
    summary_dict['selection'] = selection
    summary_dict['muP_single_cassette'] = muP_single_cassette
    summary_dict['stochastic'] = stochastic
    summary_dict['muN'] = muN
    summary_dict['muP'] = muP
    summary_dict['n_cassettes'] = n_cassettes
    summary_dict['kMB'] = kMB
    summary_dict['kPL'] = kPL
    summary_dict['Kpop'] = K
    summary_dict['V'] = V
    summary_dict['Vmax'] = Vmax
    summary_dict['Km'] = Km
    summary_dict['MIC'] = MIC    
    summary_dict['abx'] = abx_in  
    summary_dict['rep'] = rep
    summary_dict['n_growths'] = n_growths
    summary_dict['genotypes'] = species
    
    
    pop_tots = {}
    for label in species_ind_dict:
        pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()
    pop_tots['total'] = results[:,0:-1].sum(axis=1).flatten()
    
    # time at first dilution
    summary_dict['t_dilute'] = times_endpoint[0]
    
    # time till one mutant
    if np.any(pop_tots['mutants']>=1):
        summary_dict['t_1M'] = np.argwhere(pop_tots['mutants']>=1).min()*dt
    else:
        summary_dict['t_1M'] = -1
        
    # time till half mutants
    mutant_frac = pop_tots['mutants'] / pop_tots['total']
    if np.any(mutant_frac>0.5):
        summary_dict['t_half'] = np.argwhere(mutant_frac>0.5).min()*dt
    else:
        summary_dict['t_half'] = -1
    
    # time till 99% mutants
    if np.any(mutant_frac>=0.99):
        summary_dict['t_99M'] = np.argwhere(mutant_frac>=0.99).min()*dt
        summary_dict['t_tot'] = np.argwhere(mutant_frac>=0.99).min()*dt
    else:
        t_99M_ind = -1
        summary_dict['t_99M'] = -1
    t_end_ind = -1    
    
    if summary_dict['t_99M'] == -1:
        summary_dict['t_tot'] = len(pop_tots['mutants'])*dt
        t_end_ind = -1   
    else:
        summary_dict['t_tot'] = summary_dict['t_99M']
        t_end_ind = np.argwhere(mutant_frac>=0.99).min()
    
#     if t_99M_ind != -1 and t_washout_ind != -1:
#         t_end_ind = min(t_99M_ind, t_washout_ind)
#     elif t_99M_ind != -1:
#         t_end_ind = t_99M_ind
#     elif t_washout_ind != -1:
#         t_end_ind = t_washout_ind
#     summary_dict['t_tot'] = timepoints[t_end_ind]
    
    # final population state
    summary_dict['final_pop'] = results[t_end_ind,0:-1]
    # integral of producers
    summary_dict['prod_integral'] = dt*np.sum(pop_tots['producers'][0:t_end_ind])
    # total integral
    summary_dict['tot_integral'] = dt*np.sum(pop_tots['total'][0:t_end_ind])
    # median fraction of producers (should be ~the steady state)
    summary_dict['prod_frac_median'] = np.median(pop_tots['producers'][0:t_end_ind]/pop_tots['total'][0:t_end_ind])
    # average fraction of producers
    summary_dict['prod_frac_avg'] = summary_dict['prod_integral']/summary_dict['tot_integral']
    summary_dict['tot_pop_avg'] = np.mean(pop_tots['total'][0:t_end_ind])
    summary_dict['prod_pop_avg'] = np.mean(pop_tots['producers'][0:t_end_ind])
    summary_dict['total_production'] = results[t_end_ind,-1]
    summary_dict['production_rate'] = summary_dict['total_production']/summary_dict['t_tot']
    summary_dict['pn_ind_dict'] = species_ind_dict
    summary_dict['results'] = concs_endpoint
    summary_dict['production_array'] = concs_endpoint[:,-1]
    summary_dict['time_array'] = times_endpoint
    
    summary = []
    for col in summary_cols:
        summary.append(summary_dict[col])
        
    return summary

"""
The functions below are for running multiprocessed deterministic and stochastic simulations of batch dilutions for the differentiation architecture, and calculating summary statistics to return
"""
def run_diff_det_batch(muN=1, 
                       n_cassette_array=np.array([1,2,3]),
                       cassettes_equal=False,
                       n_int_cassette_array=np.array([1,2,3]),
                       muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
                       kdiffs = np.array([0.05,0.1,0.2]),
                       kMBs=np.array([1e-6]),
                       kMDs=np.array([1e-6]),
                       kMIs=np.array([1e-6]),
                       kPL=1e-4,
                       Ks=np.array([1e9,1e10,1e11]),
                       V=1, #mL
                       Vmax = 1e6/6.022e23*422.36*1e6*3600, #(molecules/cell/s)*(1 mol/6.022e23 molec)*(422.36g/mol)*(1e6ug/g)*(3600s/h)
                       Km=6.7, #ug/mL
                       MIC=1.1, #ug/mL
                       abx_in=100, #ug/mL
                       selection='recessive',
                       t_max = 12,
                       dt = 0.01,
                       summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection','CIIE',
                                                'n_div',
                                                'muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                                'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                                'n_growths','prod_integral','tot_integral','prod_frac_median',
                                                'prod_frac_avg','total_production','production_rate',
                                                'genotypes','final_pop']),
                       max_growths=1000,
                       plotting=False,
                       fixed_end=False,
                       complete_growths=False,
                       CIIE=False):
    
    circuit = 'diff'
    stochastic = False
    summary = []
    i = 0
    if plotting:
        summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection','CIIE','n_div',
                                 'muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                 'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                 'n_growths','prod_integral','tot_integral','prod_frac_median',
                                 'prod_frac_avg','total_production','production_rate','genotypes',
                                 'species_ind_dict','results','production_array','time_array'])
    for n_cassettes in n_cassette_array:
        print(f'{n_cassettes} cassettes')
        if cassettes_equal:
            n_int_cassette_array = np.array([n_cassettes])
        for n_int_cassettes in n_int_cassette_array:
            args_list = []
            for muP in muPs:
                for K in Ks:
                    V_act = V*K/1e9 # assume carrying capacity of 1e9/mL
                    for kdiff in kdiffs:
                        if t_max == 'variable':
                            args_list.append((muN, muP, n_cassettes, n_int_cassettes, kdiff, kMBs, kMDs, kMIs, K,
                                              kPL,V_act,Vmax,Km,MIC,abx_in,selection,
                                          12*(np.log(2)/muP), dt, summary_cols,max_growths,plotting,fixed_end,
                                             complete_growths,CIIE))
                        else:
                            args_list.append((muN, muP, n_cassettes, n_int_cassettes, kdiff, kMBs, kMDs, kMIs, K,
                                              kPL,V_act,Vmax,Km,MIC,abx_in,selection,
                                              t_max, dt, summary_cols,max_growths,plotting,fixed_end,complete_growths,
                                              CIIE))

            Nprocessors = min(7,len(args_list))
            MPPool = MP.Pool(Nprocessors)
            results = MPPool.map(run_diff_det_MP_batch, args_list)
            
            for batch in results:
                if i == 0:
                    results_concat = batch
                    i = 1
                else:
                    for result in batch:
                        results_concat.append(result) 
            MPPool.terminate()
    return pd.DataFrame(columns=summary_cols,data=results_concat)

def run_diff_det_MP_batch(args):
    muN = args[0]
    muP = args[1]
    n_cassettes = args[2]
    n_int_cassettes = args[3]
    kdiff = args[4]
    kMBs = args[5]
    kMDs = args[6]
    kMIs = args[7]
    K = args[8]
    kPL = args[9]
    V = args[10]
    Vmax = args[11]
    Km = args[12]
    MIC = args[13]
    abx_in = args[14]
    selection = args[15]
    t_max = args[16]
    dt = args[17]
    summary_cols = args[18]
    max_growths = args[19]
    plotting = args[20]
    fixed_end = args[21]
    complete_growths = args[22]
    CIIE = args[23]
    
    
    beta_p = 1
    cell_init = K/50
    dt_mut = dt
    stochastic = False
    circuit = 'diff'
    summary = []
    for kMB in kMBs:
        for kMD in kMDs:
            for kMI in kMIs:
                if plotting:
                    results, concs_endpoint, times_endpoint, ppm_dict, ppm_ind_dict, n_growths, species = \
                        multicassette_diff_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, n_int_cassettes, muN, 
                                                            muP, beta_p, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, 
                                                            abx_in,selection,dilution_factor=50,frac_at_dilution=0.95,
                                                            prod_frac_thresh=1e-3, stochastic=False,
                                                            max_growths=max_growths, plotting=plotting, fixed_end=fixed_end,
                                                            complete_growths=complete_growths,
                                                            CIIE=CIIE) #checked
                    

                    results_summary = diff_summary_plotting(results, concs_endpoint, times_endpoint, circuit, ppm_ind_dict, 
                                                            n_growths, muN, muP, n_cassettes, n_int_cassettes, 0, kdiff, kMB, 
                                                            kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in, selection, 
                                                            dt, stochastic, 0, summary_cols, species,CIIE) #checked
                    
                    
                else:
                    results, ppm_dict, ppm_ind_dict, n_growths, species = \
                        multicassette_diff_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, n_int_cassettes, muN, 
                                                            muP, beta_p, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, 
                                                            abx_in,selection,dilution_factor=50,frac_at_dilution=0.95,
                                                            prod_frac_thresh=1e-3, stochastic=False,
                                                            max_growths=max_growths, plotting=plotting, fixed_end=fixed_end,
                                                            complete_growths=complete_growths,
                                                            CIIE=CIIE) # checked
                    

                    results_summary = diff_summary(results, circuit, ppm_ind_dict, n_growths, muN, muP, n_cassettes, 
                                                   n_int_cassettes, 0, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, 
                                                   abx_in, selection, dt, stochastic, 0, summary_cols, species, CIIE) # checked
                    
                summary.append(results_summary)
    return summary

def run_diff_stoch_batch(muN=1, 
                         n_cassette_array=np.array([1,2,3]),
                         cassettes_equal=False,
                         n_int_cassette_array=np.array([1,2,3]),
                         muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
                         kdiffs = np.array([0.05,0.1,0.2]),
                         kMBs=np.array([1e-6]),
                         kMDs=np.array([1e-6]),
                         kMIs=np.array([1e-6]),
                         kPL=1e-4,
                         Ks=np.array([1e9,1e10,1e11]),
                         V=1, #mL
                         Vmax = 1e6/6.022e23*422.36*1e6*3600, #(molecules/cell/s)*(1 mol/6.022e23 molec)*(422.36g/mol)*(1e6ug/g)*(3600s/h)
                         Km=6.7, #ug/mL
                         MIC=1.1, #ug/mL
                         abx_in=100, #ug/mL
                         selection='recessive',
                         t_max = 12,
                         dt = 0.01,
                         summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection','CIIE',
                                                  'n_div',
                                                  'muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                                  'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                                  'n_growths','prod_integral','tot_integral','prod_frac_median',
                                                  'prod_frac_avg','total_production','production_rate',
                                                  'genotypes','final_pop']),
                         n_stoch=8,
                         max_growths=1000,
                         save_csvs = True,
                         fstring=None,
                         plotting=False,
                         fixed_end=False,
                         complete_growths=False,
                         CIIE=False):
    
    circuit = 'diff'
    stochastic = True
    if plotting:
        summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection','CIIE','n_div',
                                 'muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                 'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                 'n_growths','prod_integral','tot_integral','prod_frac_median',
                                 'prod_frac_avg','total_production','production_rate','genotypes',
                                 'species_ind_dict','results','production_array','time_array'])
    results_concat = []
      
    for n_cassettes in n_cassette_array:
        print(f'{n_cassettes} cassettes')
        if cassettes_equal:
            n_int_cassette_array = np.array([n_cassettes])
        for n_int_cassettes in n_int_cassette_array:
            for K in Ks:
                V_act = V*K/1e9 # assume carrying capacity of 1e9/mL
                print(f'{K} K')
                args_list = []
                for muP in muPs:
                    for kdiff in kdiffs:
                        for kMB in kMBs:
                            for kMD in kMDs:
                                for kMI in kMIs:
                                    if t_max == 'variable':
                                        args_list.append((muN, muP, n_cassettes, n_int_cassettes, kdiff, kMB, kMD, kMI, K,
                                                          kPL,V_act,Vmax,Km,MIC,abx_in,selection,
                                                      12*(np.log(2)/muP), dt, summary_cols, n_stoch, max_growths,
                                                         plotting,fixed_end,complete_growths,CIIE))
                                    else:
                                        args_list.append((muN, muP, n_cassettes, n_int_cassettes, kdiff, kMB, kMD, kMI, K,
                                                          kPL,V_act,Vmax,Km,MIC,abx_in,selection,
                                                      t_max, dt, summary_cols, n_stoch, max_growths,
                                                         plotting,fixed_end,complete_growths,CIIE)) # checked
                
                Nprocessors = min(7,len(args_list))
                MPPool = MP.Pool(Nprocessors)
                results = MPPool.map(run_diff_stoch_MP_batch, args_list)

    #             i = 0

                results_current = []
                for batch in results:
                    for result in batch:
                        results_current += [result]
                        results_concat += [result]
    #                 if i == 0 and n_cassettes == n_cassette_array[0]:
    #                     results_concat = batch
    #                     i = 1
    #                 else:
    #                     for result in batch:
    #                         results_concat.append(result) 
                MPPool.terminate()
                if save_csvs is True:
                    df_results_current = pd.DataFrame(columns=summary_cols,data=results_current)
                    year = str(time.localtime().tm_year)
                    month = str(time.localtime().tm_mon)
                    if len(month) == 1: month = '0' + month
                    day = str(time.localtime().tm_mday)
                    if len(day) == 1: day = '0' + day
                    date = year + month + day
                    if fstring is None:
                        df_results_current.to_csv(f'{date}_{circuit}_{n_cassettes}cass_{n_int_cassettes}intcass_{int(K)}K_batch_{t_max}.csv')
                    else:
                        df_results_current.to_csv(f'{fstring}{date}_{circuit}_{n_cassettes}cass_{n_int_cassettes}intcass_{K}K_batch_{t_max}.csv')
                    
            
    return pd.DataFrame(columns=summary_cols,data=results_concat)

def run_diff_stoch_MP_batch(args):
    muN = args[0]
    muP = args[1]
    n_cassettes = args[2]
    n_int_cassettes = args[3]
    kdiff = args[4]
    kMB = args[5]
    kMD = args[6]
    kMI = args[7]    
    K = args[8]
    kPL = args[9]
    V = args[10]
    Vmax = args[11]
    Km = args[12]
    MIC = args[13]
    abx_in = args[14]
    selection = args[15]
    t_max = args[16]
    dt = args[17]
    summary_cols = args[18]
    n_stoch = args[19]
    max_growths = args[20]
    plotting = args[21]
    fixed_end = args[22]
    complete_growths = args[23]
    CIIE = args[24]
    
    
    beta_p = 1
    cell_init = K/50
    dt_mut = dt
    stochastic = True
    circuit = 'diff'
    summary = []
#     for kMB in kMBs:
#         for kMD in kMDs:
    for i in range(n_stoch):
        if plotting:
            results, concs_endpoint, times_endpoint, ppm_dict, ppm_ind_dict, n_growths, species = \
                results, concs_endpoint, times_endpoint, ppm_dict, ppm_ind_dict, n_growths, species = \
                        multicassette_diff_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, n_int_cassettes, muN, 
                                                            muP, beta_p, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, 
                                                            abx_in,selection,dilution_factor=50,frac_at_dilution=0.95,
                                                            prod_frac_thresh=1e-3, stochastic=stochastic,
                                                            max_growths=max_growths, plotting=plotting, fixed_end=fixed_end,
                                                            complete_growths=complete_growths,CIIE=CIIE) #checked

            results_summary = diff_summary_plotting(results, concs_endpoint, times_endpoint, circuit, ppm_ind_dict, 
                                                    n_growths, muN, muP, n_cassettes, n_int_cassettes, 0, kdiff, kMB, 
                                                    kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in, selection,dt, stochastic, 
                                                    i+1,summary_cols, species,CIIE) #checked
            

        else:
            results, ppm_dict, ppm_ind_dict, n_growths, species = \
                        multicassette_diff_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, n_int_cassettes, muN, 
                                                            muP, beta_p, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, 
                                                            abx_in,selection,dilution_factor=50,frac_at_dilution=0.95,
                                                            prod_frac_thresh=1e-3, stochastic=stochastic,
                                                            max_growths=max_growths, plotting=plotting, fixed_end=fixed_end,
                                                            complete_growths=complete_growths,CIIE=CIIE) # checked

            results_summary = diff_summary(results, circuit, ppm_ind_dict, n_growths, muN, muP, n_cassettes, 
                                           n_int_cassettes, 0, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in,
                                           selection, dt, stochastic, i+1, summary_cols, species,CIIE) # checked
            
                              

        summary.append(results_summary)
#     print(summary)
    return summary

def diff_summary_plotting(results, concs_endpoint, times_endpoint, circuit, species_ind_dict, n_growths, muN, muP, 
                          n_cassettes, n_int_cassettes, n_div, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in,
                          selection, dt, stochastic, rep, summary_cols, species,CIIE): 
    
    pop_tots = {}
    summary_dict = {}
    summary_dict['circuit'] = circuit
    summary_dict['stochastic'] = stochastic
    summary_dict['muN'] = muN
    summary_dict['muP'] = muP
    summary_dict['n_cassettes'] = n_cassettes
    summary_dict['n_int_cassettes'] = n_int_cassettes
    summary_dict['selection'] = selection
    summary_dict['CIIE'] = CIIE    
    summary_dict['n_div'] = n_div
    summary_dict['kdiff'] = kdiff
    summary_dict['kMB'] = kMB
    summary_dict['kMD'] = kMD
    summary_dict['kMI'] = kMI
    summary_dict['Kpop'] = K
    summary_dict['kPL'] = kPL
    summary_dict['Kpop'] = K
    summary_dict['V'] = V
    summary_dict['Vmax'] = Vmax
    summary_dict['Km'] = Km
    summary_dict['MIC'] = MIC    
    summary_dict['abx'] = abx_in  
    summary_dict['rep'] = rep
    summary_dict['n_growths'] = n_growths
    summary_dict['genotypes'] = species
        
    pop_tots = {}
    for label in species_ind_dict:
#         pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()[0:slice_ind]
        pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()
    pop_tots['total'] = results[:,0:-2].sum(axis=1).flatten()
    
    # time at first dilution
    summary_dict['t_dilute'] = times_endpoint[0]
    
    
    # check if washout happened
#     if np.any(pop_tots['total'] < summary_dict['Kpop']*0.01):
#         t_washout_ind = np.argwhere(pop_tots['total']<summary_dict['Kpop']*0.01).min()
#         summary_dict['t_washout'] = timepoints[t_washout_ind]
# #         if summary_dict['t_end'] == -1:
# #             summary_dict['t_end'] = summary_dict['t_washout']
#         summary_dict['washout'] = True
#     else:
#         t_washout_ind = -1
#         summary_dict['t_washout'] = np.inf
#         summary_dict['washout'] = False
        
#         summary.append(M.get_param_value(param))
    
    # time till one mutant
    if np.any(pop_tots['mutants']>=1):
        summary_dict['t_1M'] = np.argwhere(pop_tots['mutants']>=1).min()*dt
    else:
        summary_dict['t_1M'] = -1
        
    # time till half mutants
    mutant_frac = pop_tots['mutants'] / pop_tots['total']
    if np.any(mutant_frac>0.5):
        summary_dict['t_half'] = np.argwhere(mutant_frac>0.5).min()*dt
    else:
        summary_dict['t_half'] = -1
    
    # time till 99% mutants
    if np.any(mutant_frac>=0.99):
        summary_dict['t_99M'] = np.argwhere(mutant_frac>=0.99).min()*dt
        summary_dict['t_tot'] = np.argwhere(mutant_frac>=0.99).min()*dt
    else:
        t_99M_ind = -1
        summary_dict['t_99M'] = -1
    t_end_ind = -1    
    
    if summary_dict['t_99M'] == -1:
        summary_dict['t_tot'] = len(pop_tots['mutants'])*dt
        t_end_ind = -1   
    else:
        summary_dict['t_tot'] = summary_dict['t_99M']
        t_end_ind = np.argwhere(mutant_frac>=0.99).min()
    
#     if t_99M_ind != -1 and t_washout_ind != -1:
#         t_end_ind = min(t_99M_ind, t_washout_ind)
#     elif t_99M_ind != -1:
#         t_end_ind = t_99M_ind
#     elif t_washout_ind != -1:
#         t_end_ind = t_washout_ind
#     summary_dict['t_tot'] = timepoints[t_end_ind]
    
    # final population state
    summary_dict['final_pop'] = results[t_end_ind,0:-2]
    # integral of producers
    summary_dict['prod_integral'] = dt*np.sum(pop_tots['producers'][0:t_end_ind])
    # total integral
    summary_dict['tot_integral'] = dt*np.sum(pop_tots['total'][0:t_end_ind])
    # median fraction of producers (should be ~the steady state)
    summary_dict['prod_frac_median'] = np.median(pop_tots['producers'][0:t_end_ind]/pop_tots['total'][0:t_end_ind])
    # average fraction of producers
    summary_dict['prod_frac_avg'] = summary_dict['prod_integral']/summary_dict['tot_integral']
    summary_dict['tot_pop_avg'] = np.mean(pop_tots['total'][0:t_end_ind])
    summary_dict['prod_pop_avg'] = np.mean(pop_tots['producers'][0:t_end_ind])
    summary_dict['total_production'] = results[t_end_ind,-1]
    summary_dict['production_rate'] = summary_dict['total_production']/summary_dict['t_tot']
    summary_dict['species_ind_dict'] = species_ind_dict
    summary_dict['results'] = concs_endpoint
    summary_dict['production_array'] = concs_endpoint[:,-1]
    summary_dict['time_array'] = times_endpoint
    
    summary = []
    for col in summary_cols:
        summary.append(summary_dict[col])
        
    return summary

def diff_summary(results, circuit, species_ind_dict, n_growths, muN, muP, n_cassettes, n_int_cassettes, n_div, kdiff, kMB, 
                 kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in, selection, dt, stochastic, rep, summary_cols, species,CIIE): 
    
    pop_tots = {}
    summary_dict = {}
    summary_dict['circuit'] = circuit
    summary_dict['stochastic'] = stochastic
    summary_dict['muN'] = muN
    summary_dict['muP'] = muP
    summary_dict['n_cassettes'] = n_cassettes
    summary_dict['n_int_cassettes'] = n_int_cassettes
    summary_dict['selection'] = selection
    summary_dict['CIIE'] = CIIE     
    summary_dict['n_div'] = n_div
    summary_dict['kdiff'] = kdiff
    summary_dict['kMB'] = kMB
    summary_dict['kMD'] = kMD
    summary_dict['kMI'] = kMI
    summary_dict['Kpop'] = K
    summary_dict['kPL'] = kPL
    summary_dict['Kpop'] = K
    summary_dict['V'] = V
    summary_dict['Vmax'] = Vmax
    summary_dict['Km'] = Km
    summary_dict['MIC'] = MIC
    summary_dict['abx'] = abx_in
    summary_dict['rep'] = rep
    summary_dict['n_growths'] = n_growths
    summary_dict['genotypes'] = species
        
    pop_tots = {}
    for label in species_ind_dict:
#         pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()[0:slice_ind]
        pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()
    pop_tots['total'] = results[:,0:-2].sum(axis=1).flatten()
    
    # time at first dilution
    summary_dict['t_dilute'] = scipy.signal.find_peaks(pop_tots['total'])[0][0]*dt
    
    # check if washout happened
#     if np.any(pop_tots['total'] < summary_dict['Kpop']*0.01):
#         t_washout_ind = np.argwhere(pop_tots['total']<summary_dict['Kpop']*0.01).min()
#         summary_dict['t_washout'] = timepoints[t_washout_ind]
# #         if summary_dict['t_end'] == -1:
# #             summary_dict['t_end'] = summary_dict['t_washout']
#         summary_dict['washout'] = True
#     else:
#         t_washout_ind = -1
#         summary_dict['t_washout'] = np.inf
#         summary_dict['washout'] = False
        
#         summary.append(M.get_param_value(param))
    
    # time till one mutant
    if np.any(pop_tots['mutants']>=1):
        summary_dict['t_1M'] = np.argwhere(pop_tots['mutants']>=1).min()*dt
    else:
        summary_dict['t_1M'] = -1
        
    # time till half mutants
    mutant_frac = pop_tots['mutants'] / pop_tots['total']
    if np.any(mutant_frac>0.5):
        summary_dict['t_half'] = np.argwhere(mutant_frac>0.5).min()*dt
    else:
        summary_dict['t_half'] = -1
    
    # time till 99% mutants
    if np.any(mutant_frac>=0.99):
        summary_dict['t_99M'] = np.argwhere(mutant_frac>=0.99).min()*dt
        summary_dict['t_tot'] = np.argwhere(mutant_frac>=0.99).min()*dt
    else:
        t_99M_ind = -1
        summary_dict['t_99M'] = -1
    t_end_ind = -1    
    
    if summary_dict['t_99M'] == -1:
        summary_dict['t_tot'] = len(pop_tots['mutants'])*dt
        t_end_ind = -1   
    else:
        summary_dict['t_tot'] = summary_dict['t_99M']
        t_end_ind = np.argwhere(mutant_frac>=0.99).min()
    
#     if t_99M_ind != -1 and t_washout_ind != -1:
#         t_end_ind = min(t_99M_ind, t_washout_ind)
#     elif t_99M_ind != -1:
#         t_end_ind = t_99M_ind
#     elif t_washout_ind != -1:
#         t_end_ind = t_washout_ind
#     summary_dict['t_tot'] = timepoints[t_end_ind]
    
    # final population state
    summary_dict['final_pop'] = results[t_end_ind,0:-2]
    # integral of producers
    summary_dict['prod_integral'] = dt*np.sum(pop_tots['producers'][0:t_end_ind])
    # total integral
    summary_dict['tot_integral'] = dt*np.sum(pop_tots['total'][0:t_end_ind])
    # median fraction of producers (should be ~the steady state)
    summary_dict['prod_frac_median'] = np.median(pop_tots['producers'][0:t_end_ind]/pop_tots['total'][0:t_end_ind])
    # average fraction of producers
    summary_dict['prod_frac_avg'] = summary_dict['prod_integral']/summary_dict['tot_integral']
    summary_dict['tot_pop_avg'] = np.mean(pop_tots['total'][0:t_end_ind])
    summary_dict['prod_pop_avg'] = np.mean(pop_tots['producers'][0:t_end_ind])
    summary_dict['total_production'] = results[t_end_ind,-1]
    summary_dict['production_rate'] = summary_dict['total_production']/summary_dict['t_tot']
    
    summary = []
    for col in summary_cols:
        summary.append(summary_dict[col])
        
    return summary
"""
The functions below are for running multiprocessed deterministic and stochastic simulations of batch dilutions for the differentiation selection (both idential and split cassette) architectures, and calculating summary statistics to return. The summary statistics function is the same as that used for the differentiation architecture simulations
"""
def run_diff_select_det_batch(muN=1,
                              split_cassettes=False,
                              n_cassette_array=np.array([1,2,3]),
                              cassettes_equal=False,n_int_cassette_array=np.array([1,2,3]),
                              n_div_array = np.array([4]),
                              muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
                              kdiffs = np.array([0.05,0.1,0.2]),
                              kMBs=np.array([1e-6]),
                              kMDs=np.array([1e-6]),
                              kMIs=np.array([1e-6]),                      
                              kPL=1e-4,
                              Ks=np.array([1e9,1e10,1e11]),
                              V=1, #mL
                              Vmax = 1e6/6.022e23*422.36*1e6*3600, #(molecules/cell/s)*(1 mol/6.022e23 molec)*(422.36g/mol)*(1e6ug/g)*(3600s/h)
                              Km=6.7, #ug/mL
                              MIC=1.1, #ug/mL
                              abx_in=100, #ug/m
                              selection='recessive',
                              t_max = 'variable',
                              dt = 0.01,
                              summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection',
                                                       'CIIE',
                                                'n_div','muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                                'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                                'n_growths','prod_integral','tot_integral','prod_frac_median',
                                                'prod_frac_avg','total_production','production_rate',
                                                'genotypes','final_pop']),
                              max_growths=1000,
                              plotting=False,
                              fixed_end=False,
                              complete_growths=False,
                              CIIE=False):
    
    if split_cassettes==True:
        circuit = 'diff_split_select'
    else:
        circuit = 'diff_select'
    if plotting:
        summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection','CIIE','n_div',
                                 'muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                 'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                 'n_growths','prod_integral','tot_integral','prod_frac_median',
                                 'prod_frac_avg','total_production','production_rate','genotypes',
                                 'species_ind_dict','results','production_array','time_array'])
    stochastic = False
    summary = []
#     year = str(time.localtime().tm_year)
#     month = str(time.localtime().tm_mon)
#     if len(month) == 1: month = '0' + month
#     day = str(time.localtime().tm_mday)
#     if len(day) == 1: month = '0' + day
#     date = year + month + day
    
#     try:
#         os.mkdir(folder)
#     except:
#         print('folder already exists')
# #     Nprocessors = len(muPs)
   
    for n_cassettes in n_cassette_array:
        print(f'{n_cassettes} cassettes')
        if cassettes_equal:
            n_int_cassette_array = np.array([n_cassettes])
        for n_int_cassettes in n_int_cassette_array:
            args_list = []
            for n_div in n_div_array:
                for muP in muPs:
        #             print(f'{muP} muP')
                    for K in Ks:
                        V_act = V*K/1e9 # assume carrying capacity of 1e9/mL
        #                 print(f'{K} Kpop')
                        for kdiff in kdiffs:
                            if t_max == 'variable':
                                args_list.append((muN, muP, n_cassettes, n_int_cassettes, kdiff, kMBs, kMDs, kMIs, K,
                                                  kPL,V_act,Vmax,Km,MIC,abx_in,selection,
                                                  12*(np.log(2)/muP), dt, summary_cols, n_div,split_cassettes,max_growths,
                                                  plotting, fixed_end, complete_growths,CIIE)) #checked
                            else:
                                args_list.append((muN, muP, n_cassettes, n_int_cassettes, kdiff, kMBs, kMDs, kMIs,K,
                                                  kPL,V_act,Vmax,Km,MIC,abx_in,selection,
                                                  t_max, dt, summary_cols, n_div,split_cassettes,max_growths,
                                                  plotting, fixed_end, complete_growths, CIIE)) #checked
            #             print(args_list)
            #             with concurrent.futures.ProcessPoolExecutor() as executor:
            Nprocessors = min(7,len(args_list))
            MPPool = MP.Pool(Nprocessors)
            results = MPPool.map(run_diff_select_det_MP_batch, args_list)
    #                 results = executor.map(run_naive_stoch_MP,args_list)
            i = 0
            for batch in results:
    #                 print(batch)
                if i == 0 and n_cassettes == n_cassette_array[0]:
                    results_concat = batch
                    i = 1
                else:
                    for result in batch:
    #                         print(result)
                        results_concat.append(result) 
            MPPool.terminate()
    return pd.DataFrame(columns=summary_cols,data=results_concat)

def run_diff_select_det_MP_batch(args):
    muN = args[0]
    muP = args[1]
    n_cassettes = args[2]
    n_int_cassettes = args[3]
    kdiff = args[4]
    kMBs = args[5]
    kMDs = args[6]
    kMIs = args[7]
    K = args[8]
    kPL = args[9]
    V = args[10]
    Vmax = args[11]
    Km = args[12]
    MIC = args[13]
    abx_in = args[14]
    selection = args[15]
    t_max = args[16]
    dt = args[17]
    summary_cols = args[18]
    n_div = args[19]
    split_cassettes = args[20]
    max_growths = args[21]
    plotting = args[22]
    fixed_end = args[23]
    complete_growths = args[24]
    CIIE = args[25]
    
    stochastic = True
    if split_cassettes==True:
        circuit = 'diff_split_select'
    else:
        circuit = 'diff_select'
    beta_p = 1
    cell_init = K/50
    dt_mut = dt
    stochastic = False
    summary = []
    
#     for kdiff in kdiffs: 
    for kMB in kMBs:
        for kMD in kMDs:
            for kMI in kMIs:
                if plotting:
                    results, concs_endpoint, times_endpoint, ppmn_dict, ppmn_ind_dict, n_growths, species = \
                        multicassette_diff_select_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, 
                                                                   n_int_cassettes, n_div, muN, muP, beta_p, kdiff, 
                                                                   kMB, kMD, kMI, kPL, K, 
                                                                   V, Vmax, Km, MIC, abx_in,selection,
                                                                   dilution_factor=50, 
                                                                   frac_at_dilution=0.95,
                                                                   prod_frac_thresh=1e-3, 
                                                                   split_cassettes=split_cassettes, 
                                                                   stochastic=stochastic, max_growths=max_growths,
                                                                   plotting=plotting,fixed_end=fixed_end,
                                                                   complete_growths=complete_growths,
                                                                   CIIE=CIIE) #checked

                    results_summary = diff_summary_plotting(results, concs_endpoint, times_endpoint, circuit, ppmn_ind_dict, 
                                                    n_growths, muN, muP, n_cassettes, n_int_cassettes, n_div, kdiff, kMB, 
                                                    kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in, selection, dt, stochastic, 0,
                                                    summary_cols, species, CIIE) #checked
                    
                    
                else:
                    results, ppmn_dict, ppmn_ind_dict, n_growths, species = \
                        multicassette_diff_select_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, 
                                                                   n_int_cassettes, n_div, muN, muP, beta_p, kdiff, 
                                                                   kMB, kMD, kMI, kPL, K, 
                                                                   V, Vmax, Km, MIC, abx_in,selection,
                                                                   dilution_factor=50, 
                                                                   frac_at_dilution=0.95,
                                                                   prod_frac_thresh=1e-3, 
                                                                   split_cassettes=split_cassettes, 
                                                                   stochastic=stochastic, max_growths=max_growths,
                                                                   plotting=plotting,fixed_end=fixed_end,
                                                                   complete_growths=complete_growths,
                                                                   CIIE=CIIE) #checked

                    results_summary = diff_summary(results, circuit, ppmn_ind_dict, n_growths, muN, muP, n_cassettes, 
                                           n_int_cassettes, n_div, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in, 
                                           selection, dt, stochastic, 0, summary_cols, species, CIIE) # checked

                summary.append(results_summary)
    return summary


def run_diff_select_stoch_batch(muN=1,
                                split_cassettes=False,
                                n_cassette_array=np.array([1,2,3]),
                                cassettes_equal=False,
                                n_int_cassette_array=np.array([1,2,3]),
                                n_div_array = np.array([4]),
                                muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
                                kdiffs = np.array([0.05,0.1,0.2]),
                                kMBs=np.array([1e-6]),
                                kMDs=np.array([1e-6]),
                                kMIs=np.array([1e-6]),                      
                                kPL=1e-4,
                                Ks=np.array([1e4,1e5,1e6,1e7]),
                                V=1,#mL
                                Vmax = 1e6/6.022e23*422.36*1e6*3600, #(molecules/cell/s)*(1 mol/6.022e23 molec)*(422.36g/mol)*(1e6ug/g)*(3600s/h)
                                Km=6.7, #ug/mL
                                MIC=1.1, #ug/mL
                                abx_in=100, #ug/m
                                selection='recessive',
                                t_max = 'variable',
                                dt = 0.01,
                                summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection',
                                                         'CIIE',
                                                'n_div','muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                                'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                                'n_growths','prod_integral','tot_integral','prod_frac_median',
                                                'prod_frac_avg','total_production','production_rate',
                                                'genotypes','final_pop']),
                                n_stoch=8,
                                max_growths=1000,
                                save_csvs=True,
                                fstring=None,
                                plotting=False,
                                fixed_end=False,
                                complete_growths=False,
                                CIIE=False):
    
    if split_cassettes==True:
        circuit = 'diff_split_select'
    else:
        circuit = 'diff_select'
    if plotting:
        summary_cols = np.array(['circuit','stochastic','n_cassettes','n_int_cassettes','selection','CIIE','n_div',
                                 'muN','muP','Kpop','kdiff','kMB','kMD','kMI','kPL','Vmax','Km',
                                 'MIC','abx','V','rep','t_dilute','t_1M','t_half','t_99M','t_tot',
                                 'n_growths','prod_integral','tot_integral','prod_frac_median',
                                 'prod_frac_avg','total_production','production_rate','genotypes',
                                 'species_ind_dict','results','production_array','time_array'])
    stochastic = True
    results_concat = []
    for n_cassettes in n_cassette_array:
        print(f'{n_cassettes} cassettes')
        if cassettes_equal:
            n_int_cassette_array = np.array([n_cassettes])
        for n_int_cassettes in n_int_cassette_array:
            for K in Ks:
                V_act = V*K/1e9 # assume carrying capacity of 1e9/mL
                print(f'{K} K')
                args_list = []
                for muP in muPs:
                    for kdiff in kdiffs:
                        for n_div in n_div_array:
                            for kMB in kMBs:
                                for kMD in kMDs:
                                    for kMI in kMIs:
                                        if t_max == 'variable':
                                            args_list.append((muN, muP, n_cassettes, n_int_cassettes, kdiff, kMB, 
                                                              kMD, kMI, K, kPL,V_act,Vmax,Km,MIC,abx_in,selection,
                                                              12*(np.log(2)/muP), dt, summary_cols, 
                                                              n_div,split_cassettes,max_growths,n_stoch,
                                                              plotting, fixed_end,complete_growths,
                                                              CIIE))#checked
                                        else:
                                            args_list.append((muN, muP, n_cassettes, n_int_cassettes, kdiff, kMB, 
                                                              kMD, kMI, K, kPL,V_act,Vmax,Km,MIC,abx_in,selection,
                                                              t_max, dt, summary_cols, 
                                                              n_div,split_cassettes,max_growths,n_stoch,
                                                              plotting, fixed_end,complete_growths,
                                                              CIIE))#checked
            #             print(args_list)
            #             with concurrent.futures.ProcessPoolExecutor() as executor:
                Nprocessors = min(7,len(args_list))
                MPPool = MP.Pool(Nprocessors)
                results = MPPool.map(run_diff_select_stoch_MP_batch, args_list)

                results_current = []
                for batch in results:
                    for result in batch:
                        results_current += [result]
                        results_concat += [result]
    #                 if i == 0 and n_cassettes == n_cassette_array[0]:
    #                     results_concat = batch
    #                     i = 1
    #                 else:
    #                     for result in batch:
    #                         results_concat.append(result) 
                MPPool.terminate()
                if save_csvs is True:
                    df_results_current = pd.DataFrame(columns=summary_cols,data=results_current)
                    year = str(time.localtime().tm_year)
                    month = str(time.localtime().tm_mon)
                    if len(month) == 1: month = '0' + month
                    day = str(time.localtime().tm_mday)
                    if len(day) == 1: day = '0' + day
                    date = year + month + day
                    if fstring is None:
                        df_results_current.to_csv(f'{date}_{circuit}_{n_cassettes}cass_{n_int_cassettes}intcass_{K}K_batch_{t_max}.csv')
                    else:
                        df_results_current.to_csv(f'{fstring}{date}_{circuit}_{n_cassettes}cass_{n_int_cassettes}intcass_{K}K_batch_{t_max}.csv')
      
    return pd.DataFrame(columns=summary_cols,data=results_concat)

def run_diff_select_stoch_MP_batch(args):
    muN = args[0]
    muP = args[1]
    n_cassettes = args[2]
    n_int_cassettes = args[3]
    kdiff = args[4]
    kMB = args[5]
    kMD = args[6]
    kMI = args[7]    
    K = args[8]
    kPL = args[9]
    V = args[10]
    Vmax = args[11]
    Km = args[12]
    MIC = args[13]
    abx_in = args[14]
    selection = args[15]
    t_max = args[16]
    dt = args[17]
    summary_cols = args[18]
    n_div = args[19]
    split_cassettes = args[20]
    max_growths = args[21]
    n_stoch = args[22]
    plotting = args[23]
    fixed_end = args[24]
    complete_growths = args[25]
    CIIE = args[26]
    
    
    stochastic = True
    if split_cassettes==True:
        circuit = 'diff_split_select'
    else:
        circuit = 'diff_select'
    beta_p = 1
    cell_init = K/50
    dt_mut = dt
    stochastic = True
    summary = []
    
#     for kdiff in kdiffs:
    for i in range(n_stoch):
        if plotting:
            results, concs_endpoint, times_endpoint, ppmn_dict, ppmn_ind_dict, n_growths, species = \
                multicassette_diff_select_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, 
                                                           n_int_cassettes, n_div, muN, muP, beta_p, kdiff, 
                                                           kMB, kMD, kMI, kPL, K, 
                                                           V, Vmax, Km, MIC, abx_in,selection,
                                                           dilution_factor=50, 
                                                           frac_at_dilution=0.95,
                                                           prod_frac_thresh=1e-3, 
                                                           split_cassettes=split_cassettes, 
                                                           stochastic=stochastic, max_growths=max_growths,
                                                           plotting=plotting,fixed_end=fixed_end,
                                                           complete_growths=complete_growths,
                                                           CIIE=CIIE) #checked

            results_summary = diff_summary_plotting(results, concs_endpoint, times_endpoint, circuit, ppmn_ind_dict, 
                                            n_growths, muN, muP, n_cassettes, n_int_cassettes, n_div, kdiff, kMB, 
                                            kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in, selection,dt, stochastic, i+1,
                                            summary_cols, species, CIIE) #checked


        else:
            results, ppmn_dict, ppmn_ind_dict, n_growths, species = \
                multicassette_diff_select_fixed_step_batch(t_max, dt, dt_mut, cell_init, n_cassettes, 
                                                           n_int_cassettes, n_div, muN, muP, beta_p, kdiff, 
                                                           kMB, kMD, kMI, kPL, K, 
                                                           V, Vmax, Km, MIC, abx_in,selection,
                                                           dilution_factor=50, 
                                                           frac_at_dilution=0.95,
                                                           prod_frac_thresh=1e-3, 
                                                           split_cassettes=split_cassettes, 
                                                           stochastic=stochastic, max_growths=max_growths,
                                                           plotting=plotting,fixed_end=fixed_end,
                                                           complete_growths=complete_growths,
                                                           CIIE=CIIE) #checked

            results_summary = diff_summary(results, circuit, ppmn_ind_dict, n_growths, muN, muP, n_cassettes, 
                                   n_int_cassettes, n_div, kdiff, kMB, kMD, kMI, kPL, K, V, Vmax, Km, MIC, abx_in,selection,
                                   dt, stochastic, i+1, summary_cols, species, CIIE) # checked

        summary.append(results_summary)
    return summary

# """
# Functions below are for running naive deterministic and stochastic simulations in a chemostat rather than with batch dilutions
# """
# @numba.jit(nopython=True)
# def multicassette_naive_fixed_step_growth_chemostat(out,K,D,source,dest,rates,mus,beta_ps,dt,dt_mut,stochastic):
#     for i in range(out.shape[0]-1):
#         if round(i*dt/dt_mut,6) == int(round(i*dt/dt_mut,6)):    
#             i_1 = multicassette_naive_fixed_step_update(out[i,:],K,D,source,dest,rates,mus,
#                                                     beta_ps,dt,dt_mut,stochastic,True)
#         else:
#             i_1 = multicassette_naive_fixed_step_update(out[i,:],K,D,source,dest,rates,mus,
#                                                     beta_ps,dt,dt_mut,stochastic,False)
#         i_1[i_1<0] = 0
#         out[i+1,:] = i_1
    
#     return out
   
        
# def multicassette_naive_fixed_step_chemostat(t, dt, dt_mut, cell_init, n_cassettes, muN, muP,
#                                    beta_p, kMB, K, D=0.2,prod_frac_thresh=1e-3,
#                                    stochastic=False,max_time=10000,selection='recessive'):
    
#     if int(dt_mut/dt) != (dt_mut/dt):
#         print('dt_mut must be an integer multiple of dt')
#         return

#     source, dest, rates, mus, beta_ps, pn_dict, pn_ind_dict = \
#             naive_multicassette_mut_list_fixed_step(kMB, muN, muP, beta_p, 
#                                             n_cassettes, selection)
    
    
#     concs = np.zeros(len(mus)+1)
#     concs[0] = cell_init
#     production = 0
#     continue_growth = True
#     producer_locs = pn_ind_dict['producers']
#     nonproducer_locs = pn_ind_dict['mutants']
#     n=0
#     while continue_growth:
#         out = np.zeros((int(t/dt)+1,len(concs)))
#         out[0,:]=concs
#         out = multicassette_naive_fixed_step_growth_chemostat(out,K,D,source,dest,rates,mus,
#                                                     beta_ps,dt,dt_mut, stochastic)
#         end_concs = out[-1,:]
#         if n == 0:
#             out_concat = out[0:-1,:]
#             n = 1
#         else:
#             out_concat = np.concatenate((out_concat,out[0:-1,:]))
#         if type(max_time) is int:
#             if len(out_concat)*dt >= max_time:
#                 continue_growth = False
#         if end_concs[0:-1].sum() < 0.001*K:
#             continue_growth = False
#         elif (end_concs[producer_locs].sum()/end_concs[0:-1].sum()) < prod_frac_thresh:
#             continue_growth = False
#         else:
#             concs = end_concs
#             if np.any(concs<0):
#                 concs[concs<0] = 0
#             if np.any(np.isnan(concs)):
#                 print('nan')
#     return out_concat, pn_dict, pn_ind_dict

# def run_naive_stoch_chemostat(muN=1, 
#                     n_cassette_array=np.array([1,2,3]),
#                     muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
#                     kMBs=np.array([1e-6]),
#                     Ds=np.array([0.2,0.3,0.4,0.5]),
#                     Ks=np.array([1e6,1e7,1e8,1e9,1e10]),
#                     selections=np.array(['recessive','additive']),
#                     dt = 0.01,
#                     summary_cols = np.array(['circuit','selection','stochastic','n_cassettes',
#                                              'muN','muP','D','Kpop','kMB','rep',
#                                              't_1M','t_half','t_99M','t_tot','prod_integral',
#                                              'tot_integral','prod_frac_median','prod_frac_avg',
#                                              'prod_Kfrac_median','prod_Kfrac_avg','total_production',
#                                              'production_rate','washout']),
#                     n_stoch=20,
#                     max_time=10000,
#                     fstring=None,
#                     save_csvs=True):
    
#     circuit = 'naive'
#     stochastic = True
#     results_concat = []
    
#     for selection in selections:
#         print(f'selection: {selection}')
#         for n_cassettes in n_cassette_array:
#             print(f'{n_cassettes} cassettes')
#             args_list = []
#             for muP in muPs:
#                 for K in Ks:
#                     for D in Ds:
#                         for kMB in kMBs:
#                             args_list.append((muN, muP, n_cassettes, selection, kMB, D, K, 
#                                               dt, summary_cols, n_stoch, max_time))
#             Nprocessors = min(7,len(args_list))
#             MPPool = MP.Pool(Nprocessors)
#             results = MPPool.map(run_naive_stoch_MP_chemostat, args_list)
#             results_current = []
#             for batch in results:
#                 for result in batch:
#                     results_current += [result]
#                     results_concat += [result]

#             MPPool.terminate()
#             if save_csvs is True:
#                 df_results_current = pd.DataFrame(columns=summary_cols,data=results_current)
#                 year = str(time.localtime().tm_year)
#                 month = str(time.localtime().tm_mon)
#                 if len(month) == 1: month = '0' + month
#                 day = str(time.localtime().tm_mday)
#                 if len(day) == 1: day = '0' + day
#                 date = year + month + day
#                 if fstring is None:
#                     df_results_current.to_csv(f'{date}_{circuit}_{selection}_{n_cassettes}cassettes_chemostat.csv')
#                 else:
#                     df_results_current.to_csv(f'{fstring}{date}_{circuit}_{selection}_{n_cassettes}cassettes_chemostat.csv')
#     return pd.DataFrame(columns=summary_cols,data=results_concat)

# def run_naive_stoch_MP_chemostat(args):
#     muN = args[0]
#     muP = args[1]
#     n_cassettes = args[2]
#     selection = args[3]
#     kMB = args[4]
#     D = args[5]
#     K = args[6]
#     dt = args[7]
#     summary_cols = args[8]
#     n_stoch = args[9]
#     max_time = args[10]
#     beta_p = 1
    
#     dt_mut = dt
#     cell_init = K/2
    
#     circuit = 'naive'
#     summary = []
#     for i in range(n_stoch):
#         results, pn_dict, pn_ind_dict = \
#             multicassette_naive_fixed_step_chemostat(12, dt, dt_mut, cell_init, n_cassettes, muN, muP,
#                                            beta_p, kMB, K, D,prod_frac_thresh=1e-3,
#                                            stochastic=True,max_time=max_time,selection=selection)
        
#         results_summary = naive_summary_chemostat(results,pn_ind_dict, muN, muP,selection, n_cassettes, 
#                                         kMB, D, K, dt, True, i+1, summary_cols)

#         summary.append(results_summary)
#     return summary

# def run_naive_det_chemostat(muN=1, 
#                     n_cassette_array=np.array([1,2,3]),
#                     muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
#                     kMBs=np.array([1e-6]),
#                     Ds=np.array([0.2,0.3,0.4,0.5]),
#                     Ks=np.array([1e6,1e7,1e8,1e9,1e10]),
#                     selections=np.array(['recessive','additive']),
#                     dt = 0.01,
#                     summary_cols = np.array(['circuit','selection','stochastic','n_cassettes',
#                                              'muN','muP','D','Kpop','kMB','rep',
#                                              't_1M','t_half','t_99M','t_tot','prod_integral',
#                                              'tot_integral','prod_frac_median','prod_frac_avg',
#                                              'prod_Kfrac_median','prod_Kfrac_avg','total_production',
#                                              'production_rate','washout']),
#                     max_time=10000):
    
#     circuit = 'naive'
#     summary = []
#     args_list = []
#     for selection in selections:
#         for n_cassettes in n_cassette_array:
#             for muP in muPs:
#                 for K in Ks:
#                     args_list.append((muN, muP, n_cassettes, selection, kMBs, Ds, K, 
#                                       dt, summary_cols,max_time))
       

#     Nprocessors = min(7,len(args_list))
#     MPPool = MP.Pool(Nprocessors)
#     results = MPPool.map(run_naive_det_MP_chemostat, args_list)
#     i = 0
# #     print(f'results: {results}')
#     for batch in results:
# #         print(f'batch: {batch}')
#         if i == 0:
#             results_concat = batch
#             i = 1
#         else:
#             for result in batch:
# #                 print(f'result: {result}')
#                 results_concat.append(result) 
#     MPPool.terminate()
#     return pd.DataFrame(columns=summary_cols,data=results_concat)

# def run_naive_det_MP_chemostat(args):
#     muN = args[0]
#     muP = args[1]
#     n_cassettes = args[2]
#     selection = args[3]
#     kMBs = args[4]
#     Ds = args[5]
#     K = args[6]
#     dt = args[7]
#     summary_cols = args[8]
#     max_time = args[9]
    
#     beta_p = 1
#     dt_mut = dt
#     cell_init = K/2
    
#     circuit = 'naive'
#     summary = []
#     for kMB in kMBs:
#         for D in Ds:
#             results, pn_dict, pn_ind_dict = \
#                 multicassette_naive_fixed_step_chemostat(12, dt, dt_mut, cell_init, n_cassettes, muN, muP,
#                                            beta_p, kMB, K, D,prod_frac_thresh=1e-3,
#                                            stochastic=False,max_time=max_time,selection=selection)
            
#             results_summary = naive_summary_chemostat(results,pn_ind_dict, muN, muP, selection, n_cassettes, 
#                                             kMB, D, K, dt, False, 0, summary_cols)

#             summary.append(results_summary)
# #     print(f'summary: {summary}')
#     return summary



# def naive_summary_chemostat(results, species_ind_dict, muN, muP, selection, n_cassettes, kMB, D, K, dt, stochastic,
#                  rep, summary_cols): 

#     summary_dict = {}
#     summary_dict['circuit'] = 'naive'
#     summary_dict['selection'] = selection
#     summary_dict['stochastic'] = stochastic
#     summary_dict['muN'] = muN
#     summary_dict['muP'] = muP
#     summary_dict['n_cassettes'] = n_cassettes
#     summary_dict['kMB'] = kMB
#     summary_dict['D'] = D
#     summary_dict['Kpop'] = K
#     summary_dict['rep'] = rep
    
#     pop_tots = {}
#     for label in species_ind_dict:
#         pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()
#     pop_tots['total'] = results[:,0:-1].sum(axis=1).flatten()
    
# #     check if washout happened
#     if np.any(pop_tots['total'] < summary_dict['Kpop']*0.001):
#         t_washout_ind = np.argwhere(pop_tots['total']<summary_dict['Kpop']*0.001).min()
#         t_washout = dt*t_washout_ind
#         summary_dict['washout'] = True
#     else:
#         summary_dict['washout'] = False
    
#     # time till one mutant
#     if np.any(pop_tots['mutants']>=1):
#         summary_dict['t_1M'] = np.argwhere(pop_tots['mutants']>=1).min()*dt
#     else:
#         summary_dict['t_1M'] = -1
        
#     # time till half mutants
#     mutant_frac = pop_tots['mutants'] / pop_tots['total']
#     if np.any(mutant_frac>0.5):
#         summary_dict['t_half'] = np.argwhere(mutant_frac>0.5).min()*dt
#     else:
#         summary_dict['t_half'] = -1
    
#     # time till 99% mutants
#     if np.any(mutant_frac>=0.99):
#         summary_dict['t_99M'] = np.argwhere(mutant_frac>=0.99).min()*dt
#     else:
#         t_99M_ind = -1
#         summary_dict['t_99M'] = -1
       
#     # time till <0.1% producers
#     producer_frac = pop_tots['producers']/pop_tots['total']
#     if np.any(producer_frac[int(-12//dt):]<1e-3):
#         producer_frac_end_ind = np.argwhere(producer_frac<1e-3)[np.argwhere(producer_frac<1e-3)>(len(producer_frac)-12//dt)].min()
#     else: 
#         producer_frac_end_ind = -1
        
#     if summary_dict['washout'] == True:
#         if producer_frac_end_ind == -1:
#             t_end_ind = t_washout_ind
#         else:
#             t_end_ind = min(t_washout_ind,producer_frac_end_ind)
#         summary_dict['t_tot'] = dt*t_end_ind
#     elif producer_frac_end_ind != -1:
#         t_end_ind = producer_frac_end_ind
#         summary_dict['t_tot'] = dt*t_end_ind
#     else:
#         t_end_ind = -1
#         summary_dict['t_tot'] = dt*len(producer_frac)
    
#     # integral of producers
#     summary_dict['prod_integral'] = dt*np.sum(pop_tots['producers'][0:t_end_ind])
#     # total integral
#     summary_dict['tot_integral'] = dt*np.sum(pop_tots['total'][0:t_end_ind])
#     # median fraction of producers (should be ~the steady state)
#     summary_dict['prod_frac_median'] = np.median(pop_tots['producers'][0:t_end_ind]/pop_tots['total'][0:t_end_ind])
#     # average fraction of producers
#     summary_dict['prod_frac_avg'] = summary_dict['prod_integral']/summary_dict['tot_integral']
#     # median fraction producer pop/K (should be ~the steady state)
#     summary_dict['prod_Kfrac_median'] = np.median(pop_tots['producers'][0:t_end_ind])/K
#     # average fraction producer pop/K
#     summary_dict['prod_Kfrac_avg'] = np.mean(pop_tots['producers'][0:t_end_ind])/K
#     summary_dict['tot_pop_avg'] = np.mean(pop_tots['total'][0:t_end_ind])
#     summary_dict['prod_pop_avg'] = np.mean(pop_tots['producers'][0:t_end_ind])
#     summary_dict['total_production'] = results[t_end_ind,-1]
#     summary_dict['production_rate'] = summary_dict['total_production']/summary_dict['t_tot']
    
#     summary = []
#     for col in summary_cols:
#         summary.append(summary_dict[col])
        
#     return summary


# """
# Functions below are for running differentiation deterministic and stochastic simulations in a chemostat rather than with batch dilutions
# """
# @numba.jit(nopython=True)
# def multicassette_diff_fixed_step_growth_chemostat(out,K,D,source,dest,rates,mus,beta_ps,dt,dt_mut,stochastic):
#     for i in range(out.shape[0]-1):
#         if round(i*dt/dt_mut,6) == int(round(i*dt/dt_mut,6)):    
#             i_1 = multicassette_diff_fixed_step_update(out[i,:],K,D,source,dest,rates,mus,
#                                                     beta_ps,dt,dt_mut,stochastic,True)
#         else:
#             i_1 = multicassette_diff_fixed_step_update(out[i,:],K,D,source,dest,rates,mus,
#                                                     beta_ps,dt,dt_mut,stochastic,False)
#         i_1[i_1<0] = 0
#         out[i+1,:] = i_1
    
#     return out
   
        
# def multicassette_diff_fixed_step_chemostat(t, dt, dt_mut, cell_init, n_cassettes, muN, muP,
#                                    beta_p, kdiff, kMB, kMD, K, D,prod_frac_thresh=1e-3,
#                                    stochastic=False,max_time=10000):
    
    
    
#     if int(dt_mut/dt) != (dt_mut/dt):
#         print('dt_mut must be an integer multiple of dt')
#         return

#     source, dest, rates, mus, beta_ps, ppm_dict, ppm_ind_dict = \
#             diff_multicassette_mut_list_fixed_step(kdiff,kMB,kMD, muN, muP, beta_p, 
#                                             n_cassettes)

#     concs = np.zeros(len(mus)+1)
#     concs[0] = cell_init
#     production = 0
#     continue_growth = True
#     producer_locs = ppm_ind_dict['producers']
#     n=0
#     while continue_growth:
#         out = np.zeros((int(t/dt)+1,len(concs)))
#         out[0,:]=concs
#         out = multicassette_diff_fixed_step_growth_chemostat(out,K,D,source,dest,rates,mus,
#                                                     beta_ps,dt,dt_mut, stochastic)
#         end_concs = out[-1,:]
#         if n == 0:
#             out_concat = out[0:-1,:]
#             n = 1
#         else:
#             out_concat = np.concatenate((out_concat,out[0:-1,:]))
#         if type(max_time) is int:
#             if len(out_concat)*dt >= max_time:
#                 continue_growth = False
#         if end_concs[0:-1].sum() < 0.001*K:
#             continue_growth = False
#         elif (end_concs[producer_locs].sum()/end_concs[0:-1].sum()) < prod_frac_thresh:
#             continue_growth = False
#         else:
#             concs = end_concs
#             if np.any(concs<0):
#                 concs[concs<0] = 0
#             if np.any(np.isnan(concs)):
#                 print('nan')
#     return out_concat, ppm_dict, ppm_ind_dict

# def run_diff_det_chemostat(muN=1, 
#                    n_cassette_array=np.array([1,2,3]),
#                    muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
#                    kdiffs = np.array([0.05,0.1,0.2]),
#                    kMBs=np.array([1e-6]),
#                    kMDs=np.array([1e-6]),
#                    Ds=np.arange(1,9)/10,
#                    Ks=np.array([1e4,1e5,1e6,1e7]),
#                    dt = 0.01,
#                    summary_cols = np.array(['circuit','stochastic','n_cassettes','n_div',
#                             'muN','muP','D','Kpop','kdiff','kMB','kMD','rep',
#                             't_1M','t_half','t_99M','t_tot','prod_integral',
#                              'tot_integral','prod_frac_median','prod_frac_avg',
#                              'prod_Kfrac_median','prod_Kfrac_avg','total_production',
#                              'production_rate','washout']),
#                    max_time=10000):
    
#     circuit = 'diff'
#     stochastic = False
#     summary = []
   
#     args_list = []
#     for n_cassettes in n_cassette_array:
#         print(f'{n_cassettes} cassettes')
#         for muP in muPs:
#             for K in Ks:
#                 args_list.append((muN, muP, n_cassettes, kdiffs, kMBs, kMDs, K,
#                               Ds, dt, summary_cols,max_time))

#     Nprocessors = min(7,len(args_list))
#     MPPool = MP.Pool(Nprocessors)
#     results = MPPool.map(run_diff_det_MP_chemostat, args_list)
#     i = 0
#     for batch in results:
#         if i == 0:
#             results_concat = batch
#             i = 1
#         else:
#             for result in batch:
#                 results_concat.append(result) 
#     MPPool.terminate()
#     return pd.DataFrame(columns=summary_cols,data=results_concat)

# def run_diff_det_MP_chemostat(args):
#     muN = args[0]
#     muP = args[1]
#     n_cassettes = args[2]
#     kdiffs = args[3]
#     kMBs = args[4]
#     kMDs = args[5]
#     K = args[6]
#     Ds = args[7]
#     dt = args[8]
#     summary_cols = args[9]
#     max_time = args[10]
    
#     beta_p = 1
#     cell_init = K/2
#     dt_mut = dt
#     stochastic = False
#     circuit = 'diff'
#     summary = []
#     for kMB in kMBs:
#         for kMD in kMDs:
#             for D in Ds:
#                 for kdiff in kdiffs:
#                     results, ppm_dict, ppm_ind_dict = \
#                         multicassette_diff_fixed_step_chemostat(12, dt, dt_mut, cell_init, n_cassettes, muN, muP,
#                                                       beta_p, kdiff, kMB, kMD, K, D, prod_frac_thresh=1e-3,
#                                                       stochastic=False, max_time=max_time)
#                     results_summary = diff_summary_chemostat(results, circuit, ppm_ind_dict, muN, muP, n_cassettes, 0,
#                                                    kdiff, kMB, kMD, D, K, dt, stochastic, 0, summary_cols)

#                     summary.append(results_summary)
#     return summary

# def run_diff_stoch_chemostat(muN=1, 
#                    n_cassette_array=np.array([1,2,3]),
#                    muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
#                    kdiffs = np.array([0.05,0.1,0.2]),
#                    kMBs=np.array([1e-6]),
#                    kMDs=np.array([1e-6]),
#                    Ds=np.arange(1,9)/10,
#                    Ks=np.array([1e4,1e5,1e6,1e7]),
#                    dt = 0.01,
#                    summary_cols = np.array(['circuit','stochastic','n_cassettes','n_div',
#                             'muN','muP','D','Kpop','kdiff','kMB','kMD','rep',
#                             't_1M','t_half','t_99M','t_tot','prod_integral',
#                              'tot_integral','prod_frac_median','prod_frac_avg',
#                              'prod_Kfrac_median','prod_Kfrac_avg','total_production',
#                              'production_rate','washout']),
#                    n_stoch=20,
#                    max_time=10000,
#                    save_csvs = True,
#                    fstring=None):
    
#     circuit = 'diff'
#     stochastic = True
#     results_concat = []
      
#     for n_cassettes in n_cassette_array:
#         print(f'{n_cassettes} cassettes')
#         for K in Ks:
#             print(f'{K} K')
#             args_list = []
#             for muP in muPs:
#                 for kdiff in kdiffs:
#                     for kMB in kMBs:
#                         for kMD in kMDs:
#                             for D in Ds:
#                                 args_list.append((muN, muP, n_cassettes, kdiff, kMB, kMD, K,
#                                               D, dt, summary_cols, n_stoch, max_time))
#             Nprocessors = min(7,len(args_list))
#             MPPool = MP.Pool(Nprocessors)
#             results = MPPool.map(run_diff_stoch_MP_chemostat, args_list)

# #             i = 0
            
#             results_current = []
#             for batch in results:
#                 for result in batch:
#                     results_current += [result]
#                     results_concat += [result]
# #                 if i == 0 and n_cassettes == n_cassette_array[0]:
# #                     results_concat = batch
# #                     i = 1
# #                 else:
# #                     for result in batch:
# #                         results_concat.append(result) 
#             MPPool.terminate()
#             if save_csvs is True:
#                 df_results_current = pd.DataFrame(columns=summary_cols,data=results_current)
#                 year = str(time.localtime().tm_year)
#                 month = str(time.localtime().tm_mon)
#                 if len(month) == 1: month = '0' + month
#                 day = str(time.localtime().tm_mday)
#                 if len(day) == 1: day = '0' + day
#                 date = year + month + day
#                 if fstring is None:
#                     df_results_current.to_csv(f'{date}_{circuit}_{n_cassettes}cassettes_{K}K_chemostat.csv')
#                 else:
#                     df_results_current.to_csv(f'{fstring}{date}_{circuit}_{n_cassettes}cassettes_{K}K_chemostat.csv')
                    
            
#     return pd.DataFrame(columns=summary_cols,data=results_concat)

# def run_diff_stoch_MP_chemostat(args):
#     muN = args[0]
#     muP = args[1]
#     n_cassettes = args[2]
#     kdiff = args[3]
#     kMB = args[4]
#     kMD = args[5]
#     K = args[6]
#     D = args[7]
#     dt = args[8]
#     summary_cols = args[9]
#     n_stoch = args[10]
#     max_time = args[11]
    
#     beta_p = 1
#     cell_init = K/50
#     dt_mut = dt
#     stochastic = True
#     circuit = 'diff'
#     summary = []
# #     for kMB in kMBs:
# #         for kMD in kMDs:
#     for i in range(n_stoch):
#         results, ppm_dict, ppm_ind_dict = \
#                         multicassette_diff_fixed_step_chemostat(12, dt, dt_mut, cell_init, n_cassettes, muN, muP,
#                                                       beta_p, kdiff, kMB, kMD, K, D, prod_frac_thresh=1e-3,
#                                                       stochastic=True, max_time=max_time)
#         results_summary = diff_summary_chemostat(results, circuit, ppm_ind_dict, muN, muP, n_cassettes, i+1,
#                                        kdiff, kMB, kMD, D, K, dt, stochastic, i+1, summary_cols)

#         summary.append(results_summary)
# #     print(summary)
#     return summary

# def diff_summary_chemostat(results, circuit, species_ind_dict, muN, muP, n_cassettes, n_div, kdiff, kMB, kMD,
#                  D, K, dt, stochastic, rep, summary_cols): 
    
#     pop_tots = {}
#     summary_dict = {}
#     summary_dict['circuit'] = circuit
#     summary_dict['stochastic'] = stochastic
#     summary_dict['muN'] = muN
#     summary_dict['muP'] = muP
#     summary_dict['n_cassettes'] = n_cassettes
#     summary_dict['n_div'] = n_div
#     summary_dict['kdiff'] = kdiff
#     summary_dict['kMB'] = kMB
#     summary_dict['kMD'] = kMD
#     summary_dict['D'] = D
#     summary_dict['Kpop'] = K
#     summary_dict['rep'] = rep
        
#     pop_tots = {}
#     for label in species_ind_dict:
# #         pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()[0:slice_ind]
#         pop_tots[label] = results[:,species_ind_dict[label]].sum(axis=1).flatten()
#     pop_tots['total'] = results[:,0:-1].sum(axis=1).flatten()
    

# #     check if washout happened
#     if np.any(pop_tots['total'] < summary_dict['Kpop']*0.001):
#         t_washout_ind = np.argwhere(pop_tots['total']<summary_dict['Kpop']*0.001).min()
#         t_washout = dt*t_washout_ind
#         summary_dict['washout'] = True
#     else:
#         summary_dict['washout'] = False
    
#     # time till one mutant
#     if np.any(pop_tots['mutants']>=1):
#         summary_dict['t_1M'] = np.argwhere(pop_tots['mutants']>=1).min()*dt
#     else:
#         summary_dict['t_1M'] = -1
        
#     # time till half mutants
#     mutant_frac = pop_tots['mutants'] / pop_tots['total']
#     if np.any(mutant_frac>0.5):
#         summary_dict['t_half'] = np.argwhere(mutant_frac>0.5).min()*dt
#     else:
#         summary_dict['t_half'] = -1
    
#     # time till 99% mutants
#     if np.any(mutant_frac>=0.99):
#         summary_dict['t_99M'] = np.argwhere(mutant_frac>=0.99).min()*dt
#     else:
#         t_99M_ind = -1
#         summary_dict['t_99M'] = -1
       
#     # time till <0.1% producers
#     producer_frac = pop_tots['producers']/pop_tots['total']
#     if np.any(producer_frac[int(-12//dt):]<1e-3):
#         producer_frac_end_ind = np.argwhere(producer_frac<1e-3)[np.argwhere(producer_frac<1e-3)>(len(producer_frac)-12//dt)].min()
#     else: 
#         producer_frac_end_ind = -1
        
#     if summary_dict['washout'] == True:
#         if producer_frac_end_ind == -1:
#             t_end_ind = t_washout_ind
#         else:
#             t_end_ind = min(t_washout_ind,producer_frac_end_ind)
#         summary_dict['t_tot'] = dt*t_end_ind
#     elif producer_frac_end_ind != -1:
#         t_end_ind = producer_frac_end_ind
#         summary_dict['t_tot'] = dt*t_end_ind
#     else:
#         t_end_ind = -1
#         summary_dict['t_tot'] = dt*len(producer_frac)
    
#     # integral of producers
#     summary_dict['prod_integral'] = dt*np.sum(pop_tots['producers'][0:t_end_ind])
#     # total integral
#     summary_dict['tot_integral'] = dt*np.sum(pop_tots['total'][0:t_end_ind])
#     # median fraction of producers (should be ~the steady state)
#     summary_dict['prod_frac_median'] = np.median(pop_tots['producers'][0:t_end_ind]/pop_tots['total'][0:t_end_ind])
#     # average fraction of producers
#     summary_dict['prod_frac_avg'] = summary_dict['prod_integral']/summary_dict['tot_integral']
#     # median fraction producer pop/K (should be ~the steady state)
#     summary_dict['prod_Kfrac_median'] = np.median(pop_tots['producers'][0:t_end_ind])/K
#     # average fraction producer pop/K
#     summary_dict['prod_Kfrac_avg'] = np.mean(pop_tots['producers'][0:t_end_ind])/K
#     summary_dict['tot_pop_avg'] = np.mean(pop_tots['total'][0:t_end_ind])
#     summary_dict['prod_pop_avg'] = np.mean(pop_tots['producers'][0:t_end_ind])
#     summary_dict['total_production'] = results[t_end_ind,-1]
#     summary_dict['production_rate'] = summary_dict['total_production']/summary_dict['t_tot']
    
#     summary = []
#     for col in summary_cols:
#         summary.append(summary_dict[col])
        
#     return summary

# """
# Functions below are for running differentiation with selection deterministic and stochastic simulations in a chemostat rather than with batch dilutions
# """
# @numba.jit(nopython=True)
# def multicassette_diff_select_fixed_step_growth_chemostat(out,K,D,source,dest,rates,diff_div_loss_locs, diff_div_gain_locs, 
#                                                           mus,beta_ps,dt,dt_mut,stochastic):
    
#     for i in range(out.shape[0]-1):
#         if round(i*dt/dt_mut,6) == int(round(i*dt/dt_mut,6)):  
#             i_1 = multicassette_diff_select_fixed_step_update(out[i,:],K,D,source,dest,rates,mus,beta_ps,dt,dt_mut,stochastic,
#                                                                     diff_div_loss_locs, diff_div_gain_locs,True)
#         else:
#             i_1 = multicassette_diff_select_fixed_step_update(out[i,:],K,D,source,dest,rates,mus,beta_ps,dt,dt_mut,stochastic,
#                                                                     diff_div_loss_locs, diff_div_gain_locs,False)
#         i_1[i_1<0] = 0
#         out[i+1,:] = i_1
    
#     return out
   
        
# def multicassette_diff_select_fixed_step_chemostat(t, dt, dt_mut, cell_init, n_cassettes, n_div, split_cassettes, muN, muP,
#                                    beta_p, kdiff, kMB, kMD, K, D,prod_frac_thresh=1e-3,
#                                    stochastic=False,max_time=10000):
    
    
    
#     if int(dt_mut/dt) != (dt_mut/dt):
#         print('dt_mut must be an integer multiple of dt')
#         return

#     if split_cassettes:
#         source, dest, rates, mus, beta_ps,diff_div_loss_locs, \
#         diff_div_gain_locs,ppmn_dict, ppmn_ind_dict = \
#                 diff_split_select_multicassette_mut_list_fixed_step(kdiff,kMB, kMD, muN, 
#                                                                     muP, beta_p, n_cassettes, n_div)
#     else: 
#         source, dest, rates, mus, beta_ps,diff_div_loss_locs, \
#         diff_div_gain_locs,ppmn_dict, ppmn_ind_dict = \
#                 diff_select_multicassette_mut_list_fixed_step(kdiff,kMB, kMD, muN, 
#                                                                     muP, beta_p, n_cassettes, n_div)

#     concs = np.zeros(len(mus)+1)
#     concs[0] = cell_init
#     continue_growth = True
#     producer_locs = ppmn_ind_dict['producers']
#     n=0
#     while continue_growth:
#         out = np.zeros((int(t/dt)+1,len(concs)))
#         out[0,:]=concs
#         out = multicassette_diff_select_fixed_step_growth_chemostat(out,K,D,source,dest,rates,
#                                                                     diff_div_loss_locs,diff_div_gain_locs,
#                                                                     mus,beta_ps,dt,dt_mut, stochastic)
#         end_concs = out[-1,:]
#         if n == 0:
#             out_concat = out[0:-1,:]
#             n = 1
#         else:
#             out_concat = np.concatenate((out_concat,out[0:-1,:]))
#         if type(max_time) is int:
#             if len(out_concat)*dt >= max_time:
#                 continue_growth = False
#         if end_concs[0:-1].sum() < 0.001*K:
#             continue_growth = False
#         elif (end_concs[producer_locs].sum()/end_concs[0:-1].sum()) < prod_frac_thresh:
#             continue_growth = False
#         else:
#             concs = end_concs
#             if np.any(concs<0):
#                 concs[concs<0] = 0
#             if np.any(np.isnan(concs)):
#                 print('nan')
#     return out_concat, ppmn_dict, ppmn_ind_dict

# def run_diff_select_det_chemostat(muN=1, 
#                    n_cassette_array=np.array([1,2,3]),
#                    n_div_array=np.array([4]),
#                    muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
#                    kdiffs = np.array([0.05,0.1,0.2]),
#                    kMBs=np.array([1e-6]),
#                    kMDs=np.array([1e-6]),
#                    Ds=np.arange(1,9)/10,
#                    Ks=np.array([1e4,1e5,1e6,1e7]),
#                    split_cassettes=False,
#                    dt = 0.01,
#                    summary_cols = np.array(['circuit','stochastic','n_cassettes','n_div',
#                                             'muN','muP','D','Kpop','kdiff','kMB','kMD','rep',
#                                             't_1M','t_half','t_99M','t_tot','prod_integral',
#                                              'tot_integral','prod_frac_median','prod_frac_avg',
#                                              'prod_Kfrac_median','prod_Kfrac_avg','total_production',
#                                              'production_rate','washout']),
#                    max_time=10000):
        
#     stochastic = False
#     summary = []
   
#     args_list = []
#     for n_cassettes in n_cassette_array:
#         for n_div in n_div_array:
#             for muP in muPs:
#                 for K in Ks:
#                     args_list.append((muN, muP, n_cassettes, n_div, kdiffs, kMBs, kMDs, K,
#                                   Ds, dt, summary_cols,max_time,split_cassettes))

#     Nprocessors = min(7,len(args_list))
#     MPPool = MP.Pool(Nprocessors)
#     results = MPPool.map(run_diff_select_det_MP_chemostat, args_list)
#     i = 0
#     for batch in results:
#         if i == 0:
#             results_concat = batch
#             i = 1
#         else:
#             for result in batch:
#                 results_concat.append(result) 
#     MPPool.terminate()
#     return pd.DataFrame(columns=summary_cols,data=results_concat)

# def run_diff_select_det_MP_chemostat(args):
#     muN = args[0]
#     muP = args[1]
#     n_cassettes = args[2]
#     n_div = args[3]
#     kdiffs = args[4]
#     kMBs = args[5]
#     kMDs = args[6]
#     K = args[7]
#     Ds = args[8]
#     dt = args[9]
#     summary_cols = args[10]
#     max_time = args[11]
#     split_cassettes = args[12]
    
#     beta_p = 1
#     cell_init = K/2
#     dt_mut = dt
#     stochastic = False
#     if split_cassettes:
#         circuit = 'diff_split_select'
#     else:
#         circuit = 'diff_select'
#     summary = []
#     for kMB in kMBs:
#         for kMD in kMDs:
#             for D in Ds:
#                 for kdiff in kdiffs:
#                     results, ppmn_dict, ppmn_ind_dict = \
#                         multicassette_diff_select_fixed_step_chemostat(12, dt, dt_mut, cell_init, n_cassettes, n_div,  
#                                                                        split_cassettes, muN, muP, beta_p, kdiff, kMB, kMD, 
#                                                                        K, D, prod_frac_thresh=1e-3, stochastic=False, 
#                                                                        max_time=max_time)
                    
#                     results_summary = diff_summary_chemostat(results, circuit, ppmn_ind_dict, muN, muP, n_cassettes, n_div,
#                                                    kdiff, kMB, kMD, D, K, dt, stochastic, 0, summary_cols)

#                     summary.append(results_summary)
#     return summary

# def run_diff_select_stoch_chemostat(muN=1, 
#                    n_cassette_array=np.array([1,2,3]),
#                    n_div_array=np.array([4]),
#                    muPs=np.array([0.1,0.3,0.5,0.7,0.9]),
#                    kdiffs = np.array([0.05,0.1,0.2]),
#                    kMBs=np.array([1e-6]),
#                    kMDs=np.array([1e-6]),
#                    Ds=np.arange(1,9)/10,
#                    Ks=np.array([1e4,1e5,1e6,1e7]),
#                    split_cassettes=False,
#                    dt = 0.01,
#                    summary_cols = np.array(['circuit','stochastic','n_cassettes','n_div',
#                                             'muN','muP','D','Kpop','kdiff','kMB','kMD','rep',
#                                             't_dilute','t_1M','t_half','t_99M','t_tot','n_growths','prod_integral',
#                                             'tot_integral','prod_frac_median','prod_frac_avg',
#                                             'prod_Kfrac_median','prod_Kfrac_avg', 'total_production',
#                                             'production_rate']),
#                    n_stoch=20,
#                    max_time=10000,
#                    save_csvs = True,
#                    fstring=None):
    
#     stochastic = True
#     results_concat = []
      
#     for n_cassettes in n_cassette_array:
#         print(f'{n_cassettes} cassettes')
#         for n_div in n_div_array:
#             print(f'{n_div} divisions')
#             for K in Ks:
#                 print(f'{K} K')
#                 args_list = []
#                 for muP in muPs:
#                     for kdiff in kdiffs:
#                         for kMB in kMBs:
#                             for kMD in kMDs:
#                                 for D in Ds:
#                                     args_list.append((muN, muP, n_cassettes, n_div, kdiff, kMB, kMD, K,
#                                                   D, dt, summary_cols, n_stoch, max_time, split_cassettes))
#                 Nprocessors = min(7,len(args_list))
#                 MPPool = MP.Pool(Nprocessors)
#                 results = MPPool.map(run_diff_select_stoch_MP_chemostat, args_list)

#     #             i = 0

#                 results_current = []
#                 for batch in results:
#                     for result in batch:
#                         results_current += [result]
#                         results_concat += [result]
#     #                 if i == 0 and n_cassettes == n_cassette_array[0]:
#     #                     results_concat = batch
#     #                     i = 1
#     #                 else:
#     #                     for result in batch:
#     #                         results_concat.append(result) 
#                 MPPool.terminate()
#                 if save_csvs is True:
#                     df_results_current = pd.DataFrame(columns=summary_cols,data=results_current)
#                     year = str(time.localtime().tm_year)
#                     month = str(time.localtime().tm_mon)
#                     if len(month) == 1: month = '0' + month
#                     day = str(time.localtime().tm_mday)
#                     if len(day) == 1: day = '0' + day
#                     date = year + month + day
#                     if fstring is None:
#                         df_results_current.to_csv(f'{date}_{circuit}_{n_cassettes}cassettes_{n_div}div_{K}K_chemostat.csv')
#                     else:
#                         df_results_current.to_csv(f'{fstring}{date}_{circuit}_{n_cassettes}cassettes_{n_div}div_{K}K_chemostat.csv')
                    
            
#     return pd.DataFrame(columns=summary_cols,data=results_concat)

# def run_diff_stoch_MP_chemostat(args):
#     muN = args[0]
#     muP = args[1]
#     n_cassettes = args[2]
#     n_div = args[3]
#     kdiff = args[4]
#     kMB = args[5]
#     kMD = args[6]
#     K = args[7]
#     D = args[8]
#     dt = args[9]
#     summary_cols = args[10]
#     n_stoch = args[11]
#     max_time = args[12]
#     split_cassettes = args[13]
    
#     beta_p = 1
#     cell_init = K/50
#     dt_mut = dt
#     stochastic = True
#     if split_cassettes:
#         circuit = 'diff_split_select'
#     else:
#         circuit = 'diff_select'
#     summary = []
#     for i in range(n_stoch):
#         results, ppmn_dict, ppmn_ind_dict = \
#                         multicassette_diff_select_fixed_step_chemostat(12, dt, dt_mut, cell_init, n_cassettes, n_div,  
#                                                                        split_cassettes, muN, muP, beta_p, kdiff, kMB, kMD, 
#                                                                        K, D, prod_frac_thresh=1e-3, stochastic=True, 
#                                                                        max_time=max_time)
                    
#         results_summary = diff_summary_chemostat(results, circuit, ppmn_ind_dict, muN, muP, n_cassettes, n_div,
#                                        kdiff, kMB, kMD, D, K, dt, stochastic, i+1, summary_cols)

#         summary.append(results_summary)
# #     print(summary)
#     return summary
