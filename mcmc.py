from project import mcmc, LFModelOptimizer, FiveParameterCLF
import numpy as np
import pickle

BOUWENS_15_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12)]
FIVE_PARAM_PRIOR = [[0.8,2.5],[0.8,2.5],[0.8,2.5],[-24,-20],[10,14]]
FIVE_PARAM_CLF_1522_OPTIMIZER = LFModelOptimizer('pts_errs_15+22.npz', FiveParameterCLF, x0=BOUWENS_15_X0, prior=FIVE_PARAM_PRIOR)
FIVE_PARAM_CLF_15_OPTIMIZER = LFModelOptimizer('pts_errs_15.npz', FiveParameterCLF, x0=BOUWENS_15_X0, prior=FIVE_PARAM_PRIOR)
FIVE_PARAM_CLF_UP_OPTIMIZER = LFModelOptimizer('pts_errs_15+22_up.npz', FiveParameterCLF, x0=BOUWENS_15_X0, prior=FIVE_PARAM_PRIOR)
FIVE_PARAM_CLF_DOWN_OPTIMIZER = LFModelOptimizer('pts_errs_15+22_down.npz', FiveParameterCLF, x0=BOUWENS_15_X0, prior=FIVE_PARAM_PRIOR)
FIVE_PARAM_CLF_2122_OPTIMIZER = LFModelOptimizer('pts_errs_21+22.npz', FiveParameterCLF, x0=BOUWENS_15_X0, prior=FIVE_PARAM_PRIOR)

optimizer = FIVE_PARAM_CLF_1522_OPTIMIZER
optimizer.optimize()
with open('fiducial_1522.pickle', 'wb') as handle:
    pickle.dump(optimizer.scipy_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
mcmc(optimizer, 'fiducial_1522.h5')

optimizer = FIVE_PARAM_CLF_15_OPTIMIZER
optimizer.optimize()
with open('fiducial_15.pickle', 'wb') as handle:
    pickle.dump(optimizer.scipy_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
mcmc(optimizer, 'fiducial_15.h5')

optimizer = FIVE_PARAM_CLF_UP_OPTIMIZER
optimizer.optimize()
with open('shifted_up.pickle', 'wb') as handle:
    pickle.dump(optimizer.scipy_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
mcmc(optimizer, 'shifted_up.h5')

optimizer = FIVE_PARAM_CLF_DOWN_OPTIMIZER
optimizer.optimize()
with open('shifted_down.pickle', 'wb') as handle:
    pickle.dump(optimizer.scipy_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
mcmc(optimizer, 'shifted_down.h5')

optimizer = FIVE_PARAM_CLF_2122_OPTIMIZER
optimizer.optimize()
with open('fiducial_2122.pickle', 'wb') as handle:
    pickle.dump(optimizer.scipy_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
mcmc(optimizer, 'fiducial_2122.h5')

