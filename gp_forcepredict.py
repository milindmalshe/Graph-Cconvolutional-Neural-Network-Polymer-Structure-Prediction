import numpy as np
import sklearn.gaussian_process as gp


##GP model starts here:
def gp_predict(xp, yp, xe, alpha=1e-5):


    # Create the GP

    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)


    model.fit(xp, yp)

    # predict next sample


    mu, sigma = model.predict(xe, return_std=True)



    return mu, sigma




def apply_GP(force_file, opt_e, d1_e, d2_e):

    #setting up test features
    xe = np.array([opt_e, d1_e, d2_e])

    try:
        X = np.loadtxt(force_file)
        xp = X[:, :-1]
        yp = X[:, -1]


        #if the size of dataset is minimum threshold (beyond 10 points): try Bayesian

        if len(xp) >= 10:
            x = xp.copy()
            x = (x - x.mean(axis=0))/(x.std(axis=0))

            # train and predict GP
            mu_y, sigma_y = gp_predict(xp=x, yp=yp, xe=xe)
            y_lo, y_hi = cox_intervals(Y_bar=mu_y, S=sigma_y, n=len(x))



        else:
            y_lo = 0
            y_hi = 0.7

    except:
        y_lo = 0
        y_hi = 0.7



    return y_lo, y_hi



def cox_intervals(Y_bar, S, n):


    Y_lo = Y_bar + 0.5*(S**2) - (0.5*(S**2) + (S**4)/(2*(n-1)))**(0.5)
    Y_hi = Y_bar + 0.5*(S**2) + (0.5*(S**2) + (S**4)/(2*(n-1)))**(0.5)



    return Y_lo, Y_hi


def convert_to_lambda(F1, F2):

    l = - np.log(float(F2)/float(F1) - 1)

    return l



def gp_refcurve(x_t, y_t, x_e, alpha=1e-5):

    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel=kernel,
                                        alpha=alpha,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)

    model.fit(x_t, y_t)

    # predict next sample


    mu, sigma = model.predict(x_e, return_std=True)



    return mu, sigma


def prep_features(f, f_min0, f_approx, t, tau_max):


    f_star = (f - f_min0)/f_approx
    tau = t/tau_max

    X = np.vstack((f_star, tau))

    return X

Y1, Y2 = cox_intervals(-0.5, 0.2, 2)
#print np.exp(Y1), np.exp(Y2)