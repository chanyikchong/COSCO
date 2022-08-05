"""
Local Regression (LOESS) estimation routine with optional 
iterative robust estimation procedure. Setting `robustify=True` 
indicates that the robust estimation procedure should be 
performed. 
"""
import numpy as np
import pandas as pd
import scipy
import warnings

warnings.simplefilter("ignore")


def loc_eval(x, b):
    """
    Evaluate `x` using locally-weighted regression parameters.
    Degree of polynomial used in loess is inferred from b. `x`
    is assumed to be a scalar.
    """
    loc_est = 0
    for i in enumerate(b):
        loc_est += i[1] * (x ** i[0])
    return loc_est


def loess(x_vals, y_vals, alpha, poly_degree=1, robustify=False):
    """
    Perform locally-weighted regression via x_vals & y_vals.
    Variables used within `loess` function:

        n         => number of data points in xvals
        m         => nbr of LOESS evaluation points
        q         => number of data points used for each
                     locally-weighted regression
        v         => x-value locations for evaluating LOESS
        locs_df    => contains local regression details for each
                     location v
        eval_df    => contains actual LOESS output for each v
        X         => n-by-(poly_degree+1) design matrix
        W         => n-by-n diagonal weight matrix for each
                     local regression
        y         => yvals
        b         => local regression coefficient estimates.
                     b = `(X^T*W*X)^-1*X^T*W*y`. Note that `@`
                     replaces np.dot in recent numpy versions.
        local_est => response for local regression
    """
    # sort dataset by x_vals:
    all_data = sorted(zip(x_vals, y_vals), key=lambda x: x[0])
    x_vals, y_vals = zip(*all_data)

    locs_df = pd.DataFrame(
        columns=[
            'loc', 'x', 'weights', 'v', 'y', 'raw_dists',
            'scale_factor', 'scaled_dists'
        ])
    eval_df = pd.DataFrame(
        columns=[
            'loc', 'est', 'b', 'v', 'g'
        ])

    n = len(x_vals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(x_vals) - min(x_vals)) / len(x_vals))
    v_lb = max(0, min(x_vals) - (.5 * avg_interval))
    v_ub = (max(x_vals) + (.5 * avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)

    # Generate design matrix based on poly_degree.
    x_cols = [np.ones_like(x_vals)]
    for j in range(1, (poly_degree + 1)):
        x_cols.append([i ** j for i in x_vals])
    X = np.vstack(x_cols).T

    for i in v:
        iter_pos = i[0]
        iter_val = i[1]

        # Determine q-nearest xvals to iter_val.
        iter_dists = sorted([(j, np.abs(j - iter_val)) for j in x_vals], key=lambda x: x[1])

        _, raw_dists = zip(*iter_dists)

        # Scale local observations by qth-nearest raw_dist.
        scale_fact = raw_dists[q - 1]
        scaled_dists = [(j[0], (j[1] / scale_fact)) for j in iter_dists]
        weights = [(j[0], ((1 - np.abs(j[1] ** 3)) ** 3 if j[1] <= 1 else 0)) for j in scaled_dists]

        # Remove xvals from each tuple:
        _, weights = zip(*sorted(weights, key=lambda x: x[0]))
        _, raw_dists = zip(*sorted(iter_dists, key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists, key=lambda x: x[0]))

        iter_df1 = pd.DataFrame({
            'loc': iter_pos,
            'x': x_vals,
            'v': iter_val,
            'weights': weights,
            'y': y_vals,
            'raw_dists': raw_dists,
            'scale_fact': scale_fact,
            'scaled_dists': scaled_dists
        })

        locs_df = pd.concat([locs_df, iter_df1])
        W = np.diag(weights)
        y = y_vals
        b = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
        local_est = loc_eval(iter_val, b)

        iter_df2 = pd.DataFrame({
            'loc': [iter_pos],
            'b': [b],
            'v': [iter_val],
            'g': [local_est]
        })

        eval_df = pd.concat([eval_df, iter_df2])

    # Reset indicies for returned DataFrames.
    locs_df.reset_index(inplace=True)
    locs_df.drop('index', axis=1, inplace=True)
    locs_df['est'] = 0
    eval_df['est'] = 0
    locs_df = locs_df[['loc', 'est', 'v', 'x', 'y', 'raw_dists',
                       'scale_fact', 'scaled_dists', 'weights']]

    if robustify:

        cycle_nbr = 1
        robust_est = [eval_df]

        while True:
            # Perform iterative robustness procedure for each local regression.
            # Evaluate local regression for each item in xvals.
            #
            # e1_i => raw residuals
            # e2_i => scaled residuals
            # r_i  => robustness weight
            reval_df = pd.DataFrame(
                columns=['loc', 'est', 'v', 'b', 'g']
            )

            for i in robust_est[-1]['loc']:
                prev_df = robust_est[-1]
                loc_df = locs_df[locs_df['loc'] == i]
                b_i = prev_df.loc[prev_df['loc'] == i, 'b'].item()
                w_i = loc_df['weights']
                v_i = prev_df.loc[prev_df['loc'] == i, 'v'].item()
                g_i = prev_df.loc[prev_df['loc'] == i, 'g'].item()
                e1_i = [k - loc_eval(j, b_i) for (j, k) in zip(x_vals, y_vals)]
                e2_i = [j / (6 * np.median(np.abs(e1_i))) for j in e1_i]
                r_i = [(1 - np.abs(j ** 2)) ** 2 if np.abs(j) < 1 else 0 for j in e2_i]
                w_f = [j * k for (j, k) in zip(w_i, r_i)]  # new weights
                W_r = np.diag(w_f)
                b_r = np.linalg.pinv(X.T @ W_r @ X) @ (X.T @ W_r @ y)
                r_iter_est = loc_eval(v_i, b_r)

                r_iter_df = pd.DataFrame({
                    'loc': [i],
                    'b': [b_r],
                    'v': [v_i],
                    'g': [r_iter_est],
                    'est': [cycle_nbr]
                })

                reval_df = pd.concat([reval_df, r_iter_df])
            robust_est.append(reval_df)

            # Compare `g` vals from two latest reval_df's in robust_est.
            i_diffs = np.abs((robust_est[-2]["g"] - robust_est[-1]["g"]) / robust_est[-2]["g"])

            if np.all(i_diffs < .05) or cycle_nbr > 20:
                break

            cycle_nbr += 1

        # Vertically bind all DataFrames from robust_est.
        eval_df = pd.concat(robust_est)

    eval_df.reset_index(inplace=True)
    eval_df.drop('index', axis=1, inplace=True)
    eval_df = eval_df[['loc', 'est', 'v', 'b', 'g']]

    return locs_df, eval_df
