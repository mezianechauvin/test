



def old_pymc_paid_total_model():

    # https://juanitorduz.github.io/pymc_mmm/
    # https://github.com/juanitorduz/website_projects/blob/master/Python/pymc_mmm.ipynb
    
    
    # modeling
    # https://juanitorduz.github.io/pymc_mmm/
    def geometric_adstock(x, alpha: float = 0.0, l_max: int = 12):
        """Geometric adstock transformation."""
        cycles = [
            pt.concatenate(
                [pt.zeros(i), x[: x.shape[0] - i]]
            )
            for i in range(l_max)
        ]
        x_cycle = pt.stack(cycles)
        w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
        return pt.dot(w, x_cycle)


    def logistic_saturation(x, lam: float = 0.5):
        """Logistic saturation transformation."""
        return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))


    def geometric_adstock_vectorized(x, alpha, l_max: int = 12):
        """Vectorized geometric adstock transformation."""
        cycles = [
            pt.concatenate(tensor_list=[pt.zeros(shape=x.shape)[:i], x[: x.shape[0] - i]])
            for i in range(l_max)
        ]
        x_cycle = pt.stack(cycles)
        x_cycle = pt.transpose(x=x_cycle, axes=[1, 2, 0])
        w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
        w = pt.transpose(w)[None, ...]
        return pt.sum(pt.mul(x_cycle, w), axis=2)

    # get data ready
    # data_df = gs_client.read_dataframe('us_agg_mel_mac', header_row_num=0)
    mel_mac_agg_df
    data_df = mel_mac_agg_df.copy(deep=True)
    float_cols = ['channel_mel_spend', 'channel_mel_agency_fees', 'channel_total_spend', 'n_conversion', 'channel_cac']
    for float_col in float_cols:
        data_df[float_col] = data_df[float_col].astype(float)

    # data_df = data_df.rename(columns={'min_date': 'date'})
    # data_df['date'] = pd.to_datetime(data_df['date'])
    # data_df['year'] = data_df['date'].dt.year    
    # data_df['n_week'] = data_df['iso_year_week'].str[-2:].astype(int)
    # data_df = data_df[data_df['year'] >= 2021].sort_values('date', ascending=True).reset_index(drop=True)
    # data_df['index'] = data_df.index
    data_df

    df = data_df[['index', 'date', 'n_week', 'year', 'channel_total_spend', 'n_conversion']].copy(deep=True).reset_index(drop=True)
    df

    df.groupby('year').size()

    # feature engineering
    t = ((df.index - df.index.min()) / (df.index.max() - df.index.min())).values
    
    n_order = 7
    periods = df['n_week'] / 52
    
    periods
    df['n_week']
    
    fourier_features = pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    
    fourier_features
    

    df.shape
    t.shape
    fourier_features.shape
    
    date = df['date'].to_numpy()
    date_index = df.index
    y = df['n_conversion'].to_numpy()
    z = df['channel_total_spend'].to_numpy()
    n_obs = y.size

    endog_scaler = MaxAbsScaler()
    endog_scaler.fit(y.reshape(-1, 1))
    y_scaled = endog_scaler.transform(y.reshape(-1, 1)).flatten()

    channel_scaler = MaxAbsScaler()
    channel_scaler.fit(z.reshape(-1, 1))
    z_scaled = channel_scaler.transform(z.reshape(-1, 1)).flatten()

    coords = {"date": date, "fourier_mode": np.arange(2 * n_order)}
    
    palette = "viridis_r"
    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 100)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))


    ##### start simple linear #####

    with pm.Model(coords=coords) as base_model:
        # --- coords ---
        base_model.add_coord(name="dat", values=date, mutable=True)
        base_model.add_coord(name="fourier_mode", values=np.arange(2 * n_order), mutable=False)

        # --- data containers ---
        z_scaled_ = pm.MutableData(name="z_scaled", value=z_scaled, dims="date")

        # --- priors ---
        ## intercept
        a = pm.Normal(name="a", mu=0, sigma=4)
        ## trend
        b_trend = pm.Normal(name="b_trend", mu=0, sigma=2)
        ## seasonality
        b_fourier = pm.Laplace(name="b_fourier", mu=0, b=2, dims="fourier_mode")
        ## regressor
        b_z = pm.HalfNormal(name="b_z", sigma=2)
        ## standard deviation of the normal likelihood
        sigma = pm.HalfNormal(name="sigma", sigma=0.5)
        # degrees of freedom of the t distribution
        nu = pm.Gamma(name="nu", alpha=25, beta=2)

        # --- model parametrization ---
        trend = pm.Deterministic(name="trend", var=a + b_trend * t, dims="date")
        seasonality = pm.Deterministic(
            name="seasonality", var=pm.math.dot(fourier_features, b_fourier), dims="date"
        )
        z_effect = pm.Deterministic(name="z_effect", var=b_z * z_scaled_, dims="date")
        mu = pm.Deterministic(name="mu", var=trend + seasonality + z_effect, dims="date")

        # --- likelihood ---
        pm.StudentT(name="likelihood", nu=nu, mu=mu, sigma=sigma, observed=y_scaled, dims="date")

        # --- prior samples ---
        base_model_prior_predictive = pm.sample_prior_predictive()

    # graph = pm.model_to_graphviz(model=base_model)
    # graph.view()

    
    # view the model
    if plots_action in ('show', 'save'):
        fig, ax = plt.subplots()

        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(base_model_prior_predictive.prior_predictive["likelihood"], p, axis=1)
            lower = np.percentile(
                base_model_prior_predictive.prior_predictive["likelihood"], 100 - p, axis=1
            )
            color_val = colors[i]
            ax.fill_between(
                x=date,
                y1=upper.flatten(),
                y2=lower.flatten(),
                color=cmap(color_val),
                alpha=0.1,
            )

        sns.lineplot(x=date, y=y_scaled, color="black", label="target (scaled)", ax=ax)
        ax.legend()
        ax.set(title="Base Model - Prior Predictive Samples")
        plt.show()

    # train model
    with base_model:
        base_model_trace = pm.sample(
            nuts_sampler="numpyro",
            draws=6_000,
            chains=4,
            idata_kwargs={"log_likelihood": True},
        )
        base_model_posterior_predictive = pm.sample_posterior_predictive(
            trace=base_model_trace
        )
        
    print(6_000 * 4)
    # 6_000 * 4 = 24_000

    # summary
    az.summary(
        data=base_model_trace,
        var_names=["a", "b_trend", "b_z", "sigma", "nu"],
    )
    
    posterior_predictive_likelihood = az.extract(
        data=base_model_posterior_predictive,
        group="posterior_predictive",
        var_names="likelihood",
    )

    posterior_predictive_likelihood_inv = endog_scaler.inverse_transform(X=posterior_predictive_likelihood)
    
    posterior_predictive_likelihood
    posterior_predictive_likelihood_inv.shape
    posterior_predictive_likelihood_inv.mean(axis=1).shape
    
    model_hdi = az.hdi(ary=base_model_trace)
    model_hdi['z_effect'].shape
    model_hdi['trend'].shape
    model_hdi['seasonality'].shape
    base_model_trace.posterior['z_effect'].mean(dim=('chain', 'draw'))
    # endog_scaler.inverse_transform(
    base_model_trace.posterior['trend'].mean(dim=('chain', 'draw'))
    base_model_trace.posterior['seasonality'].mean(dim=('chain', 'draw'))
    
    z_effect_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        base_model_trace.posterior["z_effect"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    trend_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        base_model_trace.posterior["trend"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    seasonality_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        base_model_trace.posterior["seasonality"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    likelihood_posterior_samples = endog_scaler.inverse_transform(
            X=az.extract(
                data=base_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    
    base_z_effect_mean = z_effect_posterior_samples.mean(dim=("chain", "draw"))
    z_effect_hdi = az.hdi(ary=z_effect_posterior_samples)["z_effect"]
    base_trend_mean = trend_posterior_samples.mean(dim=("chain", "draw"))
    trend_hdi = az.hdi(ary=trend_posterior_samples)["trend"]
    base_seasonality_mean = seasonality_posterior_samples.mean(dim=("chain", "draw"))
    seasonality_hdi = az.hdi(ary=seasonality_posterior_samples)["seasonality"]
    base_likelihood_mean = likelihood_posterior_samples.mean(axis=1)
    base_likelihood_mean.shape
    
    base_res_df = pd.DataFrame({'base_z_effect_mean': base_z_effect_mean, 'base_trend_mean': base_trend_mean, 'base_seasonality_mean': base_seasonality_mean, 
                  'base_likelihood_mean': base_likelihood_mean})
    
    base_res_df['total_mean'] = base_res_df[['base_z_effect_mean', 'base_trend_mean', 'base_seasonality_mean']].sum(axis=1)
    base_res_df['n_conversion'] = mel_mac_agg_df['n_conversion']
    base_res_df.to_clipboard(index=False)
    
    # roas / cac
    base_model_trace_roas = base_model_trace.copy()

    with base_model:
        pm.set_data(new_data={"z_scaled": np.zeros_like(a=z_scaled)})
        base_model_trace_roas.extend(
            other=pm.sample_posterior_predictive(trace=base_model_trace_roas, var_names=["likelihood"])
        )
    
    base_roas_numerator = (
        endog_scaler.inverse_transform(
            X=az.extract(
                data=base_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
        - endog_scaler.inverse_transform(
            X=az.extract(
                data=base_model_trace_roas,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    ).sum(axis=0)
    roas_denominator = z.sum()
    base_roas = roas_denominator / base_roas_numerator
    
    base_roas_mean = base_roas.mean()
    base_roas_mean
    base_roas_hdi = az.hdi(ary=base_roas)
    base_roas_hdi
    
    # marginal CAC
    eta = -0.10
    eta

    base_model_trace_mroas = base_model_trace.copy()

    with base_model:
        pm.set_data(new_data={"z_scaled": (1 + eta) * z_scaled})
        base_model_trace_mroas.extend(
            other=pm.sample_posterior_predictive(trace=base_model_trace_mroas, var_names=["likelihood"])
        )

    base_mroas_numerator = (
        endog_scaler.inverse_transform(
            X=az.extract(
                data=base_model_trace_mroas,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
        - endog_scaler.inverse_transform(
            X=az.extract(
                data=base_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    ).sum(axis=0)

    mroas_denominator = eta * z.sum()

    base_mroas = mroas_denominator / base_mroas_numerator

    base_mroas_mean = base_mroas.mean()
    base_mroas_hdi = az.hdi(ary=base_mroas)
    base_mroas_mean
    base_mroas_hdi
    
    print(f'base model cac: ${base_roas_mean:.0f} and base model marginal cac with {eta:.0%} increase: ${base_mroas_mean:.0f}')

    
    if plots_action in ('show', 'save'):

        axes = az.plot_trace(
            data=base_model_trace,
            var_names=["a", "b_trend", "b_fourier", "b_z", "sigma", "nu"],
            compact=True,
            backend_kwargs={
                "figsize": (12, 9),
                "layout": "constrained"
            },
        )
        fig = axes[0][0].get_figure()
        fig.suptitle("Base Model - Trace")
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 4))
        az.plot_forest(
            data=base_model_trace,
            var_names=["a", "b_trend", "b_z", "sigma"],
            combined=True,
            ax=ax
        )
        ax.set(
            title="Base Model: 94.0% HDI",
            xscale="log"
        )
        plt.show()

        fig, ax = plt.subplots()

        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(posterior_predictive_likelihood_inv, p, axis=1)
            lower = np.percentile(posterior_predictive_likelihood_inv, 100 - p, axis=1)
            color_val = colors[i]
            ax.fill_between(
                x=date,
                y1=upper,
                y2=lower,
                color=cmap(color_val),
                alpha=0.1,
            )

        sns.lineplot(
            x=date,
            y=posterior_predictive_likelihood_inv.mean(axis=1),
            color="C2",
            label="posterior predictive mean",
            ax=ax,
        )
        sns.lineplot(
            x=date,
            y=y,
            color="black",
            label="target (scaled)",
            ax=ax,
        )
        ax.legend(loc="upper left")
        ax.set(title="Base Model - Posterior Predictive Samples")
        plt.show()

        fig, ax = plt.subplots()

        for i, var_effect in enumerate(["z_effect", "trend", "seasonality"]):
            ax.fill_between(
                x=date,
                y1=model_hdi[var_effect][:, 0],
                y2=model_hdi[var_effect][:, 1],
                color=f"C{i}",
                alpha=0.3,
                label=f"$94\%$ HDI ({var_effect})",
            )
            sns.lineplot(
                x=date,
                y=base_model_trace.posterior[var_effect]
                .stack(sample=("chain", "draw"))
                .mean(axis=1),
                color=f"C{i}",
            )

        sns.lineplot(x=date, y=y_scaled, color="black", alpha=1.0, label="target (scaled)", ax=ax)
        ax.legend(title="components", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title="Base Model Components", ylabel="target (scaled)")
        plt.show()


        fig, ax = plt.subplots()
        ax.fill_between(
            x=date,
            y1=z_effect_hdi[:, 0],
            y2=z_effect_hdi[:, 1],
            color="C0",
            alpha=0.5,
            label="z_effect 94% HDI",
        )
        ax.axhline(
            y=z_effect_posterior_samples.mean(),
            color="C0",
            linestyle="--",
            label=f"posterior mean {z_effect_posterior_samples.mean().values: 0.3f}",
        )
        # sns.lineplot(x="date", y="z_effect", color="C3", data=data_df, label="z_effect", ax=ax)
        ax.legend(loc="upper right")
        ax.set(title="Media Cost Effect on Sales Estimation - Base Model")
        plt.show()


        fig, ax = plt.subplots()

        az.plot_hdi(
            x=z,
            y=z_effect_posterior_samples,
            color="C0",
            fill_kwargs={"alpha": 0.2, "label": "z_effect 94% HDI"},
            ax=ax,
        )
        sns.scatterplot(
            x="z",
            y="z_effect_pred_mean",
            color="C0",
            size="index",
            label="z_effect (pred mean)",
            data=df.assign(
                z_effect_pred_mean=z_effect_posterior_samples.mean(dim=("chain", "draw"))
            ),
            ax=ax,
        )
        # sns.scatterplot(
        #     x="z",
        #     y="z_effect",
        #     color="C3",
        #     size="index",
        #     label="z_effect (true)",
        #     data=data_df,
        #     ax=ax,
        # )
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h[:9], labels=l[:9], loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title="Base  Model - Estimated Effect");
        plt.show()

        g = sns.displot(x=base_roas, kde=True, height=5, aspect=1.5)
        ax = g.axes.flatten()[0]
        ax.axvline(
            x=base_roas_mean, color="C0", linestyle="--", label=f"mean = {base_roas_mean: 0.3f}"
        )
        ax.axvline(
            x=base_roas_hdi[0],
            color="C1",
            linestyle="--",
            label=f"HDI_lwr = {base_roas_hdi[0]: 0.3f}",
        )
        ax.axvline(
            x=base_roas_hdi[1],
            color="C2",
            linestyle="--",
            label=f"HDI_upr = {base_roas_hdi[1]: 0.3f}",
        )
        ax.axvline(x=roas_true, color="black", linestyle="--", label=f"true = {roas_true: 0.3f}")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title="Base Model mel CAC")
        plt.show()

        g = sns.displot(x=base_mroas, kde=True, height=5, aspect=1.5)
        ax = g.axes.flatten()[0]
        # ax.axvline(
        #     x=base_mroas_mean, color="C0", linestyle="--", label=f"mean = {base_mroas_mean: 0.3f}"
        # )
        # ax.axvline(
        #     x=base_mroas_hdi[0],
        #     color="C1",
        #     linestyle="--",
        #     label=f"HDI_lwr = {base_mroas_hdi[0]: 0.3f}",
        # )
        # ax.axvline(
        #     x=base_mroas_hdi[1],
        #     color="C2",
        #     linestyle="--",
        #     label=f"HDI_upr = {base_roas_hdi[1]: 0.3f}",
        # )
        # ax.axvline(x=0.0, color="gray", linestyle="--", label="zero")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title=f"Base Model MROAS ({eta:.0%} increase)")
        plt.show()


    ##### add adstock and saturation #####
    # apply saturation first as our spend is heavily conetrated in a single time period

    with pm.Model(coords=coords) as adstock_saturation_model:
        # --- data containers ---
        z_scaled_ = pm.MutableData(name="z_scaled", value=z_scaled, dims="date")
        
        # --- priors ---
        ## intercept
        a = pm.Normal(name="a", mu=0, sigma=4)
        ## trend
        b_trend = pm.Normal(name="b_trend", mu=0, sigma=2)
        ## seasonality
        b_fourier = pm.Laplace(name="b_fourier", mu=0, b=2, dims="fourier_mode")
        ## adstock effect
        alpha = pm.Beta(name="alpha", alpha=1, beta=1)
        ## saturation effect
        lam = pm.Gamma(name="lam", alpha=3, beta=1)
        ## regressor
        b_z = pm.HalfNormal(name="b_z", sigma=2)
        ## standard deviation of the normal likelihood
        sigma = pm.HalfNormal(name="sigma", sigma=0.5)
        # degrees of freedom of the t distribution
        nu = pm.Gamma(name="nu", alpha=25, beta=2)

        # --- model parametrization ---
        trend = pm.Deterministic("trend", a + b_trend * t, dims="date")
        seasonality = pm.Deterministic(
            name="seasonality", var=pm.math.dot(fourier_features, b_fourier), dims="date"
        )
        z_adstock = pm.Deterministic(
            name="z_adstock", var=geometric_adstock(x=z_scaled_, alpha=alpha, l_max=12), dims="date"
        )
        z_adstock_saturated = pm.Deterministic(
            name="z_adstock_saturated",
            var=logistic_saturation(x=z_adstock, lam=lam),
            dims="date",
        )
        z_effect = pm.Deterministic(
            name="z_effect", var=b_z * z_adstock_saturated, dims="date"
        )
        mu = pm.Deterministic(name="mu", var=trend + seasonality + z_effect, dims="date")

        # --- likelihood ---
        pm.StudentT(name="likelihood", nu=nu, mu=mu, sigma=sigma, observed=y_scaled, dims="date")

        # --- prior samples
        adstock_saturation_model_prior_predictive = pm.sample_prior_predictive()

    # pm.model_to_graphviz(model=adstock_saturation_model)

    # fit
    with adstock_saturation_model:
        adstock_saturation_model_trace = pm.sample(
            nuts_sampler="numpyro",
            draws=6_000,
            chains=4,
            idata_kwargs={"log_likelihood": True},
        )
        adstock_saturation_model_posterior_predictive = pm.sample_posterior_predictive(
            trace=adstock_saturation_model_trace,
        )


    az.summary(
        data=adstock_saturation_model_trace,
        var_names=["a", "b_trend", "b_z", "alpha", "lam", "sigma", "nu"]
    )
    
    lam_mean = adstock_saturation_model_trace.posterior['lam'].mean()
    lam_std = adstock_saturation_model_trace.posterior['lam'].std()
    
    lams = np.array([lam_mean - lam_std, lam_mean, lam_mean + lam_std])
    x = np.linspace(0, 5, 100)
    ax = plt.subplot(111)
    for l in lams:
        y = logistic_saturation(x, lam=l).eval()
        plt.plot(x, y, label=f'lam = {l}')
    plt.xlabel('spend', fontsize=12)
    plt.ylabel('f(spend)', fontsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    posterior_predictive_likelihood = az.extract(
        data=adstock_saturation_model_posterior_predictive,
        group="posterior_predictive",
        var_names="likelihood",
    )

    posterior_predictive_likelihood_inv = endog_scaler.inverse_transform(
        X=posterior_predictive_likelihood
    )
    
    # compute HDI for all the model parameters
    model_hdi = az.hdi(ary=adstock_saturation_model_trace)
    
    z_effect_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        adstock_saturation_model_trace.posterior["z_effect"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    trend_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        adstock_saturation_model_trace.posterior["trend"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    seasonality_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        adstock_saturation_model_trace.posterior["seasonality"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    likelihood_posterior_samples = endog_scaler.inverse_transform(
            X=az.extract(
                data=adstock_saturation_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    
    as_z_effect_mean = z_effect_posterior_samples.mean(dim=("chain", "draw"))
    z_effect_hdi = az.hdi(ary=z_effect_posterior_samples)["z_effect"]
    as_trend_mean = trend_posterior_samples.mean(dim=("chain", "draw"))
    trend_hdi = az.hdi(ary=trend_posterior_samples)["trend"]
    as_seasonality_mean = seasonality_posterior_samples.mean(dim=("chain", "draw"))
    seasonality_hdi = az.hdi(ary=seasonality_posterior_samples)["seasonality"]
    as_likelihood_mean = likelihood_posterior_samples.mean(axis=1)
    as_likelihood_mean.shape
    
    as_res_df = pd.DataFrame({'as_z_effect_mean': as_z_effect_mean, 'as_trend_mean': as_trend_mean, 'as_seasonality_mean': as_seasonality_mean,
                    'as_likelihood_mean': as_likelihood_mean})
    as_res_df['total_mean'] = as_res_df[['as_z_effect_mean', 'as_trend_mean', 'as_seasonality_mean']].sum(axis=1)
    as_res_df['n_conversion'] = mel_mac_agg_df['n_conversion']
    as_res_df.to_clipboard(index=False)
    
    
    z_effect_hdi = az.hdi(ary=z_effect_posterior_samples)["z_effect"]
    z_effect_hdi.shape
    
    adstock_saturation_model_trace_roas = adstock_saturation_model_trace.copy()

    with adstock_saturation_model:
        pm.set_data(new_data={"z_scaled": np.zeros_like(a=z_scaled)})
        adstock_saturation_model_trace_roas.extend(
            other=pm.sample_posterior_predictive(trace=adstock_saturation_model_trace_roas, var_names=["likelihood"])
        )
        
    adstock_saturation_roas_numerator = (
        endog_scaler.inverse_transform(
            X=az.extract(
                data=adstock_saturation_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
        - endog_scaler.inverse_transform(
            X=az.extract(
                data=adstock_saturation_model_trace_roas,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    ).sum(axis=0)
    roas_denominator = z.sum()
    adstock_saturation_roas = roas_denominator / adstock_saturation_roas_numerator
    adstock_saturation_roas_mean = adstock_saturation_roas.mean()
    adstock_saturation_roas_hdi = az.hdi(ary=adstock_saturation_roas)
    adstock_saturation_roas_mean
    adstock_saturation_roas_hdi
    
    eta = -0.10

    adstock_saturation_model_trace_mroas = adstock_saturation_model_trace.copy()

    with adstock_saturation_model:
        pm.set_data(new_data={"z_scaled": (1 + eta) * z_scaled})
        adstock_saturation_model_trace_mroas.extend(
            other=pm.sample_posterior_predictive(trace=adstock_saturation_model_trace_mroas, var_names=["likelihood"])
        )

    adstock_saturation_mroas_numerator = (
        endog_scaler.inverse_transform(
            X=az.extract(
                data=adstock_saturation_model_trace_mroas,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
        - endog_scaler.inverse_transform(
            X=az.extract(
                data=adstock_saturation_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    ).sum(axis=0)
    mroas_denominator = eta * z.sum()
    adstock_saturation_mroas = mroas_denominator / adstock_saturation_mroas_numerator
    adstock_saturation_mroas_mean = adstock_saturation_mroas.mean()
    adstock_saturation_mroas_hdi = az.hdi(ary=adstock_saturation_mroas)
    adstock_saturation_mroas_mean
    adstock_saturation_mroas_hdi
    
    print(f'base model cac: ${adstock_saturation_roas_mean:.0f} and base model marginal cac with {eta:.0%} increase: ${adstock_saturation_mroas_mean:.0f}')
    
    
    if plots_action in ('show', 'save'):

        fig, ax = plt.subplots()

        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(posterior_predictive_likelihood_inv, p, axis=1)
            lower = np.percentile(posterior_predictive_likelihood_inv, 100 - p, axis=1)
            color_val = colors[i]
            ax.fill_between(
                x=date,
                y1=upper,
                y2=lower,
                color=cmap(color_val),
                alpha=0.1,
            )

        sns.lineplot(
            x=date,
            y=posterior_predictive_likelihood_inv.mean(axis=1),
            color="C2",
            label="posterior predictive mean",
            ax=ax,
        )
        sns.lineplot(
            x=date,
            y=y,
            color="black",
            label="target",
            ax=ax,
        )
        ax.legend(loc="upper left")
        ax.set(title="Adstock-Saturation Model - Posterior Predictive Samples");
        plt.show()

        fig, ax = plt.subplots()

        for i, var_effect in enumerate(["z_effect", "trend", "seasonality"]):
            ax.fill_between(
                x=date,
                y1=model_hdi[var_effect][:, 0],
                y2=model_hdi[var_effect][:, 1],
                color=f"C{i}",
                alpha=0.3,
                label=f"$94\%$ HDI ({var_effect})",
            )
            sns.lineplot(
                x=date,
                y=adstock_saturation_model_trace.posterior[var_effect]
                .stack(sample=("chain", "draw"))
                .mean(axis=1),
                color=f"C{i}",
            )

        sns.lineplot(x=date, y=y_scaled, color="black", alpha=1.0, label="target (scaled)", ax=ax)
        ax.legend(title="components", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title="Adstock-Saturation Model Components", ylabel="target (scaled)");
        plt.show()

        fig, ax = plt.subplots()
        ax.fill_between(
            x=date,
            y1=z_effect_hdi[:, 0],
            y2=z_effect_hdi[:, 1],
            color="C0",
            alpha=0.5,
            label="z_effect 94% HDI",
        )
        ax.axhline(
            y=z_effect_posterior_samples.mean(),
            color="C0",
            linestyle="--",
            label=f"posterior mean {z_effect_posterior_samples.mean().values: 0.3f}",
        )
        # sns.lineplot(x="date", y="z_effect", color="C3", data=data_df, label="z_effect", ax=ax)
        ax.legend(loc="upper right")
        ax.set(title="Media Cost Effect on Sales Estimation - Adstock-Saturation");
        plt.show()

        fig, ax = plt.subplots()

        az.plot_hdi(
            x=z,
            y=z_effect_posterior_samples,
            color="C0",
            fill_kwargs={"alpha": 0.2, "label": "z_effect 94% HDI"},
            ax=ax,
        )
        sns.scatterplot(
            x="mel_spend",
            y="z_effect_pred_mean",
            color="C0",
            size="index",
            label="z_effect (pred mean)",
            data=df.assign(
                z_effect_pred_mean=z_effect_posterior_samples.mean(dim=("chain", "draw"))
            ),
            ax=ax,
        )
        # sns.scatterplot(
        #     x="z",
        #     y="z_effect",
        #     color="C3",
        #     size="index",
        #     label="z_effect (true)",
        #     data=data_df,
        #     ax=ax,
        # )
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h[:9], labels=l[:9], loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title="Adstock-Saturation  Model - Estimated Effect")
        plt.show()


        g = sns.displot(x=adstock_saturation_roas, kde=True, height=5, aspect=1.5)
        ax = g.axes.flatten()[0]
        ax.axvline(
            x=adstock_saturation_roas_mean, color="C0", linestyle="--", label=f"mean = {adstock_saturation_roas_mean: 0.3f}"
        )
        ax.axvline(
            x=adstock_saturation_roas_hdi[0],
            color="C1",
            linestyle="--",
            label=f"HDI_lwr = {adstock_saturation_roas_hdi[0]: 0.3f}",
        )
        ax.axvline(
            x=adstock_saturation_roas_hdi[1],
            color="C2",
            linestyle="--",
            label=f"HDI_upr = {adstock_saturation_roas_hdi[1]: 0.3f}",
        )
        ax.axvline(x=roas_true, color="black", linestyle="--", label=f"true = {roas_true: 0.3f}")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title="Adstock Saturation Model ROAS")
        plt.show()

        b_z_true = (np.arange(start=0.0, stop=1.0, step=1/n_obs) + 1) ** (-1.8)

        z_effect_eta = b_z_true * logistic_saturation(
            x=geometric_adstock(x=(1 + eta) * z, alpha=0.5, l_max=12),
            lam=0.15
        ).eval()

        g = sns.displot(x=adstock_saturation_mroas, kde=True, height=5, aspect=1.5)
        ax = g.axes.flatten()[0]
        ax.axvline(
            x=adstock_saturation_mroas_mean, color="C0", linestyle="--", label=f"mean = {adstock_saturation_mroas_mean: 0.3f}"
        )
        ax.axvline(
            x=adstock_saturation_mroas_hdi[0],
            color="C1",
            linestyle="--",
            label=f"HDI_lwr = {adstock_saturation_mroas_hdi[0]: 0.3f}",
        )
        ax.axvline(
            x=adstock_saturation_mroas_hdi[1],
            color="C2",
            linestyle="--",
            label=f"HDI_upr = {adstock_saturation_roas_hdi[1]: 0.3f}",
        )
        ax.axvline(x=mroas_true, color="black", linestyle="--", label=f"true = {roas_true: 0.3f}")
        ax.axvline(x=0.0, color="gray", linestyle="--", label="zero")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title=f"Adstock Saturation Model MROAS ({eta:.0%} increase)")
        plt.show()

    
    # add gaussian process seasonality
    z_scaled
    date
    coords.keys()
    coords
    len(np.arange(2 * n_order))
    # https://www.pymc.io/projects/examples/en/latest/fundamentals/data_container.html
    # https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-out-of-sample-predictions.html
    t.shape
    coords
    fourier_features

    with pm.Model() as asdr_model:
        # --- coords ---
        asdr_model.add_coord(name="date", values=date, mutable=True)
        asdr_model.add_coord(name="fourier_mode", values=np.arange(2 * n_order), mutable=False)
        
        # --- data containers ---
        z_scaled_ = pm.MutableData(name="z_scaled", value=z_scaled, dims="date")
        y_scaled_ = pm.MutableData(name="y_scaled", value=y_scaled, dims="date")
        t_ = pm.Data(name="t", value=t, dims="date")
        fourier_features_ = pm.MutableData(name="fourier_features", value=fourier_features, dims=("date", "fourier_mode"))

        # --- priors ---
        ## intercept
        a = pm.Normal(name="a", mu=0, sigma=4)
        ## trend
        b_trend = pm.Normal(name="b_trend", mu=0, sigma=2)
        ## seasonality
        b_fourier = pm.Laplace(name="b_fourier", mu=0, b=2, dims="fourier_mode")
        ## adstock effect
        alpha = pm.Beta(name="alpha", alpha=1, beta=1)
        ## saturation effect
        lam = pm.Gamma(name="lam", alpha=1, beta=1)
        ## gaussian random walk standard deviation
        sigma_slope = pm.HalfNormal(name="sigma_slope", sigma=0.05)
        ## standard deviation of the normal likelihood
        sigma = pm.HalfNormal(name="sigma", sigma=0.5)
        # degrees of freedom of the t distribution
        nu = pm.Gamma(name="nu", alpha=10, beta=1)

        # --- model parametrization ---
        trend = pm.Deterministic(name="trend", var=a + b_trend * t_, dims="date")
        seasonality = pm.Deterministic(
            name="seasonality", var=pm.math.dot(fourier_features_, b_fourier), dims="date"
        )
        slopes = pm.GaussianRandomWalk(
            name="slopes",
            sigma=sigma_slope,
            init_dist=pm.distributions.continuous.Normal.dist(
                name="init_dist", mu=0, sigma=2
            ),
            dims="date",
        )
        
        z_adstock = pm.Deterministic(
            name="z_adstock", var=geometric_adstock(x=z_scaled_, alpha=alpha, l_max=12), dims="date"
        )
        z_adstock_saturated = pm.Deterministic(
            name="z_adstock_saturated",
            var=logistic_saturation(x=z_adstock, lam=lam),
            dims="date",
        )
        z_effect = pm.Deterministic(
            name="z_effect", var=pm.math.exp(slopes) * z_adstock_saturated, dims="date"
        )
        mu = pm.Deterministic(name="mu", var=trend + seasonality + z_effect, dims="date")

        # --- likelihood ---
        pm.StudentT(name="likelihood", nu=nu, mu=mu, sigma=sigma, observed=y_scaled_, dims="date")

        # --- prior samples ---
        asdr_model_prior_predictive = pm.sample_prior_predictive()
    
    
    with asdr_model:
        asdr_model_trace = pm.sample(
            nuts_sampler="numpyro",
            draws=6_000,
            chains=4,
            idata_kwargs={"log_likelihood": True},
        )
        asdr_model_posterior_predictive = pm.sample_posterior_predictive(
            trace=asdr_model_trace
        )

    az.summary(
        data=asdr_model_trace,
        var_names=["a", "b_trend", "sigma_slope", "alpha", "lam", "sigma", "nu"]
    )
    
    posterior_predictive_likelihood = az.extract(
        data=asdr_model_posterior_predictive,
        group="posterior_predictive",
        var_names="likelihood",
    )

    posterior_predictive_likelihood_inv = endog_scaler.inverse_transform(
        X=posterior_predictive_likelihood
    )
    
    model_hdi = az.hdi(ary=asdr_model_trace)
    
    alpha_posterior = az.extract(data=asdr_model_trace, group="posterior", var_names="alpha")

    alpha_posterior_samples = alpha_posterior.to_numpy()[:100]
    alpha_posterior_samples

    # pass z through the adstock transformation
    geometric_adstock_posterior_samples = np.array([
        geometric_adstock(x=z, alpha=x).eval()
        for x in alpha_posterior_samples
    ])
    
    geometric_adstock_posterior_samples
    geometric_adstock_hdi = az.hdi(ary=geometric_adstock_posterior_samples)
    yerr = geometric_adstock_hdi[:, 1] - geometric_adstock_hdi[:, 0]
    
    lam_posterior = (
        az.extract(data=asdr_model_trace, group="posterior", var_names="lam")
        / channel_scaler.scale_.item()
    )
    lam_posterior_samples = lam_posterior.to_numpy()[:100]
    lam_posterior_samples
    
    
    
    lam_mean = asdr_model_trace['posterior']['lam'].mean()
    lam_std = asdr_model_trace['posterior']['lam'].std()
    
    lams = np.array([lam_mean - lam_std, lam_mean, lam_mean + lam_std])
    x = np.linspace(0, 5, 100)
    ax = plt.subplot(111)
    for l in lams:
        y = logistic_saturation(x, lam=l).eval()
        plt.plot(x, y, label=f'lam = {l}')
    plt.xlabel('spend', fontsize=12)
    plt.ylabel('f(spend)', fontsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    logistic_saturation_posterior_samples = np.array(
        [
            logistic_saturation(x=x, lam=lam_posterior_samples).eval()
            for x in data_df["z_adstock"].values
        ]
    )

    # We can now plot the estimated saturation curve against the true one.
    logistic_saturation_hdi = az.hdi(ary=logistic_saturation_posterior_samples.T)

    yerr = logistic_saturation_hdi[:, 1] - logistic_saturation_hdi[:, 0]
    
    logistic_saturation_posterior_samples.mean(axis=1)
    model_hdi_inv = az.hdi(ary=asdr_model_trace)
    
    z_effect_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        asdr_model_trace.posterior["z_effect"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    trend_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        asdr_model_trace.posterior["trend"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    seasonality_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        asdr_model_trace.posterior["seasonality"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    as_model_trace.posterior
    likelihood_posterior_samples = endog_scaler.inverse_transform(
            X=az.extract(
                data=asdr_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    
    asdr_z_effect_mean = z_effect_posterior_samples.mean(dim=("chain", "draw"))
    z_effect_hdi = az.hdi(ary=z_effect_posterior_samples)["z_effect"]
    asdr_trend_mean = trend_posterior_samples.mean(dim=("chain", "draw"))
    trend_hdi = az.hdi(ary=trend_posterior_samples)["trend"]
    asdr_seasonality_mean = seasonality_posterior_samples.mean(dim=("chain", "draw"))
    seasonality_hdi = az.hdi(ary=seasonality_posterior_samples)["seasonality"]
    asdr_likelihood_mean = likelihood_posterior_samples.mean(axis=1)
    asdr_likelihood_mean.shape
    
    asdr_res_df = pd.DataFrame({'asdr_z_effect_mean': asdr_z_effect_mean, 'asdr_trend_mean': asdr_trend_mean, 'asdr_seasonality_mean': asdr_seasonality_mean,
                    'asdr_likelihood_mean': asdr_likelihood_mean})
    asdr_res_df['total_mean'] = asdr_res_df[['asdr_z_effect_mean', 'asdr_trend_mean', 'asdr_seasonality_mean']].sum(axis=1)
    asdr_res_df['n_conversion'] = mel_mac_agg_df['n_conversion']
    asdr_res_df.to_clipboard(index=False)
    
    asdr_model_trace_roas = asdr_model_trace.copy()

    with asdr_model:
        pm.set_data(new_data={"z_scaled": np.zeros_like(a=z_scaled)})
        
        asdr_model_trace_roas.extend(
            other=pm.sample_posterior_predictive(trace=asdr_model_trace_roas, var_names=["likelihood"])
        )
    asdr_roas_numerator = (
        endog_scaler.inverse_transform(
            X=az.extract(
                data=asdr_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )

        )
        - endog_scaler.inverse_transform(
            X=az.extract(
                data=asdr_model_trace_roas,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    ).sum(axis=0)

    roas_denominator = z.sum()

    asdr_roas = roas_denominator / asdr_roas_numerator
    asdr_roas_mean = np.median(asdr_roas)
    asdr_roas_hdi = az.hdi(ary=asdr_roas)

    eta = -0.10

    asdr_model_trace_mroas = asdr_model_trace.copy()

    with asdr_model:
        pm.set_data(new_data={"z_scaled": (1 + eta) * z_scaled})
        asdr_model_trace_mroas.extend(
            other=pm.sample_posterior_predictive(trace=asdr_model_trace_mroas, var_names=["likelihood"])
        )
    asdr_mroas_numerator = (
        endog_scaler.inverse_transform(
            X=az.extract(
                data=asdr_model_trace_mroas,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
        - endog_scaler.inverse_transform(
            X=az.extract(
                data=asdr_model_posterior_predictive,
                group="posterior_predictive",
                var_names=["likelihood"],
            )
        )
    ).sum(axis=0)

    mroas_denominator = eta * z.sum()

    asdr_mroas = mroas_denominator / asdr_mroas_numerator
    asdr_mroas_mean = asdr_mroas.mean()
    asdr_mroas_hdi = az.hdi(ary=asdr_mroas)
    
    print(f'asdr model cac: ${asdr_roas_mean:.0f} and base model marginal cac with {eta:.0%} increase: ${asdr_mroas_mean:.0f}')
    
    
    if plots_action in ('show', 'save'):
        
        pm.model_to_graphviz(model=asdr_model)
        
        fig, ax = plt.subplots()

        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(
                asdr_model_prior_predictive.prior_predictive["likelihood"],
                p,
                axis=1,
            )
            lower = np.percentile(
                asdr_model_prior_predictive.prior_predictive["likelihood"], 100 - p, axis=1
            )
            color_val = colors[i]
            ax.fill_between(
                x=date,
                y1=upper.flatten(),
                y2=lower.flatten(),
                color=cmap(color_val),
                alpha=0.1,
            )

        sns.lineplot(x=date, y=y_scaled, color="black", label="target (scaled)", ax=ax)
        ax.legend()
        ax.set(title="Adstock-Saturation-Diminishing-Returns Model - Prior Predictive Samples")
        plt.show()

        axes = az.plot_trace(
            data=asdr_model_trace,
            var_names=[
                "a",
                "b_trend",
                "sigma_slope",
                "b_fourier",
                "alpha",
                "lam",
                "sigma",
                "nu",
            ],
            compact=True,
            backend_kwargs={"figsize": (12, 12), "layout": "constrained"},
        )
        fig = axes[0][0].get_figure()
        fig.suptitle("Adstock-Saturation-Diminishing-Returns Model - Trace")
        plt.show()


        fig, ax = plt.subplots(figsize=(6, 4))
        az.plot_forest(
            data=asdr_model_trace,
            var_names=["a", "b_trend", "sigma_slope", "alpha", "lam", "sigma", "nu"],
            combined=True,
            ax=ax
        )
        ax.set(
            title="Adstock-Saturation-Diminishing-Returns Model Model: 94.0% HDI",
            xscale="log"
        )
        plt.show()

        fig, ax = plt.subplots()

        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(posterior_predictive_likelihood_inv, p, axis=1)
            lower = np.percentile(posterior_predictive_likelihood_inv, 100 - p, axis=1)
            color_val = colors[i]
            ax.fill_between(
                x=date,
                y1=upper,
                y2=lower,
                color=cmap(color_val),
                alpha=0.1,
            )

        sns.lineplot(
            x=date,
            y=posterior_predictive_likelihood_inv.mean(axis=1),
            color="C2",
            label="posterior predictive mean",
            ax=ax,
        )
        sns.lineplot(
            x=date,
            y=y,
            color="black",
            label="target",
            ax=ax,
        )
        ax.legend(loc="upper left")
        ax.set(title="Adstock-Saturation-Diminishing-Returns Model - Posterior Predictive")
        plt.show()

        fig, ax = plt.subplots()

        for i, var_effect in enumerate(["z_effect", "trend", "seasonality"]):
            ax.fill_between(
                x=date,
                y1=model_hdi[var_effect][:, 0],
                y2=model_hdi[var_effect][:, 1],
                color=f"C{i}",
                alpha=0.3,
                label=f"$94\%$ HDI ({var_effect})",
            )
            sns.lineplot(
                x=date,
                y=asdr_model_trace.posterior[var_effect]
                .stack(sample=("chain", "draw"))
                .mean(axis=1),
                color=f"C{i}",
            )

        sns.lineplot(
            x=date, y=y_scaled, color="black", alpha=1.0, label="target (scaled)", ax=ax
        )
        ax.legend(title="components", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(
            title="Adstock-Saturation-Diminishing-Return Model Components",
            ylabel="target (scaled)",
        )
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 5))
        az.plot_pair(
            data=asdr_model_trace,
            var_names=["alpha", "lam"],
            kind="kde",
            divergences=True,
            ax=ax
        )
        # ax.axhline(lam_true_scaled, color="C1", linestyle="--", label="$\lambda_{true} (scaled)$")
        # ax.axvline(alpha_true, color="C4", linestyle="--", label="$\\alpha_{true}$")
        # ax.legend(title="true params", loc="upper right")
        ax.set(
            title="Adstock-Saturation-Diminishing-Returns Model",
            xlabel="$\\alpha$",
            ylabel="$\lambda$"
        )
        plt.show()


        corr, _ = pearsonr(
            x=asdr_model_trace.posterior["alpha"].stack(sample=("chain", "draw")).to_numpy(),
            y=asdr_model_trace.posterior["lam"].stack(sample=("chain", "draw")).to_numpy()
        )

        print(f"Correlation between alpha and lambda {corr: 0.3f}")

        fig, ax = plt.subplots(figsize=(8, 7))

        markers, caps, bars = ax.errorbar(
            x=data_df["z_adstock"], 
            y=geometric_adstock_posterior_samples.mean(axis=0), 
            yerr=yerr/2,
            color="C0",
            fmt='o',
            ms=1,
            capsize=5,
            label="$94\%$ HDI",
        )
        [bar.set_alpha(0.3) for bar in bars]
        ax.axline(
            xy1=(10, 10),
            slope=1.0,
            color="black",
            linestyle="--",
            label="diagonal"
        )
        ax.legend()
        ax.set(
            title="Adstock-Saturation-Diminishing-Returns Model - $\\alpha$ Estimation",
            xlabel="z_adstock (true)",
            ylabel="z_adstock (pred)",
        )
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 6))

        latex_function = r"$x\longmapsto \frac{1 - e^{-\lambda x}}{1 + e^{-\lambda x}}$"

        markers, caps, bars = ax.errorbar(
            x=data_df["z_adstock"], 
            y=logistic_saturation_posterior_samples.mean(axis=1), 
            yerr=yerr/2,
            color="C0",
            fmt='o',
            ms=3,
            capsize=5,
            label="$94\%$ HDI",
        )
        [bar.set_alpha(0.3) for bar in bars]
        sns.lineplot(
            x="z_adstock",
            y="z_adstock_saturated",
            color="C2",
            label=latex_function,
            data=data_df,
            ax=ax
        )
        ax.legend(loc="lower right", prop={"size": 15})
        ax.set(
            title="Adstock-Saturation-Diminishing-Returns Model - $\lambda$ Estimation",
            xlabel="z_adstock (true)",
            ylabel="z_adstock_saturaded (pred)",
        )
        plt.show()

        fig, axes = plt.subplots(
            nrows=4, ncols=1, figsize=(12, 9), sharex=True, sharey=False, layout="constrained"
        )

        sns.lineplot(
            x=date,
            y=z,
            color="black",
            ax=axes[0],
        )
        axes[0].set(title="z")

        for i, var_name in enumerate(["z_adstock", "z_adstock_saturated", "z_effect"]):

            var_name_posterior = endog_scaler.inverse_transform(
                    X=az.extract(data=asdr_model_trace, group="posterior", var_names=var_name)
                )
            var_name_hdi =az.hdi(ary=var_name_posterior.T)

            ax = axes[i + 1]
            sns.lineplot(
                x=date,
                y=var_name_posterior.mean(axis=1),
                color=f"C{i}",
                ax=ax,
            )
            ax.fill_between(
                x=date,
                y1=var_name_hdi[:, 0],
                y2=var_name_hdi[:, 1],
                color=f"C{i}",
                alpha=0.5,
            )
            ax.set(title=var_name)

        plt.show()

        fig, ax = plt.subplots()
        ax.fill_between(
            x=date,
            y1=z_effect_hdi[:, 0],
            y2=z_effect_hdi[:, 1],
            color="C0",
            alpha=0.5,
            label="z_effect 94% HDI",
        )
        ax.axhline(
            y=z_effect_posterior_samples.mean(),
            color="C0",
            linestyle="--",
            label=f"posterior mean {z_effect_posterior_samples.mean().values: 0.3f}",
        )
        # sns.lineplot(x="date", y="z_effect", color="C3", data=data_df, label="z_effect", ax=ax)
        ax.legend(loc="upper right")
        ax.set(
            title="Media Cost Effect Estimation - Adstock-Saturation-Diminishing-Returns Model"
        )
        plt.show()


        fig, ax = plt.subplots()

        az.plot_hdi(
            x=z,
            y=z_effect_posterior_samples,
            color="C0",
            fill_kwargs={"alpha": 0.2, "label": "z_effect 94% HDI"},
            ax=ax,
        )
        sns.scatterplot(
            x="mel_spend",
            y="z_effect_pred_mean",
            color="C0",
            size="index",
            label="z_effect (pred mean)",
            data=data_df.assign(
                z_effect_pred_mean=z_effect_posterior_samples.mean(dim=("chain", "draw"))
            ),
            ax=ax,
        )
        # sns.scatterplot(
        #     x="z", y="z_effect", color="C3", size="index", label="z_effect (true)", data=data_df
        # )
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h[:9], labels=l[:9], loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title="Adstock-Saturation-Diminishing-Returns Model - Estimated Effect");
        plt.show()
        
        z_effect_hdi = az.hdi(ary=z_effect_posterior_samples)["z_effect"]
        z_effect_hdi
        z_effect_posterior_samples.mean(dim=("chain", "draw"))

        data_df = data_df.assign(
            z_effect_pred_mean=z_effect_posterior_samples.mean(dim=("chain", "draw")),
            z_effect_hdi_lower=z_effect_hdi[:, 0],
            z_effect_hdi_upper=z_effect_hdi[:, 1],
        )


        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(10, 9), sharex=True, sharey=True, layout="constrained"
        )

        axes = axes.flatten()

        for i, year in enumerate(data_df["year"].sort_values().unique()[1:]):
            ax = axes[i]
            mask = f"year == {year}"

            # yerr = (
            #     data_df.query(mask)["z_effect_hdi_upper"]
            #     - data_df.query(mask)["z_effect_hdi_lower"]
            # )

            # markers, caps, bars = ax.errorbar(
            #     x=data_df.query(mask)["channel_total_spend"],
            #     y=data_df.query(mask)["z_effect_pred_mean"],
            #     yerr=yerr / 2,
            #     color="C0",
            #     fmt="o",
            #     ms=0,
            #     capsize=5,
            #     label="estimated effect",
            # )
            # [bar.set_alpha(0.3) for bar in bars]
            sns.regplot(
                x="channel_total_spend",
                y="z_effect_pred_mean",
                order=2,
                color="C0",
                label="pred mean effect",
                data=data_df.query(mask),
                ax=ax,
            )
            # sns.regplot(
            #     x="z",
            #     y="z_effect",
            #     order=2,
            #     color="C3",
            #     label="true effect",
            #     data=data_df.query(mask),
            #     ax=ax,
            # )
            if i == 0:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
            else:
                ax.legend().remove()
            ax.set(title=f"{year}")

        fig.suptitle("Media Cost Effect Estimation - ASDR Model", y=1.05);
        plt.show()

        g = sns.displot(x=asdr_roas, kde=True, height=5, aspect=1.5)
        ax = g.axes.flatten()[0]
        # ax.axvline(
        #     x=asdr_roas_mean, color="C0", linestyle="--", label=f"mean = {asdr_roas_mean: 0.3f}"
        # )
        # ax.axvline(
        #     x=asdr_roas_hdi[0],
        #     color="C1",
        #     linestyle="--",
        #     label=f"HDI_lwr = {asdr_roas_hdi[0]: 0.3f}",
        # )
        # ax.axvline(
        #     x=asdr_roas_hdi[1],
        #     color="C2",
        #     linestyle="--",
        #     label=f"HDI_upr = {asdr_roas_hdi[1]: 0.3f}",
        # )
        # ax.axvline(x=roas_true, color="black", linestyle="--", label=f"true = {roas_true: 0.3f}")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title="Adstock-Saturation-Diminishing-Returns Model ROAS")
        plt.show()
        
        g = sns.displot(x=asdr_mroas, kde=True, height=5, aspect=1.5)
        ax = g.axes.flatten()[0]
        ax.axvline(
            x=asdr_mroas_mean, color="C0", linestyle="--", label=f"mean = {asdr_mroas_mean: 0.3f}"
        )
        ax.axvline(
            x=asdr_mroas_hdi[0],
            color="C1",
            linestyle="--",
            label=f"HDI_lwr = {asdr_mroas_hdi[0]: 0.3f}",
        )
        ax.axvline(
            x=asdr_mroas_hdi[1],
            color="C2",
            linestyle="--",
            label=f"HDI_upr = {asdr_roas_hdi[1]: 0.3f}",
        )
        ax.axvline(x=0.0, color="gray", linestyle="--", label="zero")
        ax.axvline(x=mroas_true, color="black", linestyle="--", label=f"true = {mroas_true: 0.3f}")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(title=f"Adstock-Saturation-Diminishing-Returns Model MROAS ({eta:.0%} increase)")
        plt.show()

    # compare models
    def compare_models():
        dataset_dict = {
            "base_model": base_model_trace,
            "adstock_saturation_model": adstock_saturation_model_trace,
            "asdr_model": asdr_model_trace,
        }

        az.compare(compare_dict=dataset_dict, ic="loo", method="stacking", scale="log")

        axes = az.plot_forest(
            data=[base_model_trace, adstock_saturation_model_trace, asdr_model_trace],
            model_names=["base_model", "adstock_saturation_model", "asdr_model"],
            var_names=["a", "b_trend", "b_z", "sigma_slope", "alpha", "lam", "sigma"],
            combined=True,
            figsize=(10, 7),
        )

        axes = az.plot_forest(
            data=[base_model_trace, adstock_saturation_model_trace, asdr_model_trace],
            model_names=["base_model", "adstock_saturation_model", "asdr_model"],
            var_names=["nu"],
            combined=True,
            figsize=(8, 3),
        )

        roas_samples_df = (
            pd.DataFrame(
                data={
                    "base": base_roas,
                    "adstock-saturation": adstock_saturation_roas,
                    "asdr": asdr_mroas,
                }
            )
            .melt()
            .assign(metric="ROAS")
        )

        mroas_samples_df = (
            pd.DataFrame(
                data={
                    "base": base_mroas,
                    "adstock-saturation": adstock_saturation_mroas,
                    "asdr": asdr_mroas,
                }
            )
            .melt()
            .assign(metric="mROAS")
        )


        fig, axes = plt.subplots(
            nrows=2, ncols=1, sharex=True, sharey=False, figsize=(10, 7), layout="constrained"
        )
        sns.violinplot(x="variable", y="value", color="C0", data=roas_samples_df, ax=axes[0])
        axes[0].axhline(
            y=roas_true, color="black", linestyle="--", label=f"true value = {roas_true: 0.3f}"
        )
        axes[0].legend(loc="upper left")
        axes[0].set(title="ROAS Samples - Model Comparison", xlabel="model", ylabel="ROAS")
        sns.violinplot(x="variable", y="value", color="C1", data=mroas_samples_df, ax=axes[1])
        axes[1].axhline(
            y=mroas_true,
            color="black",
            linestyle="--",
            label=f"true value = {mroas_true: 0.3f}",
        )
        axes[1].legend(loc="upper left")
        axes[1].set(
            title=f"mROAS Samples({eta:.0%} increase) - Model Comparison",
            xlabel="model",
            ylabel="mROAS",
        )
        plt.show()
    
    
    # predict
    
    avg_spend = mel_mac_agg_df.iloc[-3:]['channel_total_spend'].mean()
    
    mel_mac_agg_df
    
    pred_weeks_df = pd.merge(pred_weeks_df, dates_df, how='left', on='iso_year_week')
    
    future_spend_df = pred_weeks_df.copy(deep=True)
    future_spend_df['channel_total_spend'] = avg_spend
    pred_z = future_spend_df['channel_total_spend'].values
    pred_z
    pred_z_scaled = channel_scaler.transform(z_pred.reshape(-1, 1)).flatten()
    pred_date = future_spend_df['date'].values
    pred_y = np.array([50_000 for i in range(len(z_pred_scaled))])
    pred_y_scaled = endog_scaler.transform(pred_y.reshape(-1, 1)).flatten()
    pred_t = np.array([t[-1] + (t[-1] - t[-2]) * i for i in range(1, 1 + len(z_pred_scaled))])
    pred_periods = future_spend_df['n_week'] / 52
    pred_fourier_features = pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * pred_periods * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    
    len(z_pred_scaled)
    len(pred_date)
    len(pred_y_scaled)
    len(pred_t)
    pred_fourier_features.shape
    
    # with the asdr_model predict future z_effect with z_pred_scaled
    # fix below, it gives input dimension  mismatch error
    # coords={'date': pred_date, 'fourier_mode': np.arange(2 * n_order)}
    # , var_names=['z_effect', 'trend', 'seasonality', 'likelihood']
    asdr_model_trace_pred = asdr_model_trace.copy()
    
    
    # Add variables to the model to get the forecast and predictions
    # https://discourse.pymc.io/t/making-out-of-sample-predictions-with-gaussianrandomwalk/9703/4
    
    with asdr_model:
        
        pm.set_data(new_data={'z_scaled': pred_z_scaled, 'y_scaled': pred_y_scaled, 't': pred_t, 'fourier_features': pred_fourier_features}, 
                    coords={'date': pred_date})
        # next
        next_slopes = pm.GaussianRandomWalk(
            name="next_slopes",
            sigma=sigma_slope,
            init_dist=pm.distributions.continuous.Normal.dist(
                name="init_dist", mu=slopes[-1], sigma=2
            ),
            dims="date",
        )
        
        next_z_effect = pm.Deterministic(
            name="next_z_effect", var=pm.math.exp(next_slopes) * z_adstock_saturated, dims="date"
        )
        next_mu = pm.Deterministic(name="next_mu", var=trend + seasonality + next_z_effect, dims="date")

        # --- next likelihood ---
        pm.StudentT(name="next_likelihood", nu=nu, mu=next_mu, sigma=sigma, observed=y_scaled_, dims="date")
        
        asdr_model_trace_pred.extend(pm.sample_posterior_predictive(trace=asdr_model_trace_pred, 
                                                                    var_names=['z_effect', 'trend', 'seasonality', 'likelihood', 
                                                                               'next_slopes', 'next_z_effect', 'next_likelihood']))
    
    with asdr_model:
        
        pm.set_data(new_data={'z_scaled': pred_z_scaled, 'y_scaled': pred_y_scaled, 't': pred_t, 'fourier_features': pred_fourier_features}, 
                    coords={'date': pred_date})
        
        asdr_model_trace_pred.extend(pm.sample_posterior_predictive(trace=asdr_model_trace_pred, 
                                                                    var_names=['z_effect', 'trend', 'seasonality', 'likelihood', 
                                                                               'next_slopes', 'next_z_effect', 'next_likelihood']))
    
    
    asdr_model_trace_pred.posterior
    asdr_model_trace_pred.posterior_predictive
    asdr_model_trace_pred.posterior['slopes'].mean(dim=('chain', 'draw'))
    asdr_model_trace_pred.posterior_predictive['slopes'].mean(dim=('chain', 'draw'))
    
    asdr_model_trace_pred.posterior_predictive
    
    z_effect_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        asdr_model_trace_pred.posterior_predictive["next_z_effect"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    trend_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        asdr_model_trace_pred.posterior_predictive["trend"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    seasonality_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        asdr_model_trace_pred.posterior_predictive["seasonality"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    likelihood_posterior_samples = endog_scaler.inverse_transform(
            X=az.extract(
                data=asdr_model_trace_pred,
                group="posterior_predictive",
                var_names=["next_likelihood"],
            )
        )
    
    asdr_z_effect_mean = z_effect_posterior_samples.mean(dim=("chain", "draw"))
    asdr_z_effect_mean
    z_effect_hdi = az.hdi(ary=z_effect_posterior_samples)["z_effect"]
    asdr_trend_mean = trend_posterior_samples.mean(dim=("chain", "draw"))
    trend_hdi = az.hdi(ary=trend_posterior_samples)["trend"]
    asdr_seasonality_mean = seasonality_posterior_samples.mean(dim=("chain", "draw"))
    seasonality_hdi = az.hdi(ary=seasonality_posterior_samples)["seasonality"]
    asdr_likelihood_mean = likelihood_posterior_samples.mean(axis=1)
    asdr_likelihood_mean.shape
    
    asdr_pred_res_df = pd.DataFrame({'asdr_z_effect_mean': asdr_z_effect_mean, 'asdr_trend_mean': asdr_trend_mean, 'asdr_seasonality_mean': asdr_seasonality_mean,
                    'asdr_likelihood_mean': asdr_likelihood_mean})
    
    asdr_pred_res_df['total_mean'] = asdr_pred_res_df[['asdr_z_effect_mean', 'asdr_trend_mean', 'asdr_seasonality_mean']].sum(axis=1)
    asdr_pred_res_df.to_clipboard(index=False)
    
    
    
    
    pass

def old_pymc_marketing_paid_total_model():
    
    # change to train on just from '2023-W08'
    
    from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
    
    min_model_week = '2021-W42'
    
    mel_mac_agg_df
    
    model_df = mel_mac_agg_df[['iso_year_week', 'date', 'channel_total_spend', 'n_conversion']].copy(deep=True)
    model_df = model_df.sort_values('date', ascending=True).reset_index(drop=True)
    model_df = model_df[model_df['iso_year_week'] >= min_model_week].reset_index(drop=True)
    model_df
    
    
    n = model_df.shape[0]
    print(f"Number of observations: {n}")
    # trend feature
    model_df['trend'] = range(n)
    x = model_df[['date', 'channel_total_spend', 'trend']]
    y = model_df['n_conversion']
    
    dummy_model = DelayedSaturatedMMM(date_column="", channel_columns="", adstock_max_lag=4)
    dummy_model.default_model_config
    
    my_sampler_config = {"progressbar": True}
    
    mmm = DelayedSaturatedMMM(
        # model_config=my_model_config,
        sampler_config=my_sampler_config,
        date_column='date',
        channel_columns=['channel_total_spend'],
        control_columns=['trend'],
        adstock_max_lag=8,
        yearly_seasonality=7,
    )
    
    mmm_fit_trace = mmm.fit(X=x, y=y, target_accept=0.95, chains=4, random_seed=42)
    
    az.summary(
        data=mmm.fit_result,
        var_names=[
            "intercept",
            "likelihood_sigma",
            "beta_channel",
            "alpha",
            "lam",
            "gamma_control",
            "gamma_fourier",
        ],
    )
    
    mmm.sample_posterior_predictive(x, extend_idata=True, combined=True)
    
    # mmm.plot_posterior_predictive(original_scale=True)
    # plt.show()
    
    get_mean_contributions_over_time_df = mmm.compute_mean_contributions_over_time(original_scale=True)
    get_mean_contributions_over_time_df.head()
    
    get_mean_contributions_over_time_df['overall_trend'] = get_mean_contributions_over_time_df['trend'] + get_mean_contributions_over_time_df['intercept']
    get_mean_contributions_over_time_df['seasonality'] = get_mean_contributions_over_time_df[[i for i in get_mean_contributions_over_time_df.columns if i not in ('channel_total_spend', 'trend', 'intercept', 'overall_trend')]].sum(axis=1)
    get_mean_contributions_over_time_df = get_mean_contributions_over_time_df[['channel_total_spend', 'overall_trend', 'seasonality']]
    get_mean_contributions_over_time_df['total_pred'] = get_mean_contributions_over_time_df.sum(axis=1)
    get_mean_contributions_over_time_df.columns = [f'{i}_contr' for i in get_mean_contributions_over_time_df.columns]
    get_mean_contributions_over_time_df = get_mean_contributions_over_time_df.reset_index(drop=False)
    model_df = pd.merge(model_df, get_mean_contributions_over_time_df, how='left', on='date')
    model_df
    model_df.to_clipboard(index=False)
    
    if write_to_gs_bool:
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={'pymc_model': model_df})
    
    pass

def old_mmm_pred():
    
    future_spend_df.info()
    future_spend_df = future_spend_df[future_spend_df['time_period'] == 'pred']
    # future_spend_df['trend'] = range(model_df['trend'].max() + 1, model_df['trend'].max() + future_spend_df.shape[0] + 1)
    future_spend_df['trend'] = last_trend_n
    future_spend_df
    full_x_pred = future_spend_df[['date', 'channel_total_spend', 'trend']].reset_index(drop=True)
    full_y_pred = mmm.sample_posterior_predictive(X_pred=full_x_pred, extend_idata=False, include_last_observations=True, original_scale=True)
    future_spend_df['pred_n_conversion'] = full_y_pred['y'].to_series().groupby('date').mean().values
    future_spend_df
    future_spend_df[['date', 'pred_n_conversion']].to_clipboard(index=False)
    
    
    other_agg_df = or_mel_pred_df.groupby('date', as_index=False).agg({'total_spend': 'sum'}).rename(columns={'total_spend': 'other_total_spend'})
    other_agg_df
    future_spend_df = pd.merge(future_spend_df, other_agg_df, how='left', on='date')
    future_spend_df = pd.merge(future_spend_df, ccv_d_fin_df.loc[ccv_d_fin_df['time_period'] == 'pred', ['date', 'ccv', 'discount', 'max_cac']], 
                               how='left', on='date')
    future_spend_df['total_spend'] = future_spend_df['channel_total_spend'] + future_spend_df['other_total_spend']
    future_spend_df['cac'] = future_spend_df['total_spend'] / future_spend_df['pred_n_conversion']
    future_spend_df['pred_roi'] = future_spend_df['ccv'] / (future_spend_df['cac'] + future_spend_df['discount'])
    future_spend_df.to_clipboard(index=False)
    
    
    
    
    
    
    
    fig = mmm.plot_direct_contribution_curves()
    [ax.set(xlabel="x") for ax in fig.axes]
    plt.show()
    
    mmm.plot_channel_contributions_grid(start=0, stop=1.5, num=12)
    plt.show()
    
    mmm.plot_channel_contributions_grid(start=0, stop=1.5, num=12, absolute_xrange=True)
    plt.show()
    
    
    model_df
    pred_weeks_df
    
    x['channel_total_spend'].mean()
    
    future_spend_df = gs_client.read_dataframe('us_future_spend', header_row_num=0)
    future_spend_df['channel_total_spend'] = future_spend_df['channel_total_spend'].str.replace(',', '').astype('float')
    future_spend_df = pd.merge(future_spend_df, dates_df[['iso_year_week', 'date', 'time_period']], how='left', on='iso_year_week')
    future_spend_df.info()
    future_spend_df = future_spend_df[future_spend_df['time_period'] == 'pred']
    future_spend_df['trend'] = range(model_df['trend'].max() + 1, model_df['trend'].max() + future_spend_df.shape[0] + 1)
    future_spend_df
    
    full_x_pred = future_spend_df[['date', 'channel_total_spend', 'trend']].reset_index(drop=True)
    full_y_pred = mmm.sample_posterior_predictive(X_pred=full_x_pred, extend_idata=False, include_last_observations=True, original_scale=True)
    future_spend_df['pred_n_conversion'] = y_pred['y'].to_series().groupby('date').mean().values
    future_spend_df[['date', 'pred_n_conversion']].to_clipboard(index=False)
    
    
    mmm.model
    
    mmm.fit_result.keys()
    
    dir(mmm.fit_result)
    
    dir(mmm.fit_result)
    mmm.fit_result
    mmm.posterior
    
    
    
    y_pred['y'].mean(axis=1).shape
    
    y_dummy = np.array([50_000 for i in range(x_pred.shape[0])])
    y_dummy
    
    
    
    
    mmm_fit_trace.posterior
    mmm_fit_trace.posterior['channel_contributions']
    mmm_fit_trace.posterior['control_contributions']
    mmm_fit_trace.posterior['fourier_contributions']
    mmm_fit_trace.posterior['intercept']
    
    # trace = 
    mmm.fit_result
    
    # channel_contributions, control_contributions, fourier_contributions, intercept
    mmm_fit_trace_pred = mmm_fit_trace.copy()
    var_names = ['channel_contributions', 'control_contributions', 'fourier_contributions', 'intercept', 'y']
    
    with mmm.model:
        mmm._data_setter(X=x_pred, y=y_dummy)
        mmm_fit_trace_pred.extend(pm.sample_posterior_predictive(trace=mmm_fit_trace_pred, var_names=var_names))
    
    mmm_fit_trace_pred.posterior_predictive
    
    mmm_fit_trace_pred.posterior_predictive['channel_contributions'].mean(dim=('chain', 'draw'))
    mmm_fit_trace_pred.posterior_predictive['control_contributions'].mean(dim=('chain', 'draw'))
    mmm_fit_trace_pred.posterior_predictive['fourier_contributions'].mean(dim=('chain', 'draw'))
    
    
    target_scaler = mmm.get_target_transformer()['scaler']
    
    channel_posterior_samples = xr.apply_ufunc(
        lambda x: target_scaler.inverse_transform(X=x.reshape(1, -1)),
        mmm_fit_trace_pred.posterior_predictive['channel_contributions'],
        input_core_dims=[['date']],
        output_core_dims=[['date']],
        vectorize=True,
    )
    channel_posterior_samples.mean(dim=('chain', 'draw')).values
    
    
    x
    x_pred
    x_pred.iloc[0][['channel_total_spend']]
    one_x_pred = x_pred.reset_index(drop=True).loc[[0], ['channel_total_spend']]
    one_leading_up = x[['channel_total_spend']].reset_index(drop=True).loc[[181], ['channel_total_spend']]
    one_leading_up
    x
    
    contribution_pred = mmm.new_spend_contributions(spend=one_x_pred, one_time=True, spend_leading_up=one_leading_up, 
                                                    prior=False, original_scale=True)
    contribution_pred.mean(dim=('chain', 'draw')).shape
    
    # https://github.com/pymc-labs/pymc-marketing/blob/fc37560b76923eb111994a5199e6c15ee5c9cf56/pymc_marketing/mmm/delayed_saturated_mmm.py#L1063
    x_pred[['channel_total_spend']]
    mmm.channel_columns
    
    
    mmm_fit_trace_pred = mmm_fit_trace.copy()
    
    mmm.adstock_max_lag
    
    
    n_channels = len(mmm.channel_columns)
    spend = np.ones(n_channels)
    spend
    
    
    new_data = create_new_spend_data(
            spend=spend,
            adstock_max_lag=self.adstock_max_lag,
            one_time=one_time,
            spend_leading_up=spend_leading_up,
        )

    new_data = (
        self.channel_transformer.transform(new_data) if not prior else new_data
    )

    idata: Dataset = self.fit_result if not prior else self.prior

    coords = {
        "time_since_spend": np.arange(
            -self.adstock_max_lag, self.adstock_max_lag + 1
        ),
        "channel": self.channel_columns,
    }
    with pm.Model(coords=coords):
        alpha = pm.Uniform("alpha", lower=0, upper=1, dims=("channel",))
        lam = pm.HalfFlat("lam", dims=("channel",))
        beta_channel = pm.HalfFlat("beta_channel", dims=("channel",))

        # Same as the forward pass of the model
        channel_adstock = geometric_adstock(
            x=new_data,
            alpha=alpha,
            l_max=self.adstock_max_lag,
            normalize=True,
            axis=0,
        )
        channel_adstock_saturated = logistic_saturation(x=channel_adstock, lam=lam)
        pm.Deterministic(
            name="channel_contributions",
            var=channel_adstock_saturated * beta_channel,
            dims=("time_since_spend", "channel"),
        )

        samples = pm.sample_posterior_predictive(
            idata,
            var_names=["channel_contributions"],
            **sample_posterior_predictive_kwargs,
        )

    channel_contributions = samples.posterior_predictive["channel_contributions"]

    if original_scale:
        channel_contributions = apply_sklearn_transformer_across_dim(
            data=channel_contributions,
            func=self.get_target_transformer().inverse_transform,
            dim_name="time_since_spend",
            combined=False,
        )
    
    
    pass




def old_paid_total_heuristic_roi_mel_mac():
    # total spend = other_ + referral_ + paid impression channel. does not include exclude_
        # Think about how to suggest spend level from model. ROI 1, 1.2, 0.8
        # zero out trend...
        # one step at a time as the past spend impacts the future conversions and spend
    
    pred_weeks_df
        
    future_spend_df = gs_client.read_dataframe('us_future_spend', header_row_num=0)
    future_spend_df['channel_total_spend'] = future_spend_df['channel_total_spend'].str.replace(',', '').astype('float')
    future_spend_df = pd.merge(future_spend_df, dates_df[['iso_year_week', 'date', 'time_period']], how='left', on='iso_year_week')
    future_spend_df
    
    # rois = [0.8, 0.9, 1]
    rois = [0.9]
    select_spends = [[] for i in rois]
    # last_trend_n = model_df.iloc[-1]['trend']
    last_trend_n = model_df.loc[model_df['iso_year_week'] == '2024-W01', 'trend'].values[0]
    select_spends
    
    n_dates = pred_weeks_df.shape[0]
    print(n_dates)
    n_date = 15
    for n_date in range(n_dates):
        print(n_date)
        pred_date = pred_weeks_df.iloc[n_date]['date']
        # pred_max_cac = ccv_d_fin_df.loc[ccv_d_fin_df['date'] == pred_date, 'max_cac'].values[0]
        pred_ccv = ccv_d_fin_df.loc[ccv_d_fin_df['date'] == pred_date, 'ccv'].values[0]
        pred_discount = ccv_d_fin_df.loc[ccv_d_fin_df['date'] == pred_date, 'discount'].values[0]
        pred_flat_spend_total = or_mel_pred_df.loc[or_mel_pred_df['date'] == pred_date, 'total_spend'].sum()
        print(pred_date, pred_ccv, pred_discount, pred_flat_spend_total)
        
        # round abc_spend_amount up to nearest million
        last_year_spend_amount = future_spend_df.loc[future_spend_df['date'] == pred_date, 'channel_total_spend'].values[0]
        top_pred_spend = np.ceil(last_year_spend_amount * 0.80 / 1_000_000) * 1_000_000
        bot_pred_spend = np.floor(top_pred_spend * 0.10 / 1_000_000) * 1_000_000
        
        pred_spend_amounts = np.arange(bot_pred_spend, top_pred_spend, 250_000)
        print('n_pred', len(pred_spend_amounts))
        pred_weeks_df
        for i_roi in range(len(rois)):
            roi = rois[i_roi]
            pred_rois = []
            for pred_spend_amount in pred_spend_amounts:
                spend_x_pred = pd.DataFrame({'date': pred_weeks_df.iloc[:n_date+1]['date'].values, 
                                        'channel_total_spend': select_spends[i_roi][:n_date] + [pred_spend_amount], 
                                        'trend': [last_trend_n for i in range(n_date+1)]})
                spend_y_pred = mmm.sample_posterior_predictive(X_pred=spend_x_pred, extend_idata=False, include_last_observations=True, original_scale=True, progressbar=False)
                spend_y_pred['y'].mean(dim=('sample')).values[-1]
                spend_y_pred['y'].to_series().groupby('date').mean().values[-1]
                pred_spend_conv = spend_y_pred['y'].to_series().groupby('date').mean().values[-1]
                pred_cac = (pred_flat_spend_total + pred_spend_amount) / pred_spend_conv
                pred_roi = pred_ccv / (pred_cac + pred_discount)
                pred_rois.append((pred_spend_amount, pred_spend_conv, pred_cac, pred_ccv, pred_roi))
            pred_rois_df = pd.DataFrame(pred_rois, columns=['spend', 'conv', 'cac', 'ccv', 'roi'])
            pred_rois_df
            pred_rois_df['diff'] = np.abs(pred_rois_df['roi'] - roi)
            prescribe_spend = pred_rois_df.loc[pred_rois_df['diff'].idxmin(), 'spend']
            select_spends[i_roi].append(prescribe_spend)
    
    # create dataframe from select_spends
    pred_weeks_df[[f'{100*i:.0f}_channel_total_spend' for i in rois]] = np.array(select_spends).T
    for roi in rois:
        spend_x_pred = pred_weeks_df[['date', f'{100*roi:.0f}_channel_total_spend']].reset_index(drop=True)
        spend_x_pred.rename(columns={f'{100*roi:.0f}_channel_total_spend': 'channel_total_spend'}, inplace=True)
        spend_x_pred['trend'] = last_trend_n
        spend_y_pred = mmm.sample_posterior_predictive(X_pred=spend_x_pred, extend_idata=False, include_last_observations=True, original_scale=True, progressbar=False)
        pred_spend_conv = spend_y_pred['y'].to_series().groupby('date').mean()
        pred_weeks_df[f'{100*roi:.0f}_channel_total_spend_conv'] = pred_spend_conv.values
        pred_weeks_df[f'{100*roi:.0f}_channel_total_spend_cac'] = ((pred_weeks_df[f'{100*roi:.0f}_channel_total_spend'] + 
                                                                    pred_weeks_df['other_total_spend'] + 
                                                                    pred_weeks_df['referral_total_spend']) / 
                                                                   pred_weeks_df[f'{100*roi:.0f}_channel_total_spend_conv'])
        pred_weeks_df[f'{100*roi:.0f}_channel_total_spend_roi'] = (pred_weeks_df['ccv'] / 
                                                                   (pred_weeks_df[f'{100*roi:.0f}_channel_total_spend_cac'] + + pred_weeks_df['discount']))
    
    pred_weeks_df
    pred_weeks_df.to_clipboard(index=False)
    
    mel_mac_agg_df
    
    pass


def test_calculate_chan_spend(cac, beta, intercept, fixed_spend):
    # Rearrange the equation to express chan_spend in terms of cac
    # cac = (chan_spend + fixed_spend) / (beta * chan_spend + intercept)
    # cac * (beta * chan_spend + intercept) = chan_spend + fixed_spend
    # cac * beta * chan_spend + cac * intercept = chan_spend + fixed_spend
    # cac * beta * chan_spend - chan_spend = fixed_spend - cac * intercept
    # chan_spend * (cac * beta - 1) = fixed_spend - cac * intercept
    # chan_spend = (fixed_spend - cac * intercept) / (cac * beta - 1)
    chan_spend = (fixed_spend - cac * intercept) / (cac * beta - 1)
    return chan_spend

def test_calculate_conv(beta, chan_spend, intercept):
    return beta * chan_spend + intercept

def test_chan_spend_calc_conv():
    cac = 250  # Example value for cac
    beta = 0.5  # Example value for beta
    intercept = 20  # Example value for intercept
    fixed_spend = 20000  # Example value for fixed_spend
    chan_spend = calculate_chan_spend(cac, beta, intercept, fixed_spend)
    conv = calculate_conv(beta, chan_spend, intercept)

    print((chan_spend + fixed_spend) / conv)









