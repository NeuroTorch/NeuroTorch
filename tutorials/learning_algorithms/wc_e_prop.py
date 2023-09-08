import pprint
from typing import Union, Callable

import neurotorch as nt
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric
from neurotorch.visualisation.connectome import visualize_init_final_weights
from neurotorch.visualisation.time_series_visualisation import *
from tutorials.learning_algorithms.dataset import get_dataloader, TimeSeriesDataset


def foresight_time_steps_updater_decorator(
        model: nt.SequentialRNN,
        prediction_mth: Union[str, Callable],
        *,
        train_foresight_time_steps: int,
        train_out_memory_size: int,
        train_hh_memory_size: int,
        val_foresight_time_steps: int,
        val_out_memory_size: int,
        val_hh_memory_size: int,
) -> Tuple[Callable, Callable]:
    """
    Decorator to update the model's foresight_time_steps, out_memory_size and hh_memory_size attributes
    according to the training/validation mode.

    :param model: The model to update
    :type model: nt.SequentialRNN
    :param prediction_mth: The prediction method to update
    :type prediction_mth: Union[str, Callable]
    :param train_foresight_time_steps: The foresight_time_steps to use during training
    :type train_foresight_time_steps: int
    :param train_out_memory_size: The out_memory_size to use during training
    :type train_out_memory_size: int
    :param train_hh_memory_size: The hh_memory_size to use during training
    :type train_hh_memory_size: int
    :param val_foresight_time_steps: The foresight_time_steps to use during validation
    :type val_foresight_time_steps: int
    :param val_out_memory_size: The out_memory_size to use during validation
    :type val_out_memory_size: int
    :param val_hh_memory_size: The hh_memory_size to use during validation
    :type val_hh_memory_size: int

    :return: The updated prediction method and the old prediction method
    :rtype: Tuple[Callable, Callable]
    """
    if isinstance(prediction_mth, str):
        prediction_mth = getattr(model, prediction_mth)
    old_pred_mth = prediction_mth

    def _pred_mth(*args, **kwargs):
        if model.training:
            model.foresight_time_steps = train_foresight_time_steps
            model.out_memory_size = train_out_memory_size
            model.hh_memory_size = train_hh_memory_size
        else:
            model.foresight_time_steps = val_foresight_time_steps
            model.out_memory_size = val_out_memory_size
            model.hh_memory_size = val_hh_memory_size
        return old_pred_mth(*args, **kwargs)
    return _pred_mth, old_pred_mth


def set_default_param(**kwargs):
    kwargs.setdefault("filename", None)
    kwargs.setdefault("dataset_length", -1)
    kwargs.setdefault("n_time_steps", -1)
    kwargs.setdefault("target_skip_first", True)
    kwargs.setdefault("dataset_randomize_indexes", False)
    kwargs.setdefault("rm_dead_units", True)
    kwargs.setdefault("smoothing_sigma", 15.0)
    kwargs.setdefault("learning_rate", 1e-2)
    kwargs.setdefault("std_weights", 1)
    kwargs.setdefault("dt", 0.02)
    kwargs.setdefault("mu", 0.0)
    kwargs.setdefault("mean_mu", 0.0)
    kwargs.setdefault("std_mu", 1.0)
    kwargs.setdefault("r", 0.1)
    kwargs.setdefault("mean_r", 0.5)
    kwargs.setdefault("std_r", 0.4)
    kwargs.setdefault("tau", 0.1)
    kwargs.setdefault("learn_mu", True)
    kwargs.setdefault("learn_r", True)
    kwargs.setdefault("learn_tau", True)
    kwargs.setdefault("force_dale_law", False)
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("n_units", 200)
    kwargs.setdefault("n_aux_units", kwargs["n_units"])
    kwargs.setdefault("use_recurrent_connection", False)
    kwargs.setdefault("add_out_layer", True)
    if not kwargs["add_out_layer"]:
        kwargs["n_aux_units"] = kwargs["n_units"]
    kwargs.setdefault("forward_sign", 0.5)
    kwargs.setdefault("activation", "sigmoid")
    kwargs.setdefault("hh_init", "inputs" if kwargs["n_aux_units"] == kwargs["n_units"] else "random")
    return kwargs


def train_with_params(
        params: Optional[dict] = None,
        n_iterations: int = 100,
        device: torch.device = torch.device("cpu"),
        **kwargs
):
    if params is None:
        params = {}
    params = set_default_param(**params)
    kwargs.setdefault("force_overwrite", False)
    torch.manual_seed(params["seed"])
    dataloader = get_dataloader(
        batch_size=kwargs.get("batch_size", 512), verbose=True, n_workers=kwargs.get("n_workers"), **params
    )
    val_params = deepcopy(params)
    val_params["dataset_length"] = 1
    val_params["n_time_steps"] = -1
    val_dataloader = get_dataloader(
        batch_size=kwargs.get("batch_size", 512), verbose=True, n_workers=kwargs.get("n_workers"), **val_params
    )
    dataset: TimeSeriesDataset = dataloader.dataset
    params = dataset.set_params_from_self(params)
    if params["n_aux_units"] < 0:
        params["n_aux_units"] = params["n_units"]
    x = dataset.full_time_series
    val_dataset: TimeSeriesDataset = val_dataloader.dataset
    x_val = val_dataset.full_time_series
    wc_layer = nt.WilsonCowanLayer(
        dataset.n_units, params["n_aux_units"],
        std_weights=params["std_weights"],
        forward_sign=params["forward_sign"],
        dt=params["dt"],
        r=params["r"],
        mean_r=params["mean_r"],
        std_r=params["std_r"],
        mu=params["mu"],
        mean_mu=params["mean_mu"],
        std_mu=params["std_mu"],
        tau=params["tau"],
        learn_r=params["learn_r"],
        learn_mu=params["learn_mu"],
        learn_tau=params["learn_tau"],
        hh_init=params["hh_init"],
        device=device,
        name="WilsonCowan_layer1",
        force_dale_law=params["force_dale_law"],
        activation=params["activation"],
        use_recurrent_connection=params["use_recurrent_connection"],
    ).build()
    layers = [wc_layer, ]
    if params["add_out_layer"]:
        out_layer = nt.Linear(
            params["n_aux_units"], x.shape[-1],
            device=device,
            use_bias=False,
            activation=params["activation"],
        )
        layers.append(out_layer)
    checkpoint_manager = nt.CheckpointManager(
        checkpoint_folder="data/tr_data/checkpoints_wc_e_prop",
        metric="val_p_var",
        minimise_metric=False,
        save_freq=max(1, int(0.1 * n_iterations)),
        start_save_at=max(1, int(0.1 * n_iterations)),
        save_best_only=True,
    )
    model = nt.SequentialRNN(
        layers=layers,
        device=device,
        foresight_time_steps=dataset.n_time_steps - 1,
        out_memory_size=dataset.n_time_steps - 1,
        hh_memory_size=1,
        checkpoint_folder=checkpoint_manager.checkpoint_folder,
    ).build()
    model.get_prediction_trace = foresight_time_steps_updater_decorator(
        model, model.get_prediction_trace,
        train_foresight_time_steps=dataset.n_time_steps - 1,
        train_out_memory_size=dataset.n_time_steps - 1,
        train_hh_memory_size=1,
        val_foresight_time_steps=val_dataset.n_time_steps - 1,
        val_out_memory_size=val_dataset.n_time_steps - 1,
        val_hh_memory_size=1,
    )[0]
    la = nt.Eprop(
        alpha=0.0,
        gamma=0.0,
        params_lr=1e-4,
        output_params_lr=2e-4,
        # default_optimizer_cls=torch.optim.AdamW,
        # default_optim_kwargs={"weight_decay": 0.2, "lr": 1e-6},
        # eligibility_traces_norm_clip_value=torch.inf,
        grad_norm_clip_value=1.0,
        # learning_signal_norm_clip_value=torch.inf,
        # feedback_weights_norm_clip_value=torch.inf,
        # feedbacks_gen_strategy="randn",
    )
    lr_scheduler = LRSchedulerOnMetric(
        'val_p_var',
        metric_schedule=np.linspace(kwargs.get("lr_schedule_start", 0.0), 1.0, 100),
        min_lr=[1e-6, 2e-6],
        retain_progress=True,
        priority=la.priority + 1,
    )
    timer = nt.callbacks.early_stopping.EarlyStoppingOnTimeLimit(
        delta_seconds=kwargs.get("time_limit", 60.0 * 60.0),
        resume_on_load=True
    )
    callbacks = [la, checkpoint_manager, lr_scheduler, timer]

    with torch.no_grad():
        W0 = nt.to_numpy(wc_layer.forward_weights.clone())
        if params["force_dale_law"]:
            sign0 = nt.to_numpy(wc_layer.forward_sign.clone())
        else:
            sign0 = None
        mu0 = wc_layer.mu.clone()
        r0 = wc_layer.r.clone()
        tau0 = wc_layer.tau.clone()
        if wc_layer.force_dale_law:
            ratio_sign_0 = (np.mean(nt.to_numpy(torch.sign(wc_layer.forward_sign))) + 1) / 2
        else:
            ratio_sign_0 = (np.mean(nt.to_numpy(torch.sign(wc_layer.forward_weights))) + 1) / 2

    trainer = nt.trainers.Trainer(
        model,
        predict_method="get_prediction_trace",
        callbacks=callbacks,
        metrics=[nt.metrics.RegressionMetrics(model, "p_var")],
    )
    print(f"{trainer}")
    history = trainer.train(
        dataloader,
        val_dataloader,
        n_iterations=n_iterations,
        exec_metrics_on_train=False,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=kwargs["force_overwrite"],
    )
    history.plot(save_path=f"data/figures/wc_eprop/tr_history.png", show=True)

    model.eval()
    try:
        model.load_checkpoint(checkpoint_manager.checkpoints_meta_path)
    except:
        model.load_checkpoint(
            checkpoint_manager.checkpoints_meta_path, load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR
        )
    model.foresight_time_steps = val_dataloader.dataset.n_time_steps - 1
    model.out_memory_size = model.foresight_time_steps
    model.hh_memory_size = 1
    x_pred = torch.concat(
        [
            torch.unsqueeze(x_val[:, 0].clone(), dim=1).to(model.device),
            model.get_prediction_trace(torch.unsqueeze(x_val[:, 0].clone(), dim=1))
        ], dim=1
    )
    loss = PVarianceLoss()(x_pred.to(x_val.device), x_val)

    out = {
        "params"              : params,
        "pVar"                : nt.to_numpy(loss.item()),
        "W"                   : nt.to_numpy(wc_layer.forward_weights),
        "sign0"               : sign0,
        "mu"                  : nt.to_numpy(wc_layer.mu),
        "r"                   : nt.to_numpy(wc_layer.r),
        "W0"                  : W0,
        "ratio_0"             : ratio_sign_0,
        "mu0"                 : nt.to_numpy(mu0),
        "r0"                  : nt.to_numpy(r0),
        "tau0"                : nt.to_numpy(tau0),
        "tau"                 : nt.to_numpy(wc_layer.tau),
        "x_pred"              : nt.to_numpy(torch.squeeze(x_pred)),
        "original_time_series": x_val.squeeze(),
        "force_dale_law"      : params["force_dale_law"],
    }
    if wc_layer.force_dale_law:
        out["ratio_end"] = (np.mean(nt.to_numpy(torch.sign(wc_layer.forward_sign))) + 1) / 2
        out["sign"] = nt.to_numpy(wc_layer.forward_sign.clone())
    else:
        out["ratio_end"] = (np.mean(nt.to_numpy(torch.sign(wc_layer.forward_weights))) + 1) / 2
        out["sign"] = None

    return out


if __name__ == '__main__':
    res = train_with_params(
        params={
            # "filename": "ts_nobaselines_fish3_800t.npy",
            # "filename": "corrected_data.npy",
            # "filename": "curbd_Adata.npy",
            "filename"                      : "Stimulus_data_2022_02_23_fish3_1.npy",
            # "filename"                      : None,
            "smoothing_sigma"               : 10.0,
            "n_units"                       : -1,
            # "n_aux_units"                   : 512,
            # "n_time_steps"                  : 500,
            # "dataset_length"                : 1,
            "dataset_randomize_indexes"     : False,
            "force_dale_law"                : True,
            "learn_mu"                      : True,
            "learn_r"                       : True,
            "learn_tau"                     : True,
            "use_recurrent_connection"      : False,
        },
        n_iterations=1_000,
        device=torch.device("cuda"),
        force_overwrite=True,
        batch_size=512,
        time_limit=1.0 * 60.0 * 60.0,
    )
    pprint.pprint({k: v for k, v in res.items() if isinstance(v, (int, float, str, bool))})

    if res["force_dale_law"]:
        print(f"initiale ratio {res['ratio_0']:.3f}, finale ratio {res['ratio_end']:.3f}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        visualize_init_final_weights(
            res["W0"], res["W"],
            fig=fig, axes=axes[0],
            show=False,
        )
        axes[0, 0].set_title("Initial weights, ratio exec {:.3f}".format(res["ratio_0"]))
        axes[0, 1].set_title("Final weights, ratio exec {:.3f}".format(res["ratio_end"]))
        sort_idx = np.argsort(res["sign"].ravel())
        axes[1, 0].plot(res["sign0"].ravel()[sort_idx])
        axes[1, 0].set_title("Initial signs")
        axes[1, 1].plot(res["sign"].ravel()[sort_idx])
        axes[1, 1].set_title("Final signs")
        plt.show()

    viz = Visualise(
        res["x_pred"],
        shape=nt.Size(
            [
                nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
                nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
            ]
        )
    )
    viz.plot_timeseries_comparison_report(
        res["original_time_series"],
        title=f"Prediction",
        filename=f"data/figures/wc_eprop/timeseries_comparison_report.png",
        show=True,
        dpi=600,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    viz_kmeans = VisualiseKMeans(
        res["original_time_series"],
        nt.Size(
            [
                nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
                nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
            ]
        )
    )
    viz_kmeans.heatmap(fig=fig, ax=axes[0], title="True time series")
    Visualise(
        viz_kmeans.permute_timeseries(res["x_pred"]),
        nt.Size(
            [
                nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
                nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
            ]
        )
    ).heatmap(fig=fig, ax=axes[1], title="Predicted time series")
    plt.savefig("data/figures/wc_eprop/heatmap.png")
    plt.show()

    Visualise(
        res["x_pred"],
        nt.Size(
            [
                nt.Dimension(None, nt.DimensionProperty.TIME, "Time [s]"),
                nt.Dimension(None, nt.DimensionProperty.NONE, "Neuron [-]"),
            ]
        )
    ).animate(
        time_interval=0.1,
        weights=res["W"],
        dt=0.1,
        show=True,
        filename="data/figures/wc_eprop/animation.gif",
        writer=None,
    )


