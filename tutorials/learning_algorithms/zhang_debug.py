from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple, NamedTuple

import torch
import neurotorch as nt
import numpy as np
import tqdm
from torch.nn import functional as F
import matplotlib.pyplot as plt


def jacobian(outputs, params):
    jac = [[] for _ in range(len(list(params)))]
    grad_outputs = torch.eye(outputs.shape[-1])
    for i in range(outputs.shape[-1]):
        # zero gradients
        _ = [p.grad.zero_() for p in params if p.grad is not None]
        # compute gradients
        outputs.backward(grad_outputs[i], retain_graph=True)
        for p_idx, param in enumerate(params):
            jac[p_idx].append(param.grad.view(-1).detach().clone())
    jac = [torch.stack(jac[i], dim=-1).T for i in range(len(list(params)))]
    return jac


u_list = []
h_list = []
lr_list = []


def gince_step(inputs, output, target, P, optimizer, **kwargs):
    """

    x.shape = [B, N_in]
    y.shape = [B, N_out]
    error.shape = [B, N_out]
    epsilon.shape = [1, N_out]
    phi.shape = [1, N_in]

    P.shape = [N_in, N_in]
    K = P[N_in, N_in] @ phi.T[N_in, 1] -> [N_in, 1]
    h = 1 / (labda[1] + kappa[1] * phi[1, N_in] @ K[N_in, 1]) -> [1]
    P = labda[1] * P[N_in, N_in] - h[1] * kappa[1] * K[N_in, 1] @ K.T[1, N_in] -> [N_in, N_in]
    grad = h[1] * K[N_in, 1] @ epsilon[1, N_out] -> [N_in, N_out]

    :param inputs: inputs of the layer
    :param output: outputs of the layer
    :param target: targets of the layer
    :param P: inverse covariance matrix of hte inputs
    :param optimizer: optimizer of the layer
    :param kwargs: Additional parameters

    :return: The updated inverse covariance matrix
    """
    labda = kwargs.get("labda", 1.0)
    kappa = kwargs.get("kappa", 1.0)
    optimizer.zero_grad()
    error = output - target  # [B, N_out]
    epsilon = error.view(-1, error.shape[-1]).mean(dim=0).view(1, -1)  # [1, N_out]
    phi = inputs.view(-1, 1, inputs.shape[-1]).mean(dim=0).detach().clone()  # [1, N_in]
    K = torch.matmul(P, phi.T)  # [N_in, N_in] @ [N_in, 1] -> [N_in, 1]
    h = 1.0 / (labda + kappa * torch.matmul(phi, K)).item()  # [1, N_in] @ [N_in, 1] -> [1]
    for p in optimizer.param_groups[0]['params']:
        p.grad = h * torch.outer(K.view(-1), epsilon.view(-1))  # [N_in, 1] @ [1, N_out] -> [N_in, N_out]
    optimizer.step()
    P = labda*P - h*kappa*torch.matmul(K, K.T)  # [N_in, 1] @ [1, N_in] -> [N_in, N_in]
    return P


def gince_grad_step(inputs, output, target, P, optimizer, **kwargs):
    """

    x.shape = [B, f_in]
    y.shape = [B, f_out]
    error.shape = [B, f_out]
    epsilon.shape = [1, f_out]
    phi.shape = [1, f_in]

    param.shape = [N_in, N_out]
    grad.shape = [N_in, N_out]

    P.shape = [f_in, f_in]
    K = P[f_in, f_in] @ phi.T[f_in, 1] -> [f_in, 1]
    h = 1 / (labda[1] + kappa[1] * phi[1, f_in] @ K[f_in, 1]) -> [1]
    P = labda[1] * P[N_in, f_in] - h[1] * kappa[1] * K[f_in, 1] @ K.T[1, f_in] -> [f_in, f_in]
    grad = h[1] * K[f_in, 1] @ epsilon[1, f_out] -> [f_in, f_out]

    :param inputs: inputs of the layer
    :param output: outputs of the layer
    :param target: targets of the layer
    :param P: inverse covariance matrix of hte inputs
    :param optimizer: optimizer of the layer
    :param kwargs: Additional parameters

    :return: The updated inverse covariance matrix
    """
    labda = kwargs.get("labda", 1.0)
    kappa = kwargs.get("kappa", 1.0)
    optimizer.zero_grad()
    output_mean = output.view(-1, 1, output.shape[-1]).mean(dim=0)  # [1, f_out]
    inputs_mean = inputs.view(-1, 1, inputs.shape[-1]).mean(dim=0)  # [1, f_in]
    param = optimizer.param_groups[0]['params'][0]  # [N_in, N_out]

    # with jacobian
    # jac = jacobian(output.view(-1), optimizer.param_groups[0]['params'])[0].view(output.shape[-1], *param.shape)  # [f_out, N_in, N_out]
    # phi = output_mean.detach().clone()  # [1, f_out]
    # K = torch.matmul(P, phi.T)  # [f_out, f_out] @ [f_out, 1] -> [f_out, 1]
    # h = 1.0 / (labda + kappa * torch.matmul(phi, K)).item()  # [1, f_out] @ [f_out, 1] -> [1]
    # for p in optimizer.param_groups[0]['params']:
    #   p.grad = h * torch.matmul(K.T, jac).view(p.shape)  # [1, f_out] @ [f_out, N_in, N_out] -> [N_in, N_out]

    # with grad
    mse_loss = F.mse_loss(output.view(target.shape), target)
    mse_loss.backward()
    grad = param.grad  # [N_in, N_out]
    epsilon = torch.sum(grad, dim=0).view(1, -1)  # sum[N_in]([N_in, N_out]) -> [1, N_out]
    phi = inputs_mean.detach().clone()  # [1, f_in]
    K = torch.matmul(P, phi.T)  # [f_in, f_in] @ [f_in, 1] -> [f_in, 1]
    h = 1.0 / (labda + kappa * torch.matmul(phi, K)).item()  # [1, f_in] @ [f_in, 1] -> [1]
    for p in optimizer.param_groups[0]['params']:
        p.grad = h * torch.matmul(K, epsilon).view(p.shape)  # [f_in, 1] @ [1, N_out] -> [f_in, N_out]
    optimizer.step()
    P = labda * P - h * kappa * torch.matmul(K, K.T)  # [f_in, 1] @ [1, f_in] -> [f_in, f_in]
    return P


def zhang_step(inputs, output, target, P, optimizer, **kwargs):
    labda = kwargs.get("labda", 1.0)
    kappa = kwargs.get("kappa", 1.0)
    optimizer.zero_grad()
    mse_loss = F.mse_loss(output.view(target.shape), target)
    mse_loss.backward()
    phi = inputs.view(-1, inputs.shape[-1], 1).mean(dim=0).detach().clone()  # [f_in, 1]
    u = torch.matmul(P, phi)  # [f_in, f_in] @ [f_in, 1] -> [f_in, 1]
    h = 1.0 / (labda + kappa * torch.matmul(phi.T, u)).item()  # [1, f_in] @ [f_in, 1] -> [1]
    lr = h * P  # [f_in, f_in]
    for p in optimizer.param_groups[0]['params']:
        # TODO: make sure f_in == N_in
        p.grad = torch.matmul(lr, p.grad).clone()  # [f_in, f_in] @ [N_in, N_out] -> [N_in, N_out]
    optimizer.step()
    P = labda * P - h * kappa * torch.matmul(u, u.T)  # [f_in, 1] @ [1, f_in] -> [f_in, f_in]
    return P


def bptt_step(output, target, optimizer, **kwargs):
    optimizer.zero_grad()
    mse_loss = F.mse_loss(output.view(target.shape), target)
    mse_loss.backward()
    optimizer.step()


def zhang_train(data, model, **kwargs):
    nt.set_seed(kwargs.get('seed', 0))
    layer = deepcopy(model.get_layer())
    layer_bptt = deepcopy(model.get_layer())

    data = nt.to_tensor(data).T
    n_time_steps, n_units = data.shape

    # set up the training
    loss_function = nt.losses.PVarianceLoss()
    J = nt.to_tensor(1.5 * np.random.randn(n_units, n_units) / np.sqrt(n_units))
    layer.forward_weights = J.clone()
    layer_bptt.forward_weights = J.clone()
    model.get_layer().forward_weights = J.clone()

    optimizer = torch.optim.SGD([layer.forward_weights], lr=kwargs.get("lr", 1.0))
    optimizer_seq = torch.optim.SGD([model.get_layer().forward_weights], lr=kwargs.get("lr", 1.0))
    optimizer_bptt = torch.optim.SGD([layer_bptt.forward_weights], lr=kwargs.get("lr", 1.0))

    # set up the learning algorithm
    trainer = nt.Trainer(model)
    learning_algorithm = nt.RLS(params=[model.get_layer().forward_weights], strategy="inputs", is_recurrent=True)
    learning_algorithm.start(trainer)

    P_nt = torch.eye(n_units)
    P_seq = torch.eye(n_units)
    losses = defaultdict(list)

    p_bar = tqdm.tqdm(range(kwargs.get("n_iterations", 10)))
    for iteration in p_bar:
        # nt.layer setup
        y_pred_nt = torch.zeros_like(data)
        hh_nt = None
        y_pred_nt[0] = data[0]
        out = y_pred_nt[0][np.newaxis, :].detach().clone()

        # bptt setup
        y_pred_nt_bptt = torch.zeros_like(data)
        hh_nt_bptt = None
        y_pred_nt_bptt[0] = torch.tanh(data[0])
        out_bptt = y_pred_nt_bptt[0][np.newaxis, :].detach().clone()

        # seq setup
        y_pred_seq = torch.zeros_like(data)
        y_pred_seq[0] = data[0]
        x_batch = y_pred_seq[0][np.newaxis, np.newaxis, :]
        y_batch = data[np.newaxis, :]
        trainer.update_state_(x_batch=x_batch, y_batch=y_batch)
        learning_algorithm.on_batch_begin(trainer)
        y_pred_seq[1:] = model.get_prediction_trace(x_batch)
        learning_algorithm.on_batch_end(trainer)

        for t in range(1, n_time_steps):
            # nt.layer step
            # inputs = y_pred_nt[t-1][np.newaxis, :].detach().clone()
            # inputs = y_pred_nt[t-1][np.newaxis, :]
            out, hh_nt = layer(out, hh_nt)
            y_pred_nt[t] = out.detach().clone()
            # y_pred_nt[:, t], hh_nt = layer(y_pred_nt[:, t-1], hh_nt)

            # hh_nt = tuple([hh_nt_i.detach().clone() for hh_nt_i in hh_nt])
            # activation = layer.activation(torch.matmul(y_pred_nt[t-1][np.newaxis, :], layer.forward_weights) - layer.mu)
            # ratio_dt_tau = layer.dt / layer.tau
            # transition_rate = (1 - hh_nt * self.r)
            # out = hh_nt * (1 - ratio_dt_tau) + transition_rate * activation * ratio_dt_tau
            # y_pred_nt[t] = out.detach().clone()
            # out_bptt, hh_nt_bptt = layer_bptt(out_bptt, hh_nt_bptt)
            # y_pred_nt_bptt[t] = out_bptt
            # P_nt, u_nt, h_nt, lr_nt = zhang_step(y_pred_nt[t-1], out, data[t], P_nt, optimizer)
            # P_nt = gince_grad_step(y_pred_nt[t-1], out, data[t], P_nt, optimizer)
            P_nt = gince_step(y_pred_nt[t-1], out, data[t], P_nt, optimizer)
            hh_nt = tuple([hh_nt_i.detach().clone() for hh_nt_i in hh_nt])
            out.detach_()

        # P_nt, u_nt, h_nt, lr_nt = zhang_step(y_pred_nt, y_pred_nt, data, P_nt, optimizer)
        # u_list.append(u_nt.mean().item())
        # h_list.append(h_nt)
        # lr_list.append(lr_nt.mean().item())
        # P_seq, u_seq, h_seq, lr_seq = zhang_step(y_pred_seq, y_pred_seq, data, P_seq, optimizer_seq)
        # P_seq = gince_step(y_pred_seq, y_pred_seq, data, P_seq, optimizer_seq)
        # bptt_step(y_pred_nt_bptt, data, optimizer_bptt)

        # compute and print loss
        mse_loss_nt = F.mse_loss(y_pred_nt, data)
        loss_nt = loss_function(y_pred_nt, data)
        mse_loss_nt_bptt = F.mse_loss(y_pred_nt_bptt, data)
        loss_nt_bptt = loss_function(y_pred_nt_bptt, data)
        loss_seq = loss_function(y_pred_seq, data)
        losses['nt.layer.pVar'].append(loss_nt.item())
        losses['nt.layer.mse'].append(mse_loss_nt.item())
        # losses['nt.layer_bptt.pVar'].append(loss_nt_bptt.item())
        # losses['nt.layer_bptt.mse'].append(mse_loss_nt_bptt.item())
        losses['seq'].append(loss_seq.item())
        p_bar.set_description(
            f"Loss nt.layer: {loss_nt.item():.4f}, "
            f"MSE Loss nt.layer: {mse_loss_nt.item():.4f}, "
            # f"Loss nt.layer_bptt: {loss_nt_bptt.item():.4f}, "
            # f"MSE Loss nt.layer_bptt: {mse_loss_nt_bptt.item():.4f}, "
            f"Loss seq: {loss_seq.item():.4f} "
        )
    fig, axes = plt.subplots(1, len(losses))
    for ax_idx, (key, val_losses) in enumerate(losses.items()):
        axes[ax_idx].plot(val_losses, label=key)
        axes[ax_idx].legend()
    plt.show()

# fig, axes = plt.subplots(1, 3)
# axes[0].plot(u_list, label='u')
# axes[0].set_title('u')
# axes[1].plot(h_list, label="h")
# axes[1].set_title('h')
# axes[2].plot(lr_list, label="lr")
# axes[2].set_title('lr')
# plt.show()


if __name__ == '__main__':
    curbd_data = np.load("data/ts/curbd_Adata.npy")
    network = nt.SequentialRNN(
        layers=[nt.WilsonCowanLayer(
            curbd_data.shape[0], curbd_data.shape[0],
            activation="tanh",
            tau=0.1,
            dt=0.01,
        )],
        foresight_time_steps=curbd_data.shape[-1]-1,
        out_memory_size=curbd_data.shape[-1]-1,
        hh_memory_size=1,
        device=torch.device("cpu"),
    ).build()
    zhang_train(curbd_data, network, n_iterations=10)



