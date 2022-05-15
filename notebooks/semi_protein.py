import torch
from torch import nn
from src.data import get_data
from src.models import get_encoder, get_decoder
from src.utils import softclip
from src.trainer_lae_elbo import LitLaplaceAutoEncoder
from src.hessian import sampler
import yaml
from copy import deepcopy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

def get_model(encoder, decoder):
    
    net = deepcopy(encoder.encoder._modules)
    decoder = decoder.decoder._modules
    max_ = max([int(i) for i in net.keys()])
    for i in decoder.keys():
        net.update({f"{max_+int(i) + 1}": decoder[i]})

    return nn.Sequential(net)

_, val_dataloader = get_data("protein")
proteins, labels = [ ], [ ]

for batch_idx, batch in enumerate(val_dataloader):
    if batch_idx > 20:
        break
    x, y = batch
    proteins.append(x)
    labels.append(y)
proteins = torch.cat(proteins, 0)
labels = torch.cat(labels, 0)

idx = torch.randperm(proteins.shape[0])
proteins = proteins[idx]
labels = labels[idx]
print(proteins.shape, labels.shape)

#selected_datapoints = proteins[:10]
#selected_labels = labels[:10]
#eval_set_datapoints = proteins[10:]
#eval_set_labels = labels[10:]



def select_data(n_select):
    selected_datapoints = torch.zeros(10*n_select, 2592, 24)
    selected_labels = torch.zeros(10*n_select,)
    eval_set_datapoints = torch.zeros(proteins.shape[0]-10*n_select, 2592, 24)
    eval_set_labels = torch.zeros(proteins.shape[0]-10*n_select, )

    count = 0
    for i in torch.unique(labels):
        idx = torch.where(labels == i)[0]
        n = len(idx)
        rand_idx = torch.randperm(n)

        selected_datapoints[i*n_select:(i+1)*n_select] = proteins[idx[rand_idx[:n_select]]]
        selected_labels[i*n_select:(i+1)*n_select] = labels[idx[rand_idx[:n_select]]]
        eval_set_datapoints[count:count+len(rand_idx[n_select:])] = proteins[idx[rand_idx[n_select:]]]
        eval_set_labels[count:count+len(rand_idx[n_select:])] = labels[idx[rand_idx[n_select:]]]

        count += len(rand_idx[n_select:])
    
    return selected_datapoints, selected_labels, eval_set_datapoints, eval_set_labels

model = sys.argv[1]
print("model:", model)

if model == 'ae':
    path_ae = "../weights/protein/ae_[use_var_dec=False]/[no_conv_True]_[use_var_decoder_False]_"
    with open(f"{path_ae}/config.yaml") as file:
        config = yaml.full_load(file)
    ae_encoder = get_encoder(config, latent_size=2)
    ae_encoder.load_state_dict(torch.load(f"{path_ae}/encoder.pth"))
    
    def encoder(data, n=1):
        return ae_encoder(data)

elif model == 'mcae':
    path_mcae = "../weights/protein/mcdropout_ae/[no_conv_True]_[dropout_rate_0.2]_[use_var_decoder_False]_"
    with open(f"{path_mcae}/config.yaml") as file:
        config = yaml.full_load(file)
    mcae_encoder = get_encoder(config, latent_size=2, dropout=config["dropout_rate"])
    mcae_encoder.load_state_dict(torch.load(f"{path_mcae}/encoder.pth"))
    
    def encoder(data, n=1):
        return torch.cat([mcae_encoder(data) for _ in range(n)], 0)

elif model == 'vae':

    path_vae = "../weights/protein/vae_[use_var_dec=False]/[no_conv_True]_[use_var_decoder_False]_"
    with open(f"{path_vae}/config.yaml") as file:
        config = yaml.full_load(file)
    vae_encoder_mu = get_encoder(config, latent_size=2)
    vae_encoder_mu.load_state_dict(torch.load(f"{path_vae}/mu_encoder.pth"))
    vae_encoder_var = get_encoder(config, latent_size=2)
    vae_encoder_var.load_state_dict(torch.load(f"{path_vae}/var_encoder.pth"))
    
    def encoder(data, n=1):
        mu = vae_encoder_mu(data)
        sigma = torch.exp(softclip(vae_encoder_var(data), min=-3))
        return torch.cat([mu + torch.randn_like(sigma) * sigma for _ in range(n)])

elif model == 'lae':

    path_lae = '../epoch=58-step=375948.ckpt'
    model = LitLaplaceAutoEncoder.load_from_checkpoint(path_lae, dataset_size=1)
    # lae_encoder = model.encoder # get_encoder(model.config, latent_size=2)
    # lae_decoder = model.decoder # get_decoder(config, latent_size=2)
    lae_net = model.net.eval() # get_model(lae_encoder, lae_decoder).eval()
    hessian_approx = sampler.DiagSampler()
    h = model.hessian # torch.load(f"{path_lae}/hessian.pth")
    sigma_q = hessian_approx.invert(h).cpu()
    
    def encoder(data, n=1):
        embeddings = [ ]
        def fw_hook_get_latent(module, input, output):
            embeddings.append(output.detach().cpu())
        temp = deepcopy(lae_net)
        temp[7].register_forward_hook(fw_hook_get_latent)

        if n == 1:
            _ = temp(data)
            return embeddings[0]

        mu_q = parameters_to_vector(temp.parameters())
        for i in range(n):
            sample = hessian_approx.sample(mu_q, sigma_q, n_samples=1)
            vector_to_parameters(sample[0], temp.parameters())
            _ = temp(data)
        return torch.cat(embeddings, 0)



with torch.inference_mode():
    reps = 2
    scaling = 100
    res_mean, res_std = [], []
    for num_datapoints in [20, 40, 60, 80, 100, 200]:
        
        selected_datapoints = proteins[:num_datapoints]
        selected_labels = labels[:num_datapoints]
        eval_set_datapoints = proteins[num_datapoints:]
        eval_set_labels = labels[num_datapoints:]
        eval_data = torch.cat([encoder(d.unsqueeze(0)) for d in eval_set_datapoints], 0)
        print(f'eval set for {num_datapoints} : {eval_data.shape}')
        res = [ ]
        for r in range(reps):
            print(f'training model {r+1}')
            train_data = torch.cat([encoder(d.unsqueeze(0), scaling) for d in selected_datapoints], 0)
            train_labels = selected_labels.repeat_interleave(scaling) if model != 'ae' else selected_labels    
            classifier = KNeighborsClassifier(1)
            classifier.fit(train_data.detach().numpy(), train_labels.detach().numpy())    
            preds = classifier.predict(eval_data.detach().numpy())
            acc = (preds == eval_set_labels.numpy()).mean()
            res.append(acc)
        res_mean.append(np.mean(res))
        res_std.append(np.std(res))
        
    np.save(f"{model}_resmean.npy", np.asarray(res_mean))
    np.save(f"{model}_resstd.npy", np.asarray(res_std))

# scales = [1, 2, 5, 10]#, 20, 100]
# res_mean, res_std = [], []
# reps = 2

# eval_data = torch.cat([encoder(d.unsqueeze(0)) for d in eval_set_datapoints], 0)
# res_mean.append([ ])
# res_std.append([ ])
# for j, s in tqdm(enumerate(scales)):
#     res = [ ]
#     for r in range(reps):
#         train_data = torch.cat([encoder(d.unsqueeze(0), s) for d in selected_datapoints], 0)
#         train_labels = selected_labels.repeat_interleave(s) if model != 'ae' else selected_labels
#         classifier = KNeighborsClassifier(1)
#         # classifier = GridSearchCV(
#         #     KNeighborsClassifier(),
#         #     {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
#         #     cv=3,
#         #     refit=True,
#         # )
#         classifier.fit(train_data.detach().numpy(), train_labels.detach().numpy())
#         preds = classifier.predict(eval_data.detach().numpy())
#         acc = (preds == eval_set_labels.numpy()).mean()
#         res.append(acc)

#     res_mean[-1].append(np.mean(res))
#     res_std[-1].append(np.std(res))

#     #preds_grid = classifier.predict(z_grid.numpy())
#     # ax[i, j].contourf(
#     #     z_grid[:,0].reshape(100, 100),
#     #     z_grid[:,1].reshape(100, 100),
#     #     preds_grid.reshape(100, 100)
#     # )
    

