import torch
import mlflow
# import hiddenlayer as HL

from model.collectdata_mdsA import collect_data
from model.collectdata_poca_KDE import collect_data_poca
from model.alt_loss_A import Loss
from model.training import trainNet, select_gpu
from model.utilities import load_full_state, count_parameters, Params, save_to_mlflow

from model.autoencoder_models import UNet
from model.autoencoder_models import UNetPlusPlus
from model.autoencoder_models import PerturbativeUNetPlusPlus

args = Params(
    batch_size=64,
    device = select_gpu(0),
    epochs=100,
    lr=1e-4,
    experiment_name='Top Models',
    asymmetry_parameter=2.5
)

events = 320000
## This is used when training with the poca kde (kde_a,kde_b) (x,y)
train_loader = collect_data_poca('/share/lazy/will/data/June30_2020_80k_1.h5',
                            '/share/lazy/will/data/June30_2020_80k_3.h5',
                            '/share/lazy/will/data/June30_2020_80k_4.h5',
                            '/share/lazy/will/data/June30_2020_80k_5.h5',
                            batch_size=args['batch_size'],
                            #device=args['device'],
                            masking=True, shuffle=True,
                            load_A_and_B=True,
                            load_xy=True,
                           ## slice = slice(0,18000)
                           )

val_loader = collect_data_poca('/share/lazy/sokoloff/ML-data_AA/20K_POCA_kernel_evts_200926.h5',
                            batch_size=args['batch_size'],
                            #device=args['device'],
                            masking=True, shuffle=True,
                            load_A_and_B=True,
                            load_xy=True,
                            ##slice = slice(18000,None)
                           )


mlflow.tracking.set_tracking_uri('file:/share/lazy/pv-finder_model_repo')
mlflow.set_experiment(args['experiment_name'])

model = UNetPlusPlus().to(args['device'])
model.to("cuda:0")
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss = Loss(epsilon=1e-5,coefficient=args['asymmetry_parameter'])

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

# load_full_state(model, optimizer, '/share/lazy/pv-finder_model_repo/17/27d5279c4a0641ecb3807f400b25a9a8/artifacts/run_stats.pyt', freeze_weights=False)

run_name = 'u-net++'

train_iter = enumerate(trainNet(model, optimizer, loss, train_loader, val_loader, args['epochs'], notebook=True))
with mlflow.start_run(run_name = run_name) as run:
    mlflow.log_artifact('train.ipynb')
    for i, result in train_iter:
        print(result.cost)
        torch.save(model, 'run_stats.pyt')
        mlflow.log_artifact('run_stats.pyt')

        save_to_mlflow({
            'Metric: Training loss':result.cost,
            'Metric: Validation loss':result.val,
            'Metric: Efficiency':result.eff_val.eff_rate,
            'Metric: False positive rate':result.eff_val.fp_rate,
            'Param: Parameters':parameters,
            'Param: Events':events,
            'Param: Asymmetry':args['asymmetry_parameter'],
            'Param: Epochs':args['epochs'],
        }, step=i)
        
