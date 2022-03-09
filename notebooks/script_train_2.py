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
from model.autoencoder_models import DenseNet as DenseNet

args = Params(
    batch_size=64,
    device = select_gpu(2),
    epochs=200,
    lr=1e-6,
    experiment_name='Top Models',
    asymmetry_parameter=2.5
)

'''
train_loader = collect_data(
    '/share/lazy/sokoloff/ML-data_A/Aug14_80K_train.h5',
      '/share/lazy/sokoloff/ML-data_AA/Oct03_80K_train.h5',
#     '/share/lazy/sokoloff/ML-data_AA/Oct03_40K_train.h5',
      '/share/lazy/will/ML_mdsA/June30_2020_80k_1.h5',
     '/share/lazy/will/ML_mdsA/June30_2020_80k_3.h5',
     '/share/lazy/will/ML_mdsA/June30_2020_80k_4.h5',
#     '/share/lazy/will/ML_mdsA/June30_2020_80k_5.h5',
#     '/share/lazy/will/ML_mdsA/June30_2020_80k_6.h5',
#     '/share/lazy/will/ML_mdsA/June30_2020_80k_7.h5',
#     '/share/lazy/will/ML_mdsA/June30_2020_80k_8.h5',
#     '/share/lazy/will/ML_mdsA/June30_2020_80k_9.h5',
    batch_size=args['batch_size'],
    masking=True,
    shuffle=False,
    load_XandXsq=False,
#     device = args['device'], 
    load_xy=False)

val_loader = collect_data(
    '/share/lazy/sokoloff/ML-data_AA/Oct03_20K_val.h5',
    batch_size=args['batch_size'],
    slice=slice(256 * 39),
    masking=True, 
    shuffle=False,
    load_XandXsq=False,
    load_xy=False)
'''

events = 320000
## This is used when training with the new KDE
train_loader = collect_data_poca(#'/share/lazy/will/data/June30_2020_80k_1.h5',
#                            '/share/lazy/will/data/June30_2020_80k_3.h5',
#                            '/share/lazy/will/data/June30_2020_80k_4.h5',
#                            '/share/lazy/will/data/June30_2020_80k_5.h5',
                            # full lhcb data here
                            '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagDown_14Nov.h5',
                            '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_JpsiPhiMagDown_12Dec.h5',
                            '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_D0piMagUp_12Dec.h5',
                            '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagUp_14Nov.h5',
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

# model = UNet().to(args['device'])
# use when loading pre-trained weights
model = torch.load('/share/lazy/pv-finder_model_repo/24/a6a99bc3871147f4a1007284ced5e156/artifacts/run_stats.pyt').to(args['device'])
# model.to("cuda:0")
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss = Loss(epsilon=1e-5,coefficient=args['asymmetry_parameter'])

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

#load_full_state(model, optimizer, '/share/lazy/pv-finder_model_repo/24/fa8a62697d5f490fb498f30c132eab7b/artifacts/run_stats.pyt')

run_name = 'dense_net* 1.3'

# tune kernel based on gpu
#torch.backends.cudnn.benchmark=True
train_iter = enumerate(trainNet(model, optimizer, loss, train_loader, val_loader, args['epochs'], notebook=True))
with mlflow.start_run(run_name = run_name) as run:
    mlflow.log_artifact('script_train_2.py')
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
            'Param: Learning Rate':args['lr'],
        }, step=i)