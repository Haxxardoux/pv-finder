import torch
import mlflow

from model.collectdata_mdsA import collect_data
from model.collectdata_poca_KDE import collect_data_poca
from model.alt_loss_A import Loss
from model.training import trainNet, select_gpu
from model.utilities import load_full_state, count_parameters, Params, save_to_mlflow

from model.autoencoder_models import UNet
from model.autoencoder_models import PerturbativeUNet
from model.autoencoder_models import UNetPlusPlus
from model.autoencoder_models import DenseNet

## params will be used in sequential order. For example, is batch_size=[64,128], then the first 
## loop will use a batch size of 64 for the first run and 128 for the second run. It will *not* 
## do all runs with 64 then all runs with 128.
##
## LIMITATIONS: works in sequence; must run script multiple times and change select_gpu() value
## to train over multiple GPUs at once. Also does not automatically load the weights from a
## previous run into the next run in the loop.
##
## IMPORTANT: if you add a new model to the 'models' params, you probably need to add this case 
## to the if-else block in the for loop. Otherwise, it will likely throw some error at you.
args = Params(
    batch_size=[64,64,64],
    device = select_gpu(1),
    epochs=[400,400,400],
    lr=[1e-6,1e-6,1e-6],
    experiment_name='Feb-2022',
    asymmetry_parameter=[2.5,2.5,2.5,2.5],
    run_name=['u-net 3','u-net++ 2','dense_net 2'],
    model = ['u-net','u-net++','dense_net']
)

## loop through arguments
for j in range(len(args.run_name)):
    ##  pv_HLT1CPU_D0piMagUp_12Dec.h5 + pv_HLT1CPU_MinBiasMagDown_14Nov.h5 contain 138810 events
    ##  pv_HLT1CPU_MinBiasMagUp_14Nov.h5 contains 51349
    ##  choose which to "load" and slices to produce 180K event training sample
    ##  and 10159 event validation sample
    train_loader = collect_data_poca(
                                  '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagDown_14Nov.h5',
                                  '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_JpsiPhiMagDown_12Dec.h5',
                                  '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_D0piMagUp_12Dec.h5',
                                  '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagUp_14Nov.h5',
                                   slice = slice(None,260000),
    ##                             device = device,
                                 batch_size=args.batch_size[j],
    ## if we are using a larger dataset (240K events, with the datasets above, and 11 GB of GPU memory),
    ## not the dataset will overflow the GPU memory; device=device will allow the data to move back
    ## and forth between the CPU and GPU memory. While this allows use of a larger dataset, it slows
    ## down performance by about 10%.  So comment out when not needed.
    ##                           device=args.device[j],
                                masking=True, shuffle=True,
                                load_A_and_B=True,
                                load_xy=True)

    # Validation dataset. You can slice to reduce the size.
    ## dataAA -> /share/lazy/sokoloff/ML-data_AA/
    val_loader = collect_data_poca(
    ##                          '/share/lazy/sokoloff/dataAA/pv_HLT1CPU_MinBiasMagDown_14Nov.h5',
                                '/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagUp_14Nov.h5',
    ##                            '/share/lazy/sokoloff/dataAA/pv_HLT1CPU_D0piMagUp_12Dec.h5',
                              batch_size=args.batch_size[j],
                              slice=slice(33000,None),
    ##                          device=args.device[j],
                              masking=True, shuffle=False,
                              load_A_and_B=True,
                              load_xy=True)

    mlflow.tracking.set_tracking_uri('file:/share/lazy/pv-finder_model_repo')
    mlflow.set_experiment(args.experiment_name)

    ## first if block: which model
    ## second (nested) if block: batch size w/ torch.load functionality
    ## TODO: make array/matrix of torch.load filepaths to correct weights and just loop through and load them in
    if args.model[j]=='u-net':
#        model = UNet().to(args.device)
        if args.batch_size[j]==64:
            model = torch.load('/share/lazy/pv-finder_model_repo/28/bffdb525299444a8bc847e2c536eadc5/artifacts/run_stats.pyt')
        elif args.batch_size[j]==128:
            model = torch.load('/share/lazy/pv-finder_model_repo/28/a923ddf744d647cc82de5493e935fd54/artifacts/run_stats.pyt')
    elif args.model[j]=='dense_net':
#        model = DenseNet().to(args.device)
        if args.batch_size[j]==64:
            model = torch.load('/share/lazy/pv-finder_model_repo/28/c3e01efe0d8244449dd5dc544111381f/artifacts/run_stats.pyt')
        elif args.batch_size[j]==128:   
            model = torch.load('/share/lazy/pv-finder_model_repo/28/2ee9d35844d74d828094c52ffa9d8c3d/artifacts/run_stats.pyt')
    elif args.model[j]=='p_u-net':
#        model = PerturbativeUNet().to(args.device)
        if args.batch_size[j]==64:
             model = torch.load('/share/lazy/pv-finder_model_repo/28/1756e382220541fdb0d12cc9ae349d85/artifacts/run_stats.pyt')
        elif args.batch_size[j]==128:
             model = torch.load('/share/lazy/pv-finder_model_repo/28/c323ef59477245659d9654be0fbca357/artifacts/run_stats.pyt')
    elif args.model[j]=='u-net++':
#        model = UNetPlusPlus().to(args.device)
        if args.batch_size[j]==64:
             model = torch.load('/share/lazy/pv-finder_model_repo/28/349ca7c0202342ed9549a39a3c68bdbe/artifacts/run_stats.pyt')
        elif args.batch_size[j]==128:
             model = torch.load('/share/lazy/pv-finder_model_repo/28/163fe193d4eb4adda44b850a8f9bf593/artifacts/run_stats.pyt')
    
    # use when loading pre-trained weights; comment out, otherwise
    #model = torch.load('/share/lazy/pv-finder_model_repo/27/7196c9de36914e2790a3106dcb0dcb1b/artifacts/run_stats.pyt').to(args.device[i])
    # sometimes need this for loading 3090 (keep commented unless you run into issues)
    #model.to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr[j])
    loss = Loss(epsilon=1e-5,coefficient=args.asymmetry_parameter[j])

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    avgEff = 0.0
    avgFP = 0.0
    
    run_name = args.run_name[j]
    
    train_iter = enumerate(trainNet(model, optimizer, loss, train_loader, val_loader, args.epochs[j], notebook=True))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact('script_train_lhcb-mc.py')
        for i, result in train_iter:
            print(result.cost)
            torch.save(model, 'run_stats.pyt')
            mlflow.log_artifact('run_stats.pyt')

            ##
            ## code for calculating average eff and fp over last 10 epochs
            ##
            # If we are on the last 10 epochs but NOT the last epoch
            if(i >= args.epochs[j]-10):
                avgEff += result.eff_val.eff_rate
                avgFP += result.eff_val.fp_rate

            # If we are on the last epoch
            if(i == args.epochs[j]-1):
                print('Averaging...\n')
                avgEff/=10
                avgFP/=10
                mlflow.log_metric('10 Eff Avg.', avgEff)
                mlflow.log_metric('10 FP Avg.', avgFP)
                print('Average Eff: ', avgEff)
                print('Average FP Rate: ', avgFP)

            ##
            ## code for saving info to mlflow
            ##
            save_to_mlflow({
                'Metric: Training loss':result.cost,
                'Metric: Validation loss':result.val,
                'Metric: Efficiency':result.eff_val.eff_rate,
                'Metric: False positive rate':result.eff_val.fp_rate,
#                'Param: Events':events,
                'Param: Asymmetry':args.asymmetry_parameter[j],
                'Param: Batch Size':args.batch_size[j],
                'Param: Epochs':args.epochs[j],
                'Param: Learning Rate':args.lr[j],
                'Param: Parameters':parameters,
            }, step=i)