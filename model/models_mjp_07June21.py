import torch
from torch import nn
import numpy as np

'''
This is a demo model intended to run and complete its training in as little time as possible. The performance 
is, therefore, expected to be quite low in comparison to other existing models.
'''
class ACN_1i4_DEMO(nn.Module):
    def __init__(self):
        super(ACN_1i4_DEMO, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x1 = self.dropout1(leaky(self.conv1(x)))
        x2 = self.dropout1(leaky(self.conv2(x1)))
        x3 = self.dropout1(leaky(self.conv3(x2)))
        x4 = self.dropout1(leaky(self.conv4(x3)))
        x5 = self.dropout1(leaky(self.conv5(x4)))
                   
        # Remove empty middle shape diminsion
        x = x5.view(x5.shape[0], x5.shape[-1])
                   
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
                                  
class ACN_2i4_DEMO(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_DEMO, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x1 = self.dropout(leaky(self.conv1(x)))
        x2 = self.dropout(leaky(self.conv2(x1)))
        x3 = self.dropout(leaky(self.conv3(x2)))
        x4 = self.dropout(leaky(self.conv4(x3)))
        x5 = self.dropout(leaky(self.conv5(x4)))
        x6 = self.dropout(leaky(self.conv6(x5)))

        ##  with a little luck, the following two lines instantiate the
        ##  6th layer and reshape its output to work as output to the
        ##  softplus activation
        x = x6.view(x6.shape[0], x6.shape[-1])

        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_DEMO(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_DEMO, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        # notice how there are less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFC = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )

        self.dropout = nn.Dropout(0.15)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## Since there is no way to exclude Xsq from being loaded in the notebook (that I know of), which 
        ## is the second input of neuronValues, we need to be changed to accomadate for this fact. 
        ## This would be done with the following code:
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        px1 = neuronValues[:, 2:4, :]  ## picks out the 1 & 2 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x1 = self.dropout(leaky(self.conv1(x)))
        x2 = self.dropout(leaky(self.conv2(x1)))
        x3 = self.dropout(leaky(self.conv3(x2)))
        x4 = self.dropout(leaky(self.conv4(x3)))
        x5 = self.dropout(leaky(self.conv5(x4)))
        x6 = self.dropout(leaky(self.conv6(x5)))

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        px1 = self.dropout(leaky(self.ppConv1(x1)))
        px1 = self.dropout(leaky(self.ppConv2(x1)))
        px1 = self.dropout(leaky(self.ppConv3(x1)))
        # Remove empty middle shape diminsion
        px1 = x1.view(x1.shape[0], x1.shape[-1])
        px1 = self.ppFC(x1)

        # Take the product of the two feature sets as an output layer. This can
        # be modified to a convolutional layer later, but this must be done in the
        # "transition" model.
        neuronValues = torch.nn.Softplus()(x0 * px1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class ACN_4i4_DEMO(nn.Module):
    def __init__(self):
        super(ACN_4i4_DEMO, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.ppConv3.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.dropout = nn.Dropout(0.15)
        

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        px1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x1 = self.dropout(leaky(self.conv1(x)))
        x2 = self.dropout(leaky(self.conv2(x1)))
        x3 = self.dropout(leaky(self.conv3(x2)))
        x4 = self.dropout(leaky(self.conv4(x3)))
        x5 = self.dropout(leaky(self.conv5(x4)))
        x6 = self.dropout(leaky(self.conv6(x5)))

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        px1 = self.dropout(leaky(self.ppConv1(x1)))
        px1 = self.dropout(leaky(self.ppConv2(x1)))
        px1 = self.dropout(leaky(self.ppConv3(x1)))
        px1 = self.ppFinalFilter(x1)
        px1 = x1.view(x1.shape[0], x1.shape[-1])

        ## Concatenated layers with conv layer tried and has negligible difference
        neuronValues = torch.nn.Softplus()(x0 * px1)

        return neuronValues

'''
Modified network architecture of benchmark with the following attributes:
NOTE: All attributes shared with benchmark are omitted
1. Three feature set using X, x, y.
2. 10 layer convolutional architecture for X feature set.
3. 4 layer conovlutional architecture for x and y feature set.
4. Takes element-wise product of the two feature sets for final layer.
7. Channel count follows the format:    01-20-10-10-10-10-07-05-01-01 (X), 20-10-10-01 (x, y),  20-01 (X, x, y)
8. Kernel size follows the format:      25-15-15-15-15-15-09-05-91 (X),    25-15-15-91 (x, y),  25-91 (X, x, y)
9. 4 skip connections, located at layers 3,5,7,9
'''
class ACN_1i4_10L_4S(nn.Module):
    def __init__(self):
        super(ACN_1i4_10L_4S, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels,
            out_channels=7,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv8 = nn.Conv1d(
            in_channels=self.conv7.out_channels,
            out_channels=3,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv9 = nn.Conv1d(
            in_channels=self.conv8.out_channels+self.conv7.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv9.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x01 = self.dropout1(leaky(self.conv1(x)))
        x2 = self.dropout1(leaky(self.conv2(x01)))
        x3 = self.dropout1(leaky(self.conv3(torch.cat([x01, x2], 1))))
        x4 = self.dropout1(leaky(self.conv4(x3)))
        x5 = self.dropout1(leaky(self.conv5(torch.cat([x3, x4], 1))))
        x6 = self.dropout1(leaky(self.conv6(x5)))
        x7 = self.dropout1(leaky(self.conv7(torch.cat([x5, x6], 1))))
        x8 = self.dropout1(leaky(self.conv8(x7)))
        x9 = self.dropout2(leaky(self.conv9(torch.cat([x7, x8], 1))))
                   
        # Remove empty middle shape diminsion
        x = x9.view(x9.shape[0], x9.shape[-1])
                   
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
                                  
class ACN_2i4_10L_4S_BN(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_10L_4S_BN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels,
            out_channels=7,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv8 = nn.Conv1d(
            in_channels=self.conv7.out_channels,
            out_channels=3,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv9 = nn.Conv1d(
            in_channels=self.conv8.out_channels+self.conv7.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv9.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.dropout = nn.Dropout(0.15)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bn8 = nn.BatchNorm1d(self.conv8.out_channels)
        self.bn9 = nn.BatchNorm1d(self.conv9.out_channels)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x01 = self.dropout(leaky(self.bn1(self.conv1(x))))
        x2 = self.dropout(leaky(self.bn2(self.conv2(x01))))
        x3 = self.dropout(leaky(self.bn3(self.conv3(torch.cat([x01, x2], 1)))))
        x4 = self.dropout(leaky(self.bn4(self.conv4(x3))))
        x5 = self.dropout(leaky(self.bn5(self.conv5(torch.cat([x3, x4], 1)))))
        x6 = self.dropout(leaky(self.bn6(self.conv6(x5))))
        x7 = self.dropout(leaky(self.bn7(self.conv7(torch.cat([x5, x6], 1)))))
        x8 = self.dropout(leaky(self.bn8(self.conv8(x7))))
        x9 = self.dropout(leaky(self.bn9(self.conv9(torch.cat([x7, x8], 1)))))

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x9)
        x = x.view(x.shape[0], x.shape[-1])

        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_P_10L_4S_BN(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_P_10L_4S_BN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels,
            out_channels=7,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv8 = nn.Conv1d(
            in_channels=self.conv7.out_channels,
            out_channels=3,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv9 = nn.Conv1d(
            in_channels=self.conv8.out_channels+self.conv7.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv9.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFC = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.dropout = nn.Dropout(0.15)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bn8 = nn.BatchNorm1d(self.conv8.out_channels)
        self.bn9 = nn.BatchNorm1d(self.conv9.out_channels)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## Since there is no way to exclude Xsq from being loaded in the notebook (that I know of) this
        ## needs to be changed to accomadate for this fact. This would be done with the following code:
        #x0 = neuronValues[:, 0:1, :]
        #x1 = neuronValues[:, 2:4, :]
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 1 & 2 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = self.dropout(leaky(self.bn1(self.conv1(x0))))
        x2 = self.dropout(leaky(self.bn2(self.conv2(x01))))
        x3 = self.dropout(leaky(self.bn3(self.conv3(torch.cat([x01, x2], 1)))))
        x4 = self.dropout(leaky(self.bn4(self.conv4(x3))))
        x5 = self.dropout(leaky(self.bn5(self.conv5(torch.cat([x3, x4], 1)))))
        x6 = self.dropout(leaky(self.bn6(self.conv6(x5))))
        x7 = self.dropout(leaky(self.bn7(self.conv7(torch.cat([x5, x6], 1)))))
        x8 = self.dropout(leaky(self.bn8(self.conv8(x7))))
        x9 = self.dropout(leaky(self.bn9(self.conv9(torch.cat([x7, x8], 1)))))

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x0 = self.finalFilter(x9)
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.dropout(leaky(self.ppConv1(x1)))
        x1 = self.dropout(leaky(self.ppConv2(x1)))
        x1 = self.dropout(leaky(self.ppConv3(x1)))
        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.ppFC(x1)

        # Take the product of the two feature sets as an output layer. This can
        # be modified to a convolutional layer later, but this must be done in the
        # "transition" model.
        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class ACN_4i4_P_10L_4S_BN(nn.Module):
    def __init__(self):
        super(ACN_4i4_P_10L_4S_BN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels,
            out_channels=7,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv8 = nn.Conv1d(
            in_channels=self.conv7.out_channels,
            out_channels=3,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv9 = nn.Conv1d(
            in_channels=self.conv8.out_channels+self.conv7.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv9.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.ppConv3.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        # layer for concatenated two feature sets
#        self.largeConv = nn.Conv1d(
#            in_channels=self.finalFilter.out_channels+self.ppFinalFilter.out_channels,
#            out_channels=1,
#            kernel_size = 91, # this is a totally random guess and has no logical backing
#            stride=1, # might be worth trying 2
#            padding=(91 - 1) // 2,
#        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.dropout = nn.Dropout(0.15)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bn8 = nn.BatchNorm1d(self.conv8.out_channels)
        self.bn9 = nn.BatchNorm1d(self.conv9.out_channels)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = self.dropout(leaky(self.bn1(self.conv1(x0))))
        x2 = self.dropout(leaky(self.bn2(self.conv2(x01))))
        x3 = self.dropout(leaky(self.bn3(self.conv3(torch.cat([x01, x2], 1)))))
        x4 = self.dropout(leaky(self.bn4(self.conv4(x3))))
        x5 = self.dropout(leaky(self.bn5(self.conv5(torch.cat([x3, x4], 1)))))
        x6 = self.dropout(leaky(self.bn6(self.conv6(x5))))
        x7 = self.dropout(leaky(self.bn7(self.conv7(torch.cat([x5, x6], 1)))))
        x8 = self.dropout(leaky(self.bn8(self.conv8(x7))))
        x9 = self.dropout(leaky(self.bn9(self.conv9(torch.cat([x7, x8], 1)))))

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x0 = self.finalFilter(x9)
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.dropout(leaky(self.ppConv1(x1)))
        x1 = self.dropout(leaky(self.ppConv2(x1)))
        x1 = self.dropout(leaky(self.ppConv3(x1)))
        x1 = self.ppFinalFilter(x1)
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        ## Concatenated layers with conv layer tried and has negligible difference
        neuronValues = torch.nn.Softplus()(x0 * x1)

        return neuronValues


'''
Modified network architecture of benchmark with the following attributes:
NOTE: All attributes shared with benchmark are omitted
1. Three feature set using X, x, y.
2. 8 layer convolutional architecture for X feature set.
3. 4 layer conovlutional architecture for x and y feature set.
4. Takes element-wise product of the two feature sets for softplus.
5. DenseNet skip connections; each layer's input are the features of each previous layers' outputs
7. Channel count follows the format:    01-20-10-10-10-10-10-01-01 (X), 02-10-01-01 (x, y)
8. Kernel size follows the format:      25-15-15-15-15-15-05-91 (X),    25-15-15-91 (x, y)
'''
## In order for this model to work, try an 8 layer, skip connection SimpleCNN 
## model as foundation
class ACN_1i4_8L_DenseNet(nn.Module):
    def __init__(self):
        super(ACN_1i4_8L_DenseNet, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv7.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x01 = self.dropout1(leaky(self.conv1(x)))
        x2 = self.dropout1(leaky(self.conv2(x01)))
        x3 = self.dropout1(leaky(self.conv3(torch.cat([x01, x2],1))))
        x4 = self.dropout1(leaky(self.conv4(torch.cat([x01,x2,x3],1))))
        x5 = self.dropout1(leaky(self.conv5(torch.cat([x01,x2,x3,x4],1))))
        x6 = self.dropout1(leaky(self.conv6(torch.cat([x01,x2,x3,x4,x5],1))))
        x7 = self.dropout2(leaky(self.conv7(torch.cat([x01,x2,x3,x4,x5,x6],1))))
        

        # Remove empty middle shape diminsion
        x = x7.view(x7.shape[0], x7.shape[-1])

        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
            
class ACN_2i4_8L_DenseNet_BN(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_8L_DenseNet_BN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels
                +self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.dropout = nn.Dropout(0.15)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bnFF = nn.BatchNorm1d(self.finalFilter.out_channels)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x01 = self.dropout(leaky(self.bn1(self.conv1(x))))
        x2 = self.dropout(leaky(self.bn2(self.conv2(x01))))
        x3 = self.dropout(leaky(self.bn3(self.conv3(torch.cat([x01,x2],1)))))
        x4 = self.dropout(leaky(self.bn4(self.conv4(torch.cat([x01,x2,x3],1)))))
        x5 = self.dropout(leaky(self.bn5(self.conv5(torch.cat([x01,x2,x3,x4],1)))))
        x6 = self.dropout(leaky(self.bn6(self.conv6(torch.cat([x01,x2,x3,x4,x5],1)))))
        x7 = self.dropout(leaky(self.bn7(self.conv7(torch.cat([x01,x2,x3,x4,x5,x6],1)))))

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.bnFF(self.finalFilter(torch.cat([x01,x2,x3,x4,x5,x6,x7],1)))
        x = x.view(x.shape[0], x.shape[-1])

        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_8L_DenseNet_BN(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_8L_DenseNet_BN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels
                +self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFC = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.dropout = nn.Dropout(0.15)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bnFF = nn.BatchNorm1d(self.finalFilter.out_channels)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## Since there is no way to exclude Xsq from being loaded in the notebook (that I know of) this
        ## needs to be changed to accomadate for this fact. This would be done with the following code:
        #x0 = neuronValues[:, 0:1, :]
        #x1 = neuronValues[:, 2:4, :]
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 1 & 2 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = self.dropout(leaky(self.bn1(self.conv1(x0))))
        x2 = self.dropout(leaky(self.bn2(self.conv2(x01))))
        x3 = self.dropout(leaky(self.bn3(self.conv3(torch.cat([x01, x2],1)))))
        x4 = self.dropout(leaky(self.bn4(self.conv4(torch.cat([x01,x2,x3],1)))))
        x5 = self.dropout(leaky(self.bn5(self.conv5(torch.cat([x01,x2,x3,x4],1)))))
        x6 = self.dropout(leaky(self.bn6(self.conv6(torch.cat([x01,x2,x3,x4,x5],1)))))
        x7 = self.dropout(leaky(self.bn7(self.conv7(torch.cat([x01,x2,x3,x4,x5,x6],1)))))
        
        x = self.bnFF(self.finalFilter(torch.cat([x01,x2,x3,x4,x5,x6,x7],1)))
        x = x.view(x.shape[0], x.shape[-1])
        
        x1 = self.dropout(leaky(self.ppConv1(x1)))
        x1 = self.dropout(leaky(self.ppConv2(x1)))
        x1 = self.dropout(leaky(self.ppConv3(x1)))
        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.ppFC(x1)

        # Take the product of the two feature sets as an output layer. This can
        # be modified to a convolutional layer later, but this must be done in the
        # "transition" model.
        neuronValues = torch.nn.Softplus()(x * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues


class ACN_4i4_8L_DenseNet_BN(nn.Module):
    def __init__(self):
        super(ACN_4i4_8L_DenseNet_BN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels
                +self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.ppConv3.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        # layer for concatenated two feature sets
        self.largeConv = nn.Conv1d(
            in_channels=self.finalFilter.out_channels+self.ppFinalFilter.out_channels,
            out_channels=1,
            kernel_size = 91, # this is a totally random guess and has no logical backing
            stride=1, # might be worth trying 2
            padding=(91 - 1) // 2,
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.dropout = nn.Dropout(0.15)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bnFF = nn.BatchNorm1d(self.finalFilter.out_channels)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = self.dropout(leaky(self.bn1(self.conv1(x0))))
        x2 = self.dropout(leaky(self.bn2(self.conv2(x01))))
        x3 = self.dropout(leaky(self.bn3(self.conv3(torch.cat([x01, x2],1)))))
        x4 = self.dropout(leaky(self.bn4(self.conv4(torch.cat([x01,x2,x3],1)))))
        x5 = self.dropout(leaky(self.bn5(self.conv5(torch.cat([x01,x2,x3,x4],1)))))
        x6 = self.dropout(leaky(self.bn6(self.conv6(torch.cat([x01,x2,x3,x4,x5],1)))))
        x7 = self.dropout(leaky(self.bn7(self.conv7(torch.cat([x01,x2,x3,x4,x5,x6],1)))))
        
        x = self.bnFF(self.finalFilter(torch.cat([x01,x2,x3,x4,x5,x6,x7],1)))
        x = x.view(x.shape[0], x.shape[-1])

        
        x1 = self.dropout(leaky(self.ppConv1(x1)))
        x1 = self.dropout(leaky(self.ppConv2(x1)))
        x1 = self.dropout(leaky(self.ppConv3(x1)))
        x1 = self.ppFinalFilter(x1)
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x * x1)

        return neuronValues  

    
'''
Modified network architecture of benchmark with the following attributes:
NOTE: All attributes shared with benchmark are omitted
1. Batch Normalization in each layer and at each step of training
2. Three feature set using X, x, y.
3. 10 layer convolutional architecture for X feature set.
4. 4 layer conovlutional architecture for x and y feature set.
5. Takes element-wise product of the two feature sets for softplus.
6. DenseNet skip connections; each layer's input are the features of each previous layers' outputs
7. Channel count follows the format:    01-25-20-15-15-10-10-10-05-01-01 (X), 02-10-01-01 (x, y)
8. Kernel size follows the format:      25-15-15-15-15-15-15-09-05-91 (X),    25-15-15-91 (x, y)
'''
## In order for this model to work, try an 8 layer, skip connection SimpleCNN 
## model as foundation
class ACN_1i4_10L_DenseNet_BN(nn.Module):
    def __init__(self):
        super(ACN_1i4_10L_DenseNet_BN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=1+self.conv1.out_channels,
            out_channels=20,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=1+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=15,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=1+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=15,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=1+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=1+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv7 = nn.Conv1d(
            in_channels=1+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )    
        
        self.conv8 = nn.Conv1d(
            in_channels=1+self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=5,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )
        
        self.conv9 = nn.Conv1d(
            in_channels=1+self.conv8.out_channels+self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.fc1 = nn.Linear(
            in_features=4000 * self.conv9.out_channels, out_features=4000
        )
        
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.35)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bn8 = nn.BatchNorm1d(self.conv8.out_channels)
        self.bn9 = nn.BatchNorm1d(self.conv9.out_channels)
        

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        
        x1 = self.dropout1(leaky(self.bn1(self.conv1(x))))
        x2 = self.dropout1(leaky(self.bn2(self.conv2(torch.cat([x,x1],1)))))
        x3 = self.dropout1(leaky(self.bn3(self.conv3(torch.cat([x,x1, x2],1)))))
        x4 = self.dropout1(leaky(self.bn4(self.conv4(torch.cat([x,x1,x2,x3],1)))))
        x5 = self.dropout1(leaky(self.bn5(self.conv5(torch.cat([x,x1,x2,x3,x4],1)))))
        x6 = self.dropout1(leaky(self.bn6(self.conv6(torch.cat([x,x1,x2,x3,x4,x5],1)))))
        x7 = self.dropout1(leaky(self.bn7(self.conv7(torch.cat([x,x1,x2,x3,x4,x5,x6],1))))) 
        x8 = self.dropout1(leaky(self.bn8(self.conv8(torch.cat([x,x1,x2,x3,x4,x5,x6,x7],1)))))
        x9 = self.dropout2(leaky(self.bn9(self.conv9(torch.cat([x,x1,x2,x3,x4,x5,x6,x7,x8],1)))))
        
        # Remove empty middle shape diminsion
        x = x9.view(x9.shape[0], x9.shape[-1])

        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
            
class ACN_2i4_10L_DenseNet_BN(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_10L_DenseNet_BN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=1+self.conv1.out_channels,
            out_channels=20,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=1+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=15,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv4 = nn.Conv1d(
            in_channels=1+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=15,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv5 = nn.Conv1d(
            in_channels=1+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv6 = nn.Conv1d(
            in_channels=1+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        self.conv7 = nn.Conv1d(
            in_channels=1+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )    
        
        self.conv8 = nn.Conv1d(
            in_channels=1+self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=5,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )
        
        self.conv9 = nn.Conv1d(
            in_channels=1+self.conv8.out_channels+self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        
        self.conv10 = nn.Conv1d(
            in_channels=1+self.conv9.out_channels+self.conv8.out_channels+self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )
        
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.35)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bn8 = nn.BatchNorm1d(self.conv8.out_channels)
        self.bn9 = nn.BatchNorm1d(self.conv9.out_channels)
        self.bn10 = nn.BatchNorm1d(self.conv10.out_channels)
        

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        
        x1 = self.dropout1(leaky(self.bn1(self.conv1(x))))
        x2 = self.dropout1(leaky(self.bn2(self.conv2(torch.cat([x,x1],1)))))
        x3 = self.dropout1(leaky(self.bn3(self.conv3(torch.cat([x,x1, x2],1)))))
        x4 = self.dropout1(leaky(self.bn4(self.conv4(torch.cat([x,x1,x2,x3],1)))))
        x5 = self.dropout1(leaky(self.bn5(self.conv5(torch.cat([x,x1,x2,x3,x4],1)))))
        x6 = self.dropout1(leaky(self.bn6(self.conv6(torch.cat([x,x1,x2,x3,x4,x5],1)))))
        x7 = self.dropout1(leaky(self.bn7(self.conv7(torch.cat([x,x1,x2,x3,x4,x5,x6],1))))) 
        x8 = self.dropout1(leaky(self.bn8(self.conv8(torch.cat([x,x1,x2,x3,x4,x5,x6,x7],1)))))
        x9 = self.dropout1(leaky(self.bn9(self.conv9(torch.cat([x,x1,x2,x3,x4,x5,x6,x7,x8],1)))))
        x10 = self.dropout2(leaky(self.bn10(self.conv10(torch.cat([x,x1,x2,x3,x4,x5,x6,x7,x8,x9],1)))))
        
        # Remove empty middle shape diminsion
        x = x10.view(x10.shape[0], x10.shape[-1])

        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_10L_DenseNet_BN(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_10L_DenseNet_BN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv7.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels
                +self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFC = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)
        self.conv6dropout = nn.Dropout(0.15)
        self.conv7dropout = nn.Dropout(0.15)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bnFF = nn.BatchNorm1d(self.finalFilter.out_channels)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## Since there is no way to exclude Xsq from being loaded in the notebook (that I know of) this
        ## needs to be changed to accomadate for this fact. This would be done with the following code:
        #x0 = neuronValues[:, 0:1, :]
        #x1 = neuronValues[:, 2:4, :]
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 1 & 2 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.bn1(self.conv1(x0)))
        x01 = self.conv1dropout(x01)
        x2 = leaky(self.bn2(self.conv2(x01)))
        x2 = self.conv2dropout(x2)
        x3 = leaky(self.bn3(self.conv3(torch.cat([x01, x2],1))))
        x3 = self.conv3dropout(x3)
        x4 = leaky(self.bn4(self.conv4(torch.cat([x01,x2,x3],1))))
        x4 = self.conv4dropout(x4)
        x5 = leaky(self.bn5(self.conv5(torch.cat([x01,x2,x3,x4],1))))
        x5 = self.conv5dropout(x5)
        x6 = leaky(self.bn6(self.conv6(torch.cat([x01,x2,x3,x4,x5],1))))
        x6 = self.conv6dropout(x6)
        x7 = leaky(self.bn7(self.conv7(torch.cat([x01,x2,x3,x4,x5,x6],1))))
        x7 = self.conv7dropout(x7)
        
        x = self.bnFF(self.finalFilter(torch.cat([x01,x2,x3,x4,x5,x6,x7],1)))
        x = x.view(x.shape[0], x.shape[-1])
        
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.ppFC(x1)

        # Take the product of the two feature sets as an output layer. This can
        # be modified to a convolutional layer later, but this must be done in the
        # "transition" model.
        neuronValues = torch.nn.Softplus()(x * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues


class ACN_4_8L_DenseNet_BN(nn.Module):
    def __init__(self):
        super(ACN_4_8L_DenseNet_BN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels+self.conv2.out_channels
                +self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels+self.conv3.out_channels
                +self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv7.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv7.out_channels+self.conv6.out_channels+self.conv5.out_channels+self.conv4.out_channels
                +self.conv3.out_channels+self.conv2.out_channels+self.conv1.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.ppConv3.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.ppFinalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        # layer for concatenated two feature sets
        self.largeConv = nn.Conv1d(
            in_channels=self.finalFilter.out_channels+self.ppFinalFilter.out_channels,
            out_channels=1,
            kernel_size = 91, # this is a totally random guess and has no logical backing
            stride=1, # might be worth trying 2
            padding=(91 - 1) // 2,
        )

        assert (
            self.largeConv.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)
        self.conv6dropout = nn.Dropout(0.15)
        self.conv7dropout = nn.Dropout(0.15)
        
        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)
        self.bn6 = nn.BatchNorm1d(self.conv6.out_channels)
        self.bn7 = nn.BatchNorm1d(self.conv7.out_channels)
        self.bnFF = nn.BatchNorm1d(self.finalFilter.out_channels)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.bn1(self.conv1(x0)))
        x01 = self.conv1dropout(x01)
        x2 = leaky(self.bn2(self.conv2(x01)))
        x2 = self.conv2dropout(x2)
        x3 = leaky(self.bn3(self.conv3(torch.cat([x01, x2],1))))
        x3 = self.conv3dropout(x3)
        x4 = leaky(self.bn4(self.conv4(torch.cat([x01,x2,x3],1))))
        x4 = self.conv4dropout(x4)
        x5 = leaky(self.bn5(self.conv5(torch.cat([x01,x2,x3,x4],1))))
        x5 = self.conv5dropout(x5)
        x6 = leaky(self.bn6(self.conv6(torch.cat([x01,x2,x3,x4,x5],1))))
        x6 = self.conv6dropout(x6)
        x7 = leaky(self.bn7(self.conv7(torch.cat([x01,x2,x3,x4,x5,x6],1))))
        x7 = self.conv7dropout(x7)
        
        x = self.bnFF(self.finalFilter(torch.cat([x01,x2,x3,x4,x5,x6,x7],1)))
        x = x.view(x.shape[0], x.shape[-1])

        
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        x1 = self.ppFinalFilter(x1)
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x * x1)

        return neuronValues  