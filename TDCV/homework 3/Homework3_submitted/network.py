import torch.nn as nn

class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv2d(3, 16, 8, stride=1, padding=0, bias=True),   #(inchannels,outchannels,kernelsize,.....)
                                nn.ReLU(inplace=True),
                                #nn.BatchNorm2d(16),
                                nn.MaxPool2d(2, stride=2, padding=0),
                                #nn.Dropout2d(),
                                nn.Conv2d(16, 7, 5, stride=1, padding=0, bias=True),
                                nn.ReLU(inplace=True),
                                #nn.BatchNorm2d(7),
                                nn.MaxPool2d(2, stride=2, padding=0)
                                #nn.Dropout2d()
                                )
        self.fc = nn.Sequential(
                                nn.Linear(12*12*7, 256),        #(infeatures,outfeatures)
                                #nn.Dropout(),
                                nn.Linear(256, 16)
                                )                             
        
    def forward(self, anchor, puller, pusher):
        
        anchor = self.conv(anchor); anchor = anchor.view(-1, 12*12*7); anchor = self.fc(anchor)
        puller = self.conv(puller); puller = puller.view(-1, 12*12*7); puller = self.fc(puller)
        pusher = self.conv(pusher); pusher = pusher.view(-1, 12*12*7); pusher = self.fc(pusher)
        
        return anchor, puller, pusher

    @property
    def is_cuda(self):
        
        return next(self.parameters()).is_cuda
