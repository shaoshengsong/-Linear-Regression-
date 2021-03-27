


import torch    
import torch.nn as nn 
import torch.nn.functional as F   
class LinearRegression(torch.nn.Module):
    def __init__(self, config):
        super(LinearRegression, self).__init__()
        self.predict = torch.nn.Linear(84, config.output_size)
        self.fc = nn.Sequential(
             nn.Linear(config.input_size, 120),
             nn.Linear(120, 84),
         )

    def forward(self, x):
        x = self.fc(x)
        x = self.predict(x)
        return x    
 
