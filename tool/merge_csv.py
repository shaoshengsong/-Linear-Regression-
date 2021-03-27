
import pandas as pd
import os
outputfile ="./1.csv"
inputfile_dir = "./merge/"
for inputfile in os.listdir(inputfile_dir):
    fullpath = (os.path.join(inputfile_dir, inputfile))
    #print(fullpath)
    dt=pd.read_csv(fullpath, header=0)
    dt.to_csv(outputfile, mode='a', index=False,header=False)




