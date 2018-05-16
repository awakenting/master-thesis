# Data
This folder contains external data from third parties as well as generated data.

## External

- Response angles from Bhattacharyya et al. 2017 (see citation below):

 file name: LVsVersusSubtendedAngle.mat

 variables:
     - 'LVs': Size-to-speed ratios where L is the Diameter of the stimulus disk and V is the approach velocity.
     - 'subtendedAngleAtResponse': The visual angle at response time.

 If you want to use this data as well please cite the original reference: 

 Bhattacharyya, K., McLean, D. L., & MacIver, M. A. (2017). Visual threat assessment and reticulospinal encoding of calibrated responses in larval zebrafish. Current Biology, 27(18), 2751–2762.e6. https://doi.org/10.1016/j.cub.2017.08.012

 Example python code to load the data into a pandas dataframe:


 ```python
import numpy as np
import pandas as pd 
import scipy.io as sio

data = sio.loadmat('LVsVersusSubtendedAngle.mat')

data.keys()

[Out] dict_keys(['__header__', '__version__', '__globals__', 'LVs', 'subtendedAngleAtResponse'])

data['subtendedAngleAtResponse'].shape

[Out] (246, 1)

clean_dict = {'lv': np.squeeze(data['LVs']), 'resp_angle': np.squeeze(data['subtendedAngleAtResponse'])}

df = pd.DataFrame(clean_dict)

df.describe()

            lv 	resp_angle
count	246.000000	246.000000
mean	0.694752 	43.558036
std 	0.308021 	27.765487
min 	0.100000 	14.290141
25% 	0.451914 	28.034274
50% 	0.714518 	34.141328
75% 	0.971854 	46.764871
max 	1.195086 	169.885266
```