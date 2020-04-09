import pandas as pd
import scipy.io as sio
from skimage import measure
import time
import numpy as np
from skimage import io
import os

## Get and arrange statistical data
# sbrt = pd.read_excel("G:\Projects\AutoQC\prostateDB.xlsx")
sbrt = pd.read_excel("H:\Public\Elguindi\CERR\CERR_core\headneckDB.xls")
## Filter by single column
# ex_1 = data.loc[data['SITE'].isin(['PROSTATE SBRT', 'PROSTATE SBRT POST-BRACHY', 'PROSTATE MODHYPO'])]

## Search Query Example
# sbrt = data.loc[ ((data['SITE'] == 'PROSTATE SBRT') | (data['SITE'] == 'PROSTATE SBRT POST-BRACHY') | (data['SITE'] == 'PROSTATE MODHYPO' )) & (data['qualityScore'] > 0)]
# sbrt = data.loc[ ((data['SITE'] == 'PROSTATE SBRT') | (data['SITE'] == 'PROSTATE SBRT POST-BRACHY') | (data['SITE'] == 'PROSTATE MODHYPO' )) &
#                  ((data['PROVIDER'] != 'ZELEFSKY, MICHAEL') & (data['PROVIDER'] != 'KOLLMEIER, MARISA A.') & (data['PROVIDER'] != 'MCBRIDE, SEAN MATTHEW') & (data['PROVIDER'] != 'GOROVETS, DANIEL JACOB'))]
# sbrt = data.loc[ ((data['SITE'] == 'PROSTATE BED WITH NODES') | (data['SITE'] == 'PROSTATE BED')) & (data['PROVIDER'] == 'ZELEFSKY, MICHAEL') ]

sbrt['volume_subtracted_cc_parotid_l'] = pd.Series(0, index=sbrt.index)
sbrt['volume_added_cc_parotid_l'] = pd.Series(0, index=sbrt.index)
sbrt['volume_parotid_l_cc'] = pd.Series(0, index=sbrt.index)
sbrt['pct_change_sub'] = pd.Series(0, index=sbrt.index)
sbrt['pct_change_add'] = pd.Series(0, index=sbrt.index)

i = 4
for k in range(0, sbrt['MRN'].size):
    if isinstance(sbrt['PlanCFileName'].iloc[k],str):
        dataPath = os.path.join(sbrt['basePath'].iloc[k], sbrt['PlanCFileName'].iloc[k])
        pixels = sio.loadmat(dataPath)
        volume = pixels['data'][i,9]
        initial = pixels['data'][i,7]
        size = pixels['data'][i,14]
        if not volume.any():
            print('no array')
        else:
            volume_initial = np.sum(initial[initial == 1]) * size[0, 0] * size[0, 1] * size[0, 2]
            volume_added = np.sum(volume[volume == 1])*size[0,0]*size[0,1]*size[0,2]
            volume_sub = np.sum(volume[volume == -1])*size[0,0]*size[0,1]*size[0,2]
            sbrt['volume_added_cc_parotid_l'].iloc[k] = volume_added
            sbrt['volume_subtracted_cc_parotid_l'].iloc[k] = volume_sub
            sbrt['volume_parotid_l_cc'].iloc[k] = volume_initial
            sbrt['pct_change_sub'].iloc[k] = volume_sub/volume_initial
            sbrt['pct_change_add'].iloc[k] = volume_added / volume_initial

# sbrt = sbrt[['DATETIME','PATIENTNAME','MRN','PROVIDER','SITE','Bladder_O_DCE','CTV_PROST_DCE','PenileBulb_DCE','Rectum_O_DCE','UrethraFoley_DCE','RectalSpacer_DCE','qualityScore','timeSpent','volume_added_cc','volume_subtracted_cc']]
sbrt.to_excel('volumeChange.xlsx')