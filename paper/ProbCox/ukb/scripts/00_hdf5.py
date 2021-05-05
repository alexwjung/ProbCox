'''
Preparing the UKB data - cleaning and filtering

- remove individuals that withdraw consent
- remove individuals with prior CVD
- remove individuals with missing baseline factors
- bring data into long format

'''
import os
import shutil
import sys
import tqdm
import numpy as np

import pandas as pd
import h5py
import torch

ROOT_DIR = '/nfs/research1/gerstung/sds/sds-ukb-cancer/'

run_id = int(sys.argv[1])
print(run_id)

try:
    shutil.rmtree(ROOT_DIR + 'projects/ProbCox/data/prepared/train/event/' + str(run_id))
    shutil.rmtree(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id))
    shutil.rmtree(ROOT_DIR + 'projects/ProbCox/data/prepared/valid/event/' + str(run_id))
    shutil.rmtree(ROOT_DIR + 'projects/ProbCox/data/prepared/valid/censored/' + str(run_id))
    shutil.rmtree(ROOT_DIR + 'projects/ProbCox/data/prepared/test/event/' + str(run_id))
    shutil.rmtree(ROOT_DIR + 'projects/ProbCox/data/prepared/test/censored/' + str(run_id))
except:
    pass

os.mkdir(ROOT_DIR + 'projects/ProbCox/data/prepared/train/event/' + str(run_id))
os.mkdir(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id))
os.mkdir(ROOT_DIR + 'projects/ProbCox/data/prepared/valid/event/' + str(run_id))
os.mkdir(ROOT_DIR + 'projects/ProbCox/data/prepared/valid/censored/' + str(run_id))
os.mkdir(ROOT_DIR + 'projects/ProbCox/data/prepared/test/event/' + str(run_id))
os.mkdir(ROOT_DIR + 'projects/ProbCox/data/prepared/test/censored/' + str(run_id))


# - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def forward_fill(x):
    for ii in range(1, x.shape[0]):
        if np.sum(x[ii, :]) == 0:
            x[ii, :] = x[ii-1, :]
    return(x)

def _mean(x):
    x = np.asarray(x)
    x = x[x != '']
    x = x.astype(float)
    if x.shape[0] != 0:
        return(np.mean(x).tolist())
    else:
        return('')

ukb_idx_remove = [1074413, 1220436, 1322418, 1373016, 1484804, 1516618, 1681957, 1898968, 2280037, 2326194, 2492542, 2672990, 2719152, 2753503, 3069616, 3100207, 3114030, 3622841, 3666774, 3683210, 4167470, 4285793, 4335116, 4426913, 4454074, 4463723, 4470984, 4735907, 4739147, 4940729, 5184071, 5938951]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

actionable_codes = ["I20 (angina pectoris)", "I21 (acute myocardial infarction)", "I22 (subsequent myocardial infarction)", "I23 (certain current complications following acute myocardial infarction)", "I24 (other acute ischaemic heart diseases)", "I25 (chronic ischaemic heart disease)", "I60 (subarachnoid haemorrhage)", "I61 (intracerebral haemorrhage)", "I62 (other nontraumatic intracranial haemorrhage)", "I63 (cerebral infarction)", "I64 (stroke, not specified as haemorrhage or infarction)", "I65 (occlusion and stenosis of precerebral arteries, not resulting in cerebral infarction)", "I66 (occlusion and stenosis of cerebral arteries, not resulting in cerebral infarction)", "I67 (other cerebrovascular diseases)", "I68 (cerebrovascular disorders in diseases classified elsewhere)", "I69 (sequelae of cerebrovascular disease)", "I46 (cardiac arrest)", "I50 (heart failure)", 'G45 (transient cerebral ischaemic attacks and related syndromes)']

event_codes = ["I21 (acute myocardial infarction)", "I22 (subsequent myocardial infarction)", "I23 (certain current complications following acute myocardial infarction)", "I24 (other acute ischaemic heart diseases)"]


icd10_codes = pd.read_csv(ROOT_DIR + 'projects/ProbCox/data/icd10_codes.csv', header=None)
icd10_codes.iloc[:, 1] = icd10_codes.iloc[:, 1].apply(lambda x: x[20:])
icd10_code_names = np.asarray(icd10_codes.loc[icd10_codes.iloc[:, 1].apply(lambda x: x not in actionable_codes), 1])
icd10_code_names.shape
icd10_codes = icd10_codes.groupby(0).first()

data_iterator = pd.read_csv(ROOT_DIR + 'main/44968/ukb44968.csv', iterator=True, chunksize=1, nrows=1000, skiprows=lambda x: x in np.arange(1, 1000*run_id).tolist())

for _, dd in tqdm.tqdm(enumerate(data_iterator)):

    dd.reset_index(inplace=True)

    dd = dd.astype(str)
    dd = dd.replace('nan', '')

    # baseline
    eid = np.asarray(dd['eid']).astype(int)
    if eid in ukb_idx_remove:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('removed_id; ' + str(eid[0]))
            f.write('\n')
        continue

    sex = np.asarray(dd['31-0.0']).astype(int)
    birthyear = np.asarray(dd['34-0.0']).astype(str)[0]
    birthmonth = np.asarray(dd['52-0.0']).astype(str)[0]
    if len(birthmonth) == 1:
        birthmonth = '0' + birthmonth
    EOO_reason = np.asarray(dd['190-0.0']).astype(str)
    EOO = np.asarray(dd['191-0.0']).astype('datetime64[D]')
    birthdate = np.datetime64(birthyear + '-' + birthmonth, 'D')
    if EOO == EOO:
        pass
    else:
        EOO = np.datetime64('2021-03-01', 'D')[None]

    # administrative
    assessment_dates = np.asarray(dd[['53-'+str(ii)+'.0' for ii in range(0,4)]])
    BMI = np.asarray(dd[['21001-'+str(ii)+'.0' for ii in range(0,4)]])
    vigorous_activity = np.asarray(dd[['904-'+str(ii)+'.0' for ii in range(4)]])
    smoking = np.asarray(dd[['1239-'+str(ii)+'.0' for ii in range(4)]])
    alcohol = np.asarray(dd[['1558-'+str(ii)+'.0' for ii in range(4)]])
    LDL = np.asarray(dd[['30780-'+str(ii)+'.0' for ii in range(2)]])
    HDL = np.asarray(dd[['30760-'+str(ii)+'.0' for ii in range(2)]])
    triglyceride = np.asarray(dd[['30870-'+str(ii)+'.0' for ii in range(2)]])
    diastolic = np.asarray([_mean(dd[[str(ll)+'-'+str(ii)+'.'+str(jj) for jj in range(2) for ll in ['4079', 94]]]) for ii in range(4)])[None, :].astype(str)
    systolic = np.asarray([_mean(dd[[str(ll)+'-'+str(ii)+'.'+str(jj) for jj in range(2) for ll in ['4080', 95]]]) for ii in range(4)])[None, :].astype(str)
    idx = assessment_dates != ''
    assessment_dates = assessment_dates[idx]

    if (EOO - np.asarray([assessment_dates[0]]).astype('datetime64[D]')).astype(int) < 366:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('less_1year; ' + str(eid[0]))
            f.write('\n')
        continue
    if BMI[0, 0] != '':
        BMI = BMI[idx]
        BMI = BMI[BMI!='']
        BMI = np.concatenate((BMI[None, :], np.repeat(BMI[-1], assessment_dates.shape[0]-BMI.shape[0])[None, :]), axis=1)
        BMI = BMI.astype(float)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    if LDL[0, 0] != '':
        LDL = LDL[LDL!='']
        LDL = np.concatenate((LDL[None, :], np.repeat(LDL[-1], assessment_dates.shape[0]-LDL.shape[0])[None, :]), axis=1)
        LDL = LDL.astype(float)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    if HDL[0, 0] != '':
        HDL = HDL[HDL!='']
        HDL = np.concatenate((HDL[None, :], np.repeat(HDL[-1], assessment_dates.shape[0]-HDL.shape[0])[None, :]), axis=1)
        HDL = HDL.astype(float)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    if triglyceride[0, 0] != '':
        triglyceride = triglyceride[triglyceride!='']
        triglyceride = np.concatenate((triglyceride[None, :], np.repeat(triglyceride[-1], assessment_dates.shape[0]-triglyceride.shape[0])[None, :]), axis=1)
        triglyceride = triglyceride.astype(float)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    if diastolic[0, 0] != '':
        diastolic = diastolic[idx]
        diastolic = diastolic[diastolic!='']
        diastolic = np.concatenate((diastolic[None, :], np.repeat(diastolic[-1], assessment_dates.shape[0]-diastolic.shape[0])[None, :]), axis=1)
        diastolic = diastolic.astype(float)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    if systolic[0, 0] != '':
        systolic = systolic[idx]
        systolic = systolic[systolic!='']
        systolic = np.concatenate((systolic[None, :], np.repeat(systolic[-1], assessment_dates.shape[0]-systolic.shape[0])[None, :]), axis=1)
        systolic = systolic.astype(float)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    if vigorous_activity[0, 0] != '':
        vigorous_activity = vigorous_activity[idx]
        vigorous_activity = vigorous_activity[vigorous_activity!='']
        vigorous_activity = np.concatenate((vigorous_activity[None, :], np.repeat(vigorous_activity[-1], assessment_dates.shape[0]-vigorous_activity.shape[0])[None, :]), axis=1)
        vigorous_activity = (vigorous_activity.astype(int) >= 2).astype(int)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    if smoking[0, 0] != '':
        smoking = smoking[idx]
        smoking = smoking[smoking!='']
        smoking = np.concatenate((smoking[None, :], np.repeat(smoking[-1], assessment_dates.shape[0]-smoking.shape[0])[None, :]), axis=1)
        smoking = (smoking.astype(int) == 1).astype(int)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    if alcohol[0, 0] != '':
        alcohol = alcohol[idx]
        alcohol = alcohol[alcohol!='']
        alcohol = np.concatenate((alcohol[None, :], np.repeat(alcohol[-1], assessment_dates.shape[0]-alcohol.shape[0])[None, :]), axis=1)
        alcohol = (alcohol.astype(int) >=2).astype(int)
    else:
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('var_missing; ' + str(eid[0]))
            f.write('\n')
        continue
    assessment_dates = assessment_dates.astype('datetime64[D]')

    baseline = np.concatenate((np.repeat(sex, len(assessment_dates))[:, None], alcohol.T, smoking.T, BMI.T, LDL.T, HDL.T, triglyceride.T, diastolic.T, systolic.T, vigorous_activity.T), axis=1).astype(float)

    d_codes = []
    d_dates = []
    for ii in range(0, 3000, 2):
        try:
            a = np.asarray(dd[[str(130000 + ii) + '-0.0']])[0, 0]
            b = np.asarray(dd[[str(130000 + ii + 1) + '-0.0']])[0, 0]
            if np.logical_and(a != '', b != ''):
                d_codes.extend(np.asarray(icd10_codes.loc[str(130000 + ii + 1) + '-0.0']).tolist())
                d_dates.append(a)
        except:
            pass
    d_dates = np.asarray(d_dates).astype('datetime64[D]')
    d_codes = np.asarray(d_codes).astype('str')
    event = 0
    action = 0


    ll_events = []
    ll_actions = []


    # check if relevant codes are present
    for ii in range(len(d_codes)):
        if d_codes[ii] in actionable_codes:
            ll_actions.append(d_dates[ii])
            action = 1

        if d_codes[ii] in event_codes:
            ll_events.append(d_dates[ii])
            event = 1

    event_dates = np.asarray(ll_events).astype('datetime64[D]')
    action_dates = np.asarray(ll_actions).astype('datetime64[D]')

    if event:
        event_dates = np.min(event_dates)
        action_dates = np.min(action_dates)
        if (event_dates - action_dates).astype(int) >= 364:
            EOO = np.minimum(action_dates, EOO)
            EOO = np.minimum(np.datetime64('2020-03-01', 'D')[None], EOO)
            event = 0
        else:
            EOO = np.minimum(event_dates, EOO)
            EOO = (EOO - 365)
    elif action:
        action_dates = np.min(action_dates)
        EOO = np.minimum(action_dates, EOO)
        EOO = np.minimum(np.datetime64('2020-03-01', 'D')[None], EOO)

    else:
        EOO = np.minimum(np.datetime64('2020-03-01', 'D')[None], EOO)

    if EOO <= np.min(assessment_dates):
        with open(ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/removed.txt', "a") as f:
            f.write('event_prior; ' + str(eid[0]))
            f.write('\n')
            continue

    idx = np.logical_and(d_dates>= birthdate, d_dates<=EOO)
    d_dates = d_dates[idx]
    d_codes = d_codes[idx]

    dates = np.concatenate((birthdate[None], assessment_dates, d_dates, EOO))

    baseline = np.concatenate((np.zeros((1, 10)).astype(float), baseline, np.zeros((d_dates.shape[0] + 1, 10)).astype(float)))
    d_codes = np.concatenate((np.repeat('', 1 + assessment_dates.shape[0]), d_codes, np.repeat('', 1)))
    d_codes = (d_codes[:, None] == icd10_code_names).astype(int)

    idx_sort = np.argsort(dates)
    dates = dates[idx_sort]

    baseline = baseline[idx_sort, :]
    d_codes = d_codes[idx_sort, :]

    # collapse
    d_codes = np.concatenate([np.sum(d_codes[dates==ii, :], axis=0)[None, :] for ii in np.unique(dates)])
    baseline = np.concatenate([np.sum(baseline[dates==ii, :], axis=0)[None, :] for ii in np.unique(dates)])
    dates = np.unique(dates)

    # push through time:
    d_codes = np.minimum(1, np.cumsum(d_codes, axis=0))
    baseline = forward_fill(baseline)
    X = np.concatenate((baseline, d_codes), axis=1)

    time_diff = (dates[1:] - dates[:-1]).astype(int)

    time = np.concatenate((np.cumsum(np.concatenate((np.asarray([0]), time_diff)))[:-1, None], np.cumsum(np.concatenate((np.asarray([0]), time_diff)))[1:, None]), axis=1)
    time = np.concatenate((time, np.zeros((time.shape[0], 1))),axis=1)

    # start at assesment
    idx = np.where(dates == assessment_dates[0])[0][0]
    time = time[idx:, :]
    X = X[idx:-1, :]
    time[-1, -1] = event

    data = {'time': torch.from_numpy(time),
           'X': torch.from_numpy(X),
           'date': np.min(assessment_dates)[None]}

    if np.random.binomial(1, 0.7, (1,)):
        if event:
            torch.save(data, ROOT_DIR + 'projects/ProbCox/data/prepared/train/event/' + str(run_id) + '/'+ str(eid[0]))
        else:
            torch.save(data, ROOT_DIR + 'projects/ProbCox/data/prepared/train/censored/' + str(run_id) + '/'+ str(eid[0]))
    elif np.random.binomial(1, 0.666, (1,)):
        if event:
            torch.save(data, ROOT_DIR + 'projects/ProbCox/data/prepared/test/event/' + str(run_id) + '/'+ str(eid[0]))
        else:
            torch.save(data, ROOT_DIR + 'projects/ProbCox/data/prepared/test/censored/' + str(run_id) + '/'+ str(eid[0]))
    else:
        if event:
            torch.save(data, ROOT_DIR + 'projects/ProbCox/data/prepared/valid/event/' + str(run_id) + '/'+ str(eid[0]))
        else:
            torch.save(data, ROOT_DIR + 'projects/ProbCox/data/prepared/valid/censored/' + str(run_id) + '/'+ str(eid[0]))

            
print('finished')
