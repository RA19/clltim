from celery import shared_task, current_task, states
from dateutil.relativedelta import relativedelta
from django.utils import timezone
from prediction.utils import percent, csv_row
from prediction.models import Prediction, PredictionStatus, CeleryTaskPhase
from django.conf import settings
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import csv
import json
from scipy.stats import randint as sp_randint
from sklearn import preprocessing
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
warnings.filterwarnings(module='sklearn*', action='ignore', category=RuntimeWarning)
warnings.filterwarnings(module='sklearn*', action='ignore', category=UserWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet, SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, accuracy_score
from sklearn import metrics
import xgboost as xgb
import seaborn as sns
import os
import logging

logger = logging.getLogger('celery')


# SCALING AND NAN TO MEAN FUNCITONS
def Nan2Mean(b):
    a = np.copy(b)
    # Nan to mean and Scaling
    col_mean = np.nanmean(a, axis=0)
    #    print('infinity probs found before:' +str(np.sum(np.logical_and(~np.isnan(col_mean),~np.isfinite(col_mean)))))
    #    print('nan  probs found before:' +str(np.sum(np.isnan(col_mean))))
    col_mean[np.isnan(col_mean)] = 0;
    if np.sum(~np.isfinite(col_mean)) > 0:

        col_cnt = 0
        for c in col_mean:
            if np.logical_or(str(c) == '-inf', str(c) == 'inf'):
                #                print(col_cnt)
                miny = np.nanmin(a[np.isfinite(a[:, col_cnt]), col_cnt])
                maxy = np.nanmax(a[np.isfinite(a[:, col_cnt]), col_cnt])
                Listy = np.argwhere(np.logical_and(~np.isfinite(a[:, col_cnt]), ~np.isnan(a[:, col_cnt])))
                for l in range(np.shape(Listy)[0]):
                    if str(a[Listy[l][0], col_cnt]) == '-inf':
                        a[Listy[l][0], col_cnt] = miny
                    elif str(a[Listy[l][0], col_cnt]) == 'inf':
                        a[Listy[l][0], col_cnt] = maxy
            col_cnt = col_cnt + 1
    col_mean = np.nanmean(a, axis=0)
    col_mean[np.isnan(col_mean)] = 0;


    # Find indicies that you need to replace
    inds = np.where(np.isnan(a))


    a[inds] = np.take(col_mean, inds[1])

    return a, col_mean


def Nan2Mean_useColmean(b, col_mean):
    a = np.copy(b)
    inds = np.where(np.isnan(a))
    a[inds] = np.take(col_mean, inds[1])

    return a


def update_state(step, total_steps, phase):
    current_task.update_state(state='PROGRESS', meta={'percent': percent(step, total_steps), 'phase': phase})


@shared_task
def process_ml(prediction_id):

    start = timezone.now()

    phase1_total_steps = 7
    phase1_steps = 0

    phase2_total_steps = 5
    phase2_steps = 0

    try:
        prediction = Prediction.objects.get(id=prediction_id)
        prediction.status = PredictionStatus.PROGRESS.name
        prediction.save()
    except Exception as e:
        current_task.update_state(state=states.FAILURE)
        logger.error("Error CODE: 1000 -> %s", str(e))
        raise Exception("Error CODE: 1000")

    logger.debug("Phase 1/5: Initializing CLL-TIM")

    phase1_steps += 1
    update_state(phase1_steps, phase1_total_steps, CeleryTaskPhase.PHASE_1.value)

    filename = '{id}.csv'.format(id=str(prediction.id))
    csv_path = os.path.join(settings.MEDIA_ROOT, os.path.join('predictions', str(prediction.id)))
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    csv_path = os.path.join(csv_path, filename)

    # Creating CSV file From the Diagnosis Form
    if prediction.via_form:
        field_names = ['PatientID', 'Diag_Date', 'Prediction_Point', 'Variable_Category', 'Variable_Name',
                       'Date_of_Test', 'Value_of_Test', 'Units_of_Test', ]
        try:
            with open(csv_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(field_names)
                diagnosis_data = json.loads(prediction.diagnosis_data)
                baseline = diagnosis_data['baseline']
                patient_id = baseline['patient_id']
                diag_date = baseline['diag_date']
                pred_point = baseline['prediction_point']

                # Writing BaseLine
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'binet_stage', baseline['binet_stage']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'ighv_unmut', baseline['ighv_unmut']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'ecog', baseline['ecog']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'age', baseline['age']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'beta2m-baseline', baseline['beta2m']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'cd38', baseline['cd38']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'zap70', baseline['zap70']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'gender', baseline['gender']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'del13', baseline['del13']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'tri12', baseline['tri12']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'del11', baseline['del11']))
                writer.writerow(csv_row(patient_id, diag_date, pred_point, 'del17', baseline['del17']))

                # Writing Routine Lab Tests
                routines = diagnosis_data['routine_tests']
                for index, test in enumerate(routines, start=1):
                    lab_date = test["lab-date"]
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'beta2m', test["beta2m"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'leuk', test["leuk"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'thr', test["thr"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'aec', test["aec"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'haem', test["haem"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'urat', test["urat"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'baso', test["baso"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'trans', test["trans"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'psa', test["psa"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'smudge', test["smudge"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'amy', test["amy"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'alc', test["alc"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'mic', test["mic"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'mcv', test["mcv"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'prom', test["prom"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'crp', test["crp"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'gluc', test["gluc"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'crea', test["crea"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'creak', test["creak"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'hema', test["hema"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'egfr', test["egfr"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'eos', test["eos"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'ldh', test["ldh"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'neu', test["neu"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'mch', test["mch"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'mchc', test["mchc"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'blast', test["blast"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'myel', test["myel"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'metamyel', test["metamyel"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'alb', test["alb"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'alp', test["alp"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'inr', test["inr"], lab_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'na-plus', test["na-plus"], lab_date))

                # Writing Bloodculture Tests
                bloodculture_tests = diagnosis_data['bloodculture']
                for index, bloodculture_test in enumerate(bloodculture_tests, start=1):
                    infec_date = bloodculture_test['infec-date']
                    infec = bloodculture_test['infec']
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'infec', infec, infec_date))

                # Writing Pathology Tests
                pathology_tests = diagnosis_data['pathology']
                for index, pathology_test in enumerate(pathology_tests, start=1):
                    inflam_date = pathology_test['inflam-date']
                    inflam = pathology_test['inflam']
                    rare_path_date = pathology_test['rare-path-date']
                    rare_path = pathology_test['rare-path']
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'inflam', inflam, inflam_date))
                    writer.writerow(csv_row(patient_id, diag_date, pred_point, 'rare-path', rare_path, rare_path_date))

        except Exception as e:
            current_task.update_state(state=states.FAILURE)
            logger.error("Error CODE: 1001 -> %s", str(e))
            raise Exception("Error CODE: 1001")

    # Closing any open figures
    plt.close('all')

    logger.debug("Phase 1/5: Initializing CLL-TIM")
    phase1_steps += 1
    update_state(phase1_steps, phase1_total_steps, CeleryTaskPhase.PHASE_1.value)

    ###############################################################################
    # VARIABLES CURRENTLY USED FOR BETA CLL-TIM SERVER 
    ###############################################################################
    BaselineList = ['Binet Stage', 'IGHV_unmut', 'ECOG', 'FAMCLL', 'Age', 'Beta2m', 'CD38', 'ZAP70', 'Gender', 'del13',
                    'tri12', 'del11', 'del17']
    BaselineList_Orig = ['binet', 'umut', 'WHOPERFORMANCE', 'FAMCLL', 'Age', 'beta2m', 'CD38', 'ZAP70', 'Gender',
                         'del13', 'tri12', 'del11', 'del17']
    LabList = ['NA+', 'ALB', 'ALP', 'INR', 'BETA2M', 'LEUK', 'THR', 'AEC', 'HAEM', 'URAT', 'BASO', 'TRANS', 'PSA',
               'SMUDGE', 'AMY', 'ALC', 'MIC', 'MCV', 'PROM', 'CRP', 'GLUC', 'CREA', 'CREAK', 'HEMA',
               'EGFR', 'EOS', 'LDH', 'NEU', 'MCH', 'MCHC', 'BLAST', 'MYEL', 'METAMYEL']
    LabList_Orig = ['NA+', 'ALB', 'ALP', 'INR', 'B2M', 'LEUK', 'THR', 'NPU01960', 'HAEM', 'NPU03688', 'BAS', 'NPU03607',
                    'NPU08669', 'NPU17597', 'NPU19653', 'LYM', 'NPU26631', 'NPU01944', 'NPU03974', 'CRP', 'GLUC', 'CRE',
                    'NPU19656', 'NPU01961',
                    'EGFR', 'EOS', 'LDH', 'NEU', 'NPU02320', 'NPU02321', 'NPU03972', 'NPU03976', 'NPU03978']
    BloodcultureList = ['Infec']

    phase1_steps += 1
    update_state(phase1_steps, phase1_total_steps, CeleryTaskPhase.PHASE_1.value)

    ###############################################################################
    # Data Loading
    ###############################################################################
    TransList = np.load(os.path.join(settings.ML_DATA, 'TrainsListing.npy'))

    phase1_steps += 1
    update_state(phase1_steps, phase1_total_steps, CeleryTaskPhase.PHASE_1.value)

    # Loading Patient Data
    PData_CV = csv_path
    P01 = pd.read_csv(PData_CV, sep=',', float_precision='round_trip')

    phase1_steps += 1
    update_state(phase1_steps, phase1_total_steps, CeleryTaskPhase.PHASE_1.value)

    ToRemo = np.squeeze(np.argwhere(np.logical_and(P01['Variable_Category'] == 'Lab', pd.isnull(P01['Value_of_Test']))))
    P01 = P01.drop(ToRemo).reset_index()

    # Loading Dates to Int Conversion
    Date2Int = pd.read_csv(os.path.join(settings.ML_DATA, 'Date2Int.csv'), sep=',', float_precision='round_trip')

    # Date Conversions
    try:
        DiagDate = np.argwhere(Date2Int['F3'] == P01['Diag_Date'][0])[0][0]
    except Exception as e:
        current_task.update_state(state=states.FAILURE)
        logger.error("Error CODE: 1002 -> %s", str(e))
        raise Exception("Error CODE: 1002")

    P01['Diag_Date'] = DiagDate
    PredPoint = np.argwhere(Date2Int['F3'] == P01['Prediction_Point'][0])[0][0]
    P01['Prediction_Point'] = PredPoint
    for n in range(np.shape(P01)[0]):
        update_state(n, np.shape(P01)[0], CeleryTaskPhase.PHASE_1.value)
        P01['Date_of_Test'][n] = np.argwhere(Date2Int['F3'] == P01['Date_of_Test'][n])[0][0]
        P01['Diag_Date'][n]
        P01['Date_of_Test'][n]

    phase1_steps += 1
    update_state(phase1_steps, phase1_total_steps, CeleryTaskPhase.PHASE_1.value)

    # Loading Feature Decoding Information and Training Data
    LabbiesNames = pd.read_csv(os.path.join(settings.ML_DATA, 'LabTestFeatStruc.csv'), sep=',',
                               float_precision='round_trip')
    FeatNamesMatrix = np.load(os.path.join(settings.ML_DATA, 'FeatNamesMatrix_7288.npy'))
    FeatListNames = np.load(os.path.join(settings.ML_DATA, 'FeatListNames_7288.npy'))
    TrainMatrix = np.load(os.path.join(settings.ML_DATA, 'DanishTMatrix.npy'))

    # CLL-TIM LOADING
    Collect_CVData = np.load(os.path.join(settings.ML_DATA, '_Collect_CVData.npy'))
    Collect_ModelNames = np.load(os.path.join(settings.ML_DATA, '_Collect_ModelNames.npy'))
    CurrEnsSize = 28
    ToKeep = np.load(os.path.join(settings.ML_DATA, '28_ToKeep.npy'))
    Collect_CVData_Chosen = Collect_CVData[ToKeep, :]
    Collect_ModelNames_Chosen = Collect_ModelNames[ToKeep, :]
    Collect_FeatIndies = np.load(os.path.join(settings.ML_DATA, 'CLL_TIM_FeatIndexes.npy'))
    N_Chosen = len(ToKeep)
    ALLFeats = np.load(os.path.join(settings.ML_DATA, 'ALLFeats_.npy'))
    FuncFeatNames = pd.read_csv(os.path.join(settings.ML_DATA, 'FeatureFunctionalNames.csv'), sep='\t', header=None)

    FAllInds = 0
    for nn in range(0, CurrEnsSize):
        FAllInds = np.append(FAllInds, Collect_FeatIndies.item()[Collect_CVData_Chosen[nn, 0]])

    FAInds = np.unique(FAllInds)

    phase1_steps += 1
    update_state(phase1_steps, phase1_total_steps, CeleryTaskPhase.PHASE_1.value)

    ###############################################################################
    # Setting Up Patient Feature Vector
    ###############################################################################
    CT_FNM = FeatNamesMatrix[FAInds, :]
    CT_FNM_Baseline = CT_FNM[CT_FNM[:, 1] == 'Baseline', :]
    CT_FNM_Labka = np.copy(LabbiesNames)
    CT_FNM_BC = CT_FNM[CT_FNM[:, 2] == 'BC', :]

    logger.debug("Phase 2/5: Encoding Features from Patient Data")
    phase2_steps += 1
    update_state(phase2_steps, phase2_total_steps, CeleryTaskPhase.PHASE_2.value)
    ###############################################################################
    # Feature Extraction
    ###############################################################################
    PPoint = P01['Prediction_Point'][0]  # Extracting Prediction Point

    ################################################################################
    # Extracting Infection Dates and Applying Infection Features
    InfecDates = P01['Date_of_Test']['Infec' == P01['Variable_Name']]
    ToKeep1 = (PPoint - InfecDates) > 0  # No Future Points
    Infec_90 = PPoint - InfecDates[np.logical_and(InfecDates >= (PPoint - 90), ToKeep1)]
    Infec_365 = PPoint - InfecDates[np.logical_and(InfecDates >= (PPoint - 365), ToKeep1)]
    Infec_Inf = PPoint - InfecDates[np.logical_and(InfecDates >= (PPoint - 365 * 20), ToKeep1)]

    Infec_90 = np.int64(Infec_90.astype(None))
    SInfec_90 = Infec_90[np.argsort(-Infec_90)]
    Infec_365 = np.int64(Infec_365.astype(None))

    SInfec_365 = Infec_365[np.argsort(-Infec_365)]
    Infec_Inf = np.int64(Infec_Inf.astype(None))
    SInfec_Inf = Infec_Inf[np.argsort(-Infec_Inf)]

    phase2_steps += 1
    update_state(phase2_steps, phase2_total_steps, CeleryTaskPhase.PHASE_2.value)

    # if no infections = zero
    IFeats = np.zeros((12, 1))

    if len(SInfec_Inf) >= 3:
        Slp = np.polyfit(SInfec_Inf, np.arange(1, len(SInfec_Inf) + 1), 1)
        IFeats[0] = Slp[0]  # slope1_coefa

    if len(SInfec_90) >= 3:
        Slp = np.polyfit(SInfec_90, np.arange(1, len(SInfec_90) + 1), 2)
        IFeats[1] = Slp[2]  # slope2_coefc

    if len(Infec_90 > 0):
        IFeats[2] = np.nanmean(Infec_90)
        IFeats[5] = np.nanmin(Infec_90)

        IFeats[7] = scipy.stats.skew(Infec_90)
    if len(Infec_365) > 0:
        IFeats[3] = np.nanmean(Infec_365)
        IFeats[4] = np.nanstd(Infec_365)
    if len(Infec_Inf) > 0:
        IFeats[6] = np.nanmax(Infec_Inf)

        IFeats[8] = scipy.stats.kurtosis(Infec_Inf, axis=0, fisher=False)

    if len(SInfec_Inf) == 2:
        Changes = SInfec_Inf[0] - SInfec_Inf[1]
        IFeats[9] = np.nanmin(Changes)
        IFeats[11] = np.nanmean(Changes)
    elif len(SInfec_Inf) > 2:

        Changes = SInfec_Inf[0:-1] - SInfec_Inf[1:];
        IFeats[9] = np.nanmin(Changes)
        IFeats[11] = np.nanmean(Changes)

    if len(SInfec_365) == 2:
        Changes = SInfec_365[0] - SInfec_365[1]
        IFeats[10] = np.nanmean(Changes)
    elif len(SInfec_365) > 2:

        Changes = SInfec_365[0:-1] - SInfec_365[1:];
        IFeats[10] = np.nanmean(Changes)

    if len(Infec_90) == 0:
        IFeats[1] = 0
        IFeats[2] = 0
        IFeats[5] = 0
        IFeats[7] = 0
    elif len(Infec_90) == 1:
        IFeats[7] = np.nan
        IFeats[1] = np.nan
    elif len(Infec_90) == 2:
        IFeats[1] = np.nan

    if len(Infec_365) == 0:
        IFeats[3] = 0
        IFeats[4] = 0
        IFeats[10] = 0

    elif len(Infec_365) == 1:
        IFeats[10] = np.nan

    if len(Infec_Inf) == 0:
        IFeats[0] = 0
        IFeats[6] = 0
        IFeats[8] = 0
        IFeats[9] = 0
        IFeats[11] = 0
    elif len(Infec_Inf) == 1:
        IFeats[8] = np.nan
        IFeats[0] = np.nan
        IFeats[9] = np.nan
        IFeats[11] = np.nan
    elif len(Infec_Inf) == 2:
        IFeats[0] = np.nan
    #
   
    IFeats = np.squeeze(IFeats)
    Infec_P = np.copy(np.transpose(IFeats))

    phase2_steps += 1
    update_state(phase2_steps, phase2_total_steps, CeleryTaskPhase.PHASE_2.value)

    ################################################################################

    ################################################################################
    # Extracting baseline variables and creating baseline features

    Baseline_P = np.zeros((1, 32))
    # Going through Each of the Baseline Features Features and checking if they are in Patient Data
    bcnt = -1
    for CurrBVar in CT_FNM_Baseline[:, 0]:
        CB = CurrBVar.split('_')[0]
        bcnt = bcnt + 1

        CurrBVar_T = BaselineList[np.argwhere(CB == np.array(BaselineList_Orig))[0][0]]  # translated baseline name
        ConVal = -99
        if np.sum(CurrBVar_T == P01['Variable_Name']) > 0:

            Where = np.argwhere(CurrBVar_T == P01['Variable_Name'])[0][0]

            if 'binet' in CB:

                if '_A' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 'A':
                        ConVal = 1
                    else:
                        ConVal = 0
                if '_B' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 'B':
                        ConVal = 1
                    else:
                        ConVal = 0

                if '_C' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 'C':
                        ConVal = 1
                    else:
                        ConVal = 0
                if '_NA' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 'NA':
                        ConVal = 1
                    else:
                        ConVal = 0

            elif 'Age' in CB:
                ConVal = np.int64(P01['Value_of_Test'][Where]) / 100

            elif 'WHOPERFORMANCE' in CB:

                if 'Can Walk' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 1:
                        ConVal = 1
                    else:
                        ConVal = 0
                if 'Good Condition' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 0:
                        ConVal = 1
                    else:
                        ConVal = 0


            elif 'Gender' in CB:
                if P01['Value_of_Test'][Where] == 'Female':
                    ConVal = 1
                elif P01['Value_of_Test'][Where] == 'Male':
                    ConVal = 0
                else:
                    ConVal = -99
            elif 'umut' in CB:
                if 'umut_1' in CurrBVar:

                    if P01['Value_of_Test'][Where] == 'Unmutated':
                        ConVal = 1
                    else:
                        ConVal = 0
                if 'umut_0' in CurrBVar:

                    if P01['Value_of_Test'][Where] == 'Mutated':
                        ConVal = 1
                    else:
                        ConVal = 0
                if 'umut_NA' in CurrBVar:

                    if P01['Value_of_Test'][Where] == 'NA':
                        ConVal = 1
                    else:
                        ConVal = 0
            else:
                if '_0' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 'No':
                        ConVal = 1
                    else:
                        ConVal = 0
                if '_1' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 'Yes':
                        ConVal = 1
                    else:
                        ConVal = 0

                if '_NA' in CurrBVar:
                    if P01['Value_of_Test'][Where] == 'NA':
                        ConVal = 1
                    else:
                        ConVal = 0

        Baseline_P[0, bcnt] = ConVal
    Baseline_P = np.squeeze(np.copy((Baseline_P)))
    Baseline_P[Baseline_P == -99] = 0

    phase2_steps += 1
    update_state(phase2_steps, phase2_total_steps, CeleryTaskPhase.PHASE_2.value)

    ################################################################################

    ################################################################################
    ### Extracting Lab Variables and Creating Lab Features
    Lab_P = np.empty((1, 120))
    Lab_P[:] = np.nan
    PPoint = P01['Prediction_Point'][0]
    VV = np.copy(CT_FNM_Labka)
    v = -1
    lcnt = -1
    for CL in CT_FNM_Labka[:, 0]:
        v = v + 1
        lcnt = lcnt + 1
        CurrLVar_T = LabList[np.argwhere(CL == np.array(LabList_Orig))[0][0]]  # translated baseline name
        ConVal = -99
        if np.sum(CurrLVar_T == P01['Variable_Name']) > 0:
            ConVal = -99
            Where = CurrLVar_T == P01['Variable_Name']
            VChunk = np.squeeze(
                np.column_stack((P01['Date_of_Test'][Where], P01['Value_of_Test'][Where])).astype(float))

            if VV[v, 1] == 365:
                LBack = 365
            elif VV[v, 1] == 90:
                LBack = 90
            else:
                LBack = 365 * 10

            if np.size(VChunk) > 2:
                ToKeep1 = (PPoint - VChunk[:, 0]) > 0  # No Future Points
                ToKeep2 = VChunk[:, 0] >= (PPoint - LBack)
            else:
                ToKeep1 = (PPoint - VChunk[0]) > 0
                ToKeep2 = VChunk[0] >= (PPoint - LBack)

            ToKeep = np.logical_and(ToKeep1, ToKeep2)
            VChunk = VChunk[ToKeep, :]
            if len(VChunk) != 0:
                CLDates = VChunk[:, 0]
                CLValues = VChunk[:, 1]
                # Checking for same tests on the same day
                Un_Dates = np.unique(CLDates)
                Un_Dates_L = np.shape(Un_Dates)[0]
                CLDa = np.zeros((Un_Dates_L, 1))
                CLVa = np.zeros((Un_Dates_L, 1))
                if Un_Dates_L != np.shape(CLDates):
                    for ud in range(Un_Dates_L):
                        CLDa[ud] = Un_Dates[ud]
                        CLVa[ud] = np.nanmean(CLValues[CLDates == Un_Dates[ud]])
                    if len(CLDa) > 1:
                        CLDates = np.squeeze(np.copy(CLDa))
                        CLValues = np.squeeze(np.copy(CLVa))

                if VV[v, 2] == 'Instance':
                    if VV[v, 3] == 'Cnt':
                        CurrFeatVal = np.shape(CLValues)[0]
                    elif VV[v, 3] == 'latest':
                        CurrFeatVal = np.min(PPoint - CLDates)
                    elif VV[v, 3] == 'Mean':
                        CurrFeatVal = np.nanmean(PPoint - CLDates)
                elif VV[v, 2] == 'nanmax':
                    CurrFeatVal = np.nanmax(CLValues)
                elif VV[v, 2] == 'nanmin':
                    CurrFeatVal = np.nanmin(CLValues)
                elif VV[v, 2] == 'nanmean':
                    CurrFeatVal = np.nanmean(CLValues)
                elif VV[v, 2] == 'nanmedian':
                    CurrFeatVal = np.nanmedian(CLValues)
                elif VV[v, 2] == 'nanstd':
                    CurrFeatVal = np.nanstd(CLValues)
                elif VV[v, 2] == 'kurtosis':
                    CurrFeatVal = scipy.stats.kurtosis(CLValues, axis=0, fisher=False)
                elif VV[v, 2] == 'skewness':
                    CurrFeatVal = scipy.stats.skew(CLValues)

                if Un_Dates_L >= 3:
                    if VV[v, 2] == 'kurtosis':
                        CurrFeatVal = scipy.stats.kurtosis(CLValues, axis=0, fisher=False)
                    elif VV[v, 2] == 'skewness':
                        CurrFeatVal = scipy.stats.skew(CLValues)
                    elif VV[v, 2] == 'Slope1':
                        if VV[v, 3] == 'coefa':
                            Slp1 = np.polyfit(CLDates, CLValues, 1)
                            CurrFeatVal = Slp1[0]
                        elif VV[v, 3] == 'coefb':
                            Slp1 = np.polyfit(CLDates, CLValues, 1)
                            CurrFeatVal = Slp1[1]

                    elif VV[v, 2] == 'Slope2':
                        if VV[v, 3] == 'coefa':

                            Slp1 = np.polyfit(CLDates, CLValues, 2)
                            CurrFeatVal = Slp1[0]
                        elif VV[v, 3] == 'coefb':
                            Slp1 = np.polyfit(CLDates, CLValues, 2)
                            CurrFeatVal = Slp1[1]
                        elif VV[v, 3] == 'coefc':
                            Slp1 = np.polyfit(CLDates, CLValues, 2)
                            CurrFeatVal = Slp1[2]

                if Un_Dates_L >= 2:
                    if VV[v, 2] == 'kurtosis':
                        CurrFeatVal = scipy.stats.kurtosis(CLValues, axis=0, fisher=False)
                    elif VV[v, 2] == 'skewness':
                        CurrFeatVal = scipy.stats.skew(CLValues)

                Lab_P[0, lcnt] = CurrFeatVal
                if CurrFeatVal > 0:
                    stop = 1

    Lab_P = np.squeeze(np.copy(np.transpose(Lab_P)))
    ################################################################################

    OutputVal = np.concatenate(
        (Baseline_P, np.zeros((64,)), Lab_P, Infec_P))  # Joining Feature Vectors
    CheckFeats = np.column_stack((CT_FNM, OutputVal))
    # END OF FEATURE GENERATION #

    phase2_steps += 1
    update_state(phase2_steps, phase2_total_steps, CeleryTaskPhase.PHASE_2.value)

    logger.debug('Phase 3/5: Generating Predictions from CLL-TIMs base-learners')
    ################################################################################
    # Generating 28 Base-learner Predictions
    ################################################################################
    AllProbs = np.zeros((28, 1))
    FinalFeats = np.zeros((1, 228))
    RealFeats = np.zeros((1, 228))
    RealFeatsU = np.zeros((1, 228))
    PopMatrix = np.zeros((4149, 228))
    PopMatrixU = np.zeros((4149, 228))
    # Generating Base-learner Predictions
    for nn in range(0, CurrEnsSize):
        update_state(nn, CurrEnsSize, CeleryTaskPhase.PHASE_3.value)
        # Loading Base Learner
        Curr_BL = pickle.load(open(os.path.join(settings.ML_DATA, 'CLL-TIM_BLN_' + str(nn) + '.sav'), 'rb'))
        # Loading feature indices of base-learner
        FInds = Collect_FeatIndies.item()[Collect_CVData_Chosen[nn, 0]]
        fcnt = 0
        FeatsofBL = np.zeros((len(FInds))) - 99
        for ff in FInds:
            FeatsofBL[fcnt] = np.squeeze(np.argwhere(FAInds == ff))
            fcnt = fcnt + 1
        FeatsofBL = np.int64(FeatsofBL)
        AATrain = TrainMatrix[:, FInds]
        VMatrix = np.row_stack((OutputVal, OutputVal))

        VMatrix = VMatrix[:, FeatsofBL]
        if Collect_ModelNames_Chosen[nn, 1] != 'XGB':  # normalization if not XGBoost

            [TMatrix, get_colmean] = Nan2Mean(TrainMatrix[:, FInds])
            VMatrix = Nan2Mean_useColmean(VMatrix, get_colmean)
            scaler = preprocessing.StandardScaler()
            scaler.fit(TMatrix)
            TMatrix = scaler.transform(TMatrix)
            VMatrix = scaler.transform(VMatrix)

        AATrainNorm = TrainMatrix[:, FInds]

        mx = np.nanmax(AATrainNorm, axis=0)
        mn = np.nanmin(AATrainNorm, axis=0)
        AATrainNorm = (AATrainNorm - mn) / (mx - mn)
        VMatrixNorm = (VMatrix - mn) / (mx - mn)
        Probs = Curr_BL.predict_proba(VMatrix)

        AllProbs[nn] = Probs[0, 1]

        if np.sum(Collect_ModelNames_Chosen[nn, 1] == np.array(['Elastic', 'LogisticRegression'])) > 0:
            yoyo = 1  # TO DO SHAP VALUES FOR THESE TWO
            print('we are LR/ELASTIC')
            X_train_summary = shap.kmeans(TMatrix, 10)
            explainer = shap.KernelExplainer(Curr_BL.predict_proba, X_train_summary)
            shap_values = explainer.shap_values(VMatrix)

        elif np.sum(Collect_ModelNames_Chosen[nn, 1] == np.array(['XGB'])) > 0:
            explainer = shap.TreeExplainer(Curr_BL)
            shap_values = explainer.shap_values(VMatrix)

            cnty = 0
            for s in TransList[FInds]:
                cind = np.int64(np.squeeze(np.argwhere(ALLFeats == s)))

                if cind.size > 0:
                    FinalFeats[0, cind] = FinalFeats[0, cind] + shap_values[0, cnty]
                    RealFeats[0, cind] = VMatrixNorm[0, cnty]
                    RealFeatsU[0, cind] = VMatrix[0, cnty]
                    PopMatrixU[:, cind] = AATrain[:, cnty]
                    PopMatrix[:, cind] = AATrainNorm[:, cnty]
                    cnty = cnty + 1

    ProbRisk = np.mean(AllProbs)  # Generating Mean Probabilistic Risk from CLL-TIM
    # END OF CLL-TIM Prediction
    ################################################################################

    logger.debug('Phase 4/5: Generating Personalized Risk Factors')

    ################################################################################
    ## Generation of Personalized Risk Factors
    ################################################################################
    Toppies = ['', '', '', '', '', '', '', '', '', '']
    ToppiesContribs = np.zeros((1, 10))
    ToppiesRealVals = np.zeros((1, 10))
    ToppiesRealValsU = np.zeros((1, 10))
    TopInds = np.zeros((1, 10))
    for n in range(np.shape(FinalFeats)[0]):
        update_state(n, np.shape(FinalFeats)[0], CeleryTaskPhase.PHASE_4.value)
        Top = np.argsort(-1 * FinalFeats[n, :])
        TopVals = np.sort(-1 * FinalFeats[n, :])

        TopLR = np.array(FuncFeatNames[0][Top[0:5]])
        for rr in range(5):
            if FuncFeatNames[2][Top[rr]] == 1:
                if RealFeats[0, Top[rr]] == 0:
                    TopLR[rr] = FuncFeatNames[1][Top[rr]]
            else:

                if np.isnan(RealFeats[0, Top[rr]]):
                    TopLR[rr] = FuncFeatNames[1][Top[rr]]

        TopHR = np.array(FuncFeatNames[0][Top[-5:]])
        rrcnt = -1
        for rr in np.array([223, 224, 225, 226, 227]):
            rrcnt = rrcnt + 1
            if FuncFeatNames[2][Top[rr]] == 1:
                if RealFeats[0, Top[rr]] == 0:
                    TopHR[rrcnt] = FuncFeatNames[1][Top[rr]]
            else:

                if np.isnan(RealFeats[0, Top[rr]]):
                    TopHR[rrcnt] = FuncFeatNames[1][Top[rr]]

        Toppies = np.row_stack((Toppies, np.append(TopHR, TopLR)))
        ToppiesContribs = np.row_stack((ToppiesContribs, np.append(TopVals[-5:], TopVals[0:5])))
        ToppiesRealVals = np.row_stack((ToppiesRealVals, np.append(RealFeats[n, Top[-5:]], RealFeats[n, Top[0:5]])))
        ToppiesRealValsU = np.row_stack((ToppiesRealValsU, np.append(RealFeatsU[n, Top[-5:]], RealFeatsU[n, Top[0:5]])))
        TopInds = np.row_stack((TopInds, np.append(Top[-5:], Top[0:5])))

    ToppiesRealVals = ToppiesRealVals[1:, :]
    ToppiesRealValsU = ToppiesRealValsU[1:, :]
    ToppiesContribs = ToppiesContribs[1:, :]
    Toppies = Toppies[1:, :]
    TopInds = np.int64(TopInds[1:, :])
    ################################################################################
    PCnt = 0
    if ProbRisk > 0.5:
        print('Patient ' + str(PData_CV) + ': Predicted as High-Risk of Infection or CLL Treatment in the next 2-years')
        if ProbRisk > 0.58:

            print('Note: CLL-TIM predicts this with High-Confidence')
        else:

            print('Note: CLL-TIM predicts this with Low-Confidence')

        print('')
        print('Personalized High-Risk Factors for Patient: ' + str(PData_CV))
        for nnn in range(5, 10):
            PCnt = PCnt + 1
            if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeats[0, TopInds[0, nnn]])):
                print(str(PCnt) + ': ' + Toppies[0, nnn])
            else:

                if ToppiesRealValsU[0, nnn] <= np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]):
                    print(str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
                        ToppiesRealValsU[0, nnn]) + ' lower than the population median of ' +
                          str(np.round(np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]), 1)))
                else:
                    print(str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
                        ToppiesRealValsU[0, nnn]) + ' higher than the population median of ' +
                          str(np.round(np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]), 1)))

    else:
        print('Patient ' + str(PData_CV) + ': Predicted as Low-Risk of Infection or CLL Treatment in the next 2-years')

        if ProbRisk < 0.28:

            print('Note: CLL-TIM predicts this with High-Confidence')
        else:

            print('Note: CLL-TIM predicts this with Low-Confidence')
        print('')
        print('Personalized Low-Risk Factors for Patient: ' + str(PData_CV))
        for nnn in range(4, -1, -1):

            PCnt = PCnt + 1
            if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeats[0, TopInds[0, nnn]])):

                print(str(PCnt) + ': ' + Toppies[0, nnn])
            else:

                if ToppiesRealValsU[0, nnn] <= np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]):
                    print(str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
                        ToppiesRealValsU[0, nnn]) + ' lower than the population median of ' +
                          str(np.round(np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]), 1)))
                else:
                    print(str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
                        ToppiesRealValsU[0, nnn]) + ' higher than the population median of ' +
                          str(np.round(np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]), 1)))

    ################################################################################
    ## Visualization of Personalized Risk Factors
    ################################################################################

    logger.debug('Phase 5/5: Generating Results')

    p1 = ["silver", "olivedrab", "royalblue", "steelblue", "lightsalmon", "darkorange", 'firebrick', 'royalblue',
          'mediumspringgreen', 'aqua', 'saddlebrown', 'palevioletred', 'brown', 'olivedrab', 'black']
    p2 = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf']
    pal = np.append(p1, p2)
    import pylab as P
    mks = ["o",
           "*",
           "^",
           "p",
           "s",
           "D",
           "h",
           "X",
           "v",
           '+']
    for p in [0]:
        plt.close('all')
        #      plt.figure(figsize=(25,25),dpi=1200)
        plt.figure(dpi=3000)  #
        plt.figure()

        #      ddVals=0
        #      ddInds=0
        #
        #      IndyVec=0
        #      cntyred=0
        #      cntyblue=1
        LW = 0.1


        plt.subplot(2, 1, 1)
        # plt.xlim(-30,30)

        # plt.bar(np.arange(10),ToppiesContribs[p,:])

        Flippy = np.flip(abs(ToppiesContribs[p, 5:]))
        Flippy = ToppiesContribs[p, 5:]
        HL = np.min(abs(ToppiesContribs[p, :])) - 0.0001
        plt.barh(0, Flippy[0], color=(1, 0, 0), height=0.3, edgecolor='black')

        P.arrow(0.001 + Flippy[0], 0, -Flippy[0] - HL - 0.01, 0, ec=(1, 0, 0), fc="black", width=0, head_width=0.1,
                head_length=HL)
        bbox = {'facecolor': 'lightgray', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
        plt.text(0, -0.2, Toppies[p, 5], {'ha': 'right', 'va': 'top', 'bbox': bbox}, rotation=0, color='black',
                 fontsize=7)
        bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
        plt.text(0 - HL, 0, str(np.round(Flippy[0], 2)), {'ha': 'right', 'va': 'center', 'bbox': bbox}, rotation=0,
                 color='black', fontsize=5)
        cnty = 0.2
        for n in range(1, 5):
            update_state(n, 5, CeleryTaskPhase.PHASE_5.value)
            plt.barh(0, Flippy[n], left=np.sum(Flippy[0:n]), color=(1, cnty, cnty), height=0.3, edgecolor='black')
            P.arrow(0.001 + np.sum(Flippy[0:n]) + Flippy[n], 0, -Flippy[n] - HL - 0.01, 0, ec=(1, cnty, cnty),
                    fc="black", width=0, head_width=0.1, head_length=HL)

            P.arrow(np.sum(Flippy[0:n]), -0.2, 0, (-0.2 * n), ec='black', fc="black", width=0.05, head_width=0.1,
                    head_length=0)
            bbox = {'facecolor': 'lightgray', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
            plt.text(np.sum(Flippy[0:n]), (-0.2 * n) - 0.2, Toppies[p, n + 5],
                     {'ha': 'right', 'va': 'top', 'bbox': bbox}, rotation=0, color='black', fontsize=7)

            bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
            plt.text(np.sum(Flippy[0:n]) - HL, 0, str(np.round(Flippy[n], 2)),
                     {'ha': 'right', 'va': 'center', 'bbox': bbox}, rotation=0, color='black', fontsize=5)
            cnty = cnty + 0.2

        P.arrow(np.sum(Flippy), 0.37, -np.sum(Flippy) - 0.3, 0, ec='black', fc=(1, 0, 0), width=0.04, head_width=0.1,
                head_length=0.3, linewidth=0.1)
        plt.text(np.sum(Flippy) / 2, 0.5, 'Top High-Risk Contributors', {'ha': 'center', 'va': 'bottom'}, fontsize=5)

        # plt.ylim(-1,1)

        Flippy = np.flip(abs(ToppiesContribs[p, 0:5]))
        plt.barh(0, Flippy[0], color=(0, 0, 1), height=0.3, edgecolor='black')
        HL = np.min(abs(ToppiesContribs[p, :])) - 0.0001
        P.arrow(0.001 + Flippy[0], 0, -Flippy[0] + HL - 0.01, 0, fc='black', ec=(0, 0, 1), head_width=0.1, width=0,
                head_length=HL, shape='full')
        # P.arrow( Flippy[0],0.05,-Flippy[0]+HL, 0,fc=(0,0,1), ec='k',head_width=0.1, head_length=HL,shape='left')
        bbox = {'facecolor': 'lightgray', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
        plt.text(0, +0.2, Toppies[p, n], {'ha': 'left', 'va': 'bottom', 'bbox': bbox}, rotation=0, color='black',
                 fontsize=7)
        bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
        plt.text(0 + HL, 0, str(np.round(Flippy[0], 2)), {'ha': 'left', 'va': 'center', 'bbox': bbox}, rotation=0,
                 color='black', fontsize=5)
        cnty = 0.2
        for n in range(1, 5):
            update_state(n, 5, CeleryTaskPhase.PHASE_5.value)
            #      HL=np.min(abs(Flippy[n]))-0.001
            plt.barh(0, Flippy[n], left=np.sum(Flippy[0:n]), color=(cnty, cnty, 1), height=0.3, edgecolor='black')
            P.arrow(0.001 + Flippy[n] + np.sum(Flippy[0:n]), 0, -Flippy[n] + HL - 0.01, 0, fc='black',
                    ec=(cnty, cnty, 1), width=0, head_width=0.1, head_length=HL, shape='full')
            #      P.arrow( Flippy[n]+np.sum(Flippy[0:n]),0.05,-Flippy[n]+HL, 0,fc=(cnty,cnty,1), ec="k",head_width=0.1, head_length=HL,shape='left')
            P.arrow(np.sum(Flippy[0:n]), 0.2, 0, (0.2 * n), ec='black', fc="black", width=0.05, head_width=0.1,
                    head_length=0)
            bbox = {'facecolor': 'lightgray', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
            plt.text(np.sum(Flippy[0:n]), (0.2 * n) + 0.2, Toppies[p, 4 - n],
                     {'ha': 'left', 'va': 'bottom', 'bbox': bbox}, rotation=0, color='black', fontsize=7)
            bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
            plt.text(np.sum(Flippy[0:n]) + HL, 0, str(np.round(Flippy[n], 2)),
                     {'ha': 'left', 'va': 'center', 'bbox': bbox}, rotation=0, color='black', fontsize=5)
            cnty = cnty + 0.2

        P.arrow(np.sum(Flippy), -0.37, -np.sum(Flippy) + 0.3, 0, ec='black', fc=(0, 0, 1), width=0.04, head_width=0.1,
                head_length=0.3, linewidth=0.1)
        plt.text(np.sum(Flippy) / 2, -0.5, 'Top Low-Risk Contributors', {'ha': 'center', 'va': 'top'}, fontsize=5)

        plt.ylim(-1, 1)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

   

    Xinit = -70
    if ProbRisk > 0.5:
        plt.text(Xinit, 1.5, 'Patient ' + str(
            PData_CV) + ': Predicted as High-Risk of Infection or CLL Treatment in the next 2 years', color='red')

        if ProbRisk > 0.58:
            nin = 0
            plt.text(Xinit, 1.2, 'Confidence Level is ' + str(np.round(ProbRisk, 2)) + ': High-Confidence')
        else:
            nin = 0
            plt.text(Xinit, 1.2, 'Confidence Level is ' + str(np.round(ProbRisk, 2)) + ': Low-Confidence')
    else:
        plt.text(Xinit, 1.5, 'Patient ' + str(
            PData_CV) + ': Predicted as Low-Risk of Infection or CLL Treatment in the next 2 years', color='blue')

        if ProbRisk < 0.28:
            nin = 0
            plt.text(Xinit, 1.2, 'Confidence Level is ' + str(np.round(ProbRisk, 2)) + ': High-Confidence')
        else:
            nin = 0
            plt.text(Xinit, 1.2, 'Confidence Level is ' + str(np.round(ProbRisk, 2)) + ': Low-Confidence')

    Yinit = 0.4
    hh = 0.2
    SF = 50

    plt.barh(Yinit, SF * 0.28, left=Xinit + SF * 0, color='blue', height=hh, edgecolor='black')
    plt.barh(Yinit, SF * 0.22, left=Xinit + SF * 0.28, color='lightblue', height=hh, edgecolor='black')
    plt.barh(Yinit, SF * 0.08, left=Xinit + SF * 0.5, color='pink', height=hh, edgecolor='black')
    plt.barh(Yinit, SF * 0.42, left=Xinit + SF * 0.58, color='red', height=hh, edgecolor='black')
    if ProbRisk > 0.5:
        P.arrow(Xinit + SF * ProbRisk, Yinit + -(hh / 2), 0, hh + 0.3, ec='red', fc="red", width=0.01, head_width=0.5,
                head_length=0.05)
    else:
        P.arrow(Xinit + SF * ProbRisk, Yinit + -(hh / 2), 0, hh + 0.3, ec='blue', fc="blue", width=0.01, head_width=0.5,
                head_length=0.05)
    plt.text(Xinit + SF * 0.14, Yinit + -(hh / 2 + 0.05), 'High-Confidence', {'ha': 'center', 'va': 'top'}, fontsize=5,
             color='black')
    plt.text(Xinit + SF * 0.45, Yinit + -(hh / 2 + 0.05), 'Low-Confidence ', {'ha': 'center', 'va': 'top'}, fontsize=5,
             color='black')
    plt.text(Xinit + SF * 0.80, Yinit + -(hh / 2 + 0.05), 'High-Confidence ', {'ha': 'center', 'va': 'top'}, fontsize=5,
             color='black')
    bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}

    plt.text(Xinit + SF * 1, Yinit + (hh / 2 + 0.25), '1', {'ha': 'center', 'va': 'top'}, fontsize=8, color='black')
    plt.text(Xinit + SF * 0.5, Yinit + (hh / 2 + 0.25), '0.5', {'ha': 'center', 'va': 'top'}, fontsize=8, color='black')
    plt.text(Xinit + SF * 0, Yinit + (hh / 2 + 0.25), '0', {'ha': 'center', 'va': 'top'}, fontsize=8, color='black')

    #
    P.arrow(Xinit + SF * 0.5, Yinit + (hh / 2 + 0.07), SF * (0.5 - 0.05), 0, ec='red', fc="red", width=0.005,
            head_width=0.1, head_length=SF * 0.05)
    P.arrow(Xinit + SF * 0.5, Yinit + (hh / 2 + 0.07), SF * (-0.5 + 0.05), 0, ec='blue', fc="blue", width=0.005,
            head_width=0.1, head_length=SF * 0.05)
    P.arrow(Xinit + SF * 0.5, Yinit + (hh / 2 + 0.07), SF * 0, 0.05, ec='black', fc="black", width=0.005, head_width=0,
            head_length=0)
    plt.text(Xinit + SF * ProbRisk, Yinit + 0.6, str(np.round(ProbRisk, 2)),
             {'ha': 'center', 'va': 'top', 'bbox': bbox}, fontsize=7, color='black')

    bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
    plt.text(Xinit + SF * 0.9, Yinit + (hh / 2 + 0.2), 'High-Risk', {'ha': 'right', 'va': 'top', 'bbox': bbox},
             fontsize=5, color='Red')
    plt.text(Xinit + SF * 0.1, Yinit + (hh / 2 + 0.2), 'Low-Risk', {'ha': 'left', 'va': 'top', 'bbox': bbox},
             fontsize=5, color='Blue')
    #

    #

    plt.subplot(2, 1, 2)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    PCnt = 0
    if ProbRisk > 0.5:

        plt.text(0, 1, 'Personalized High-Risk Factors for Patient: ' + str(PData_CV), weight="bold")
        CurrXPos = 1
        for nnn in range(5, 10):
            update_state(nnn, 10, CeleryTaskPhase.PHASE_5.value)
            CurrXPos = CurrXPos - 0.2
            PCnt = PCnt + 1
            if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeats[0, TopInds[0, nnn]])):
                plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn])
            elif np.logical_and(FuncFeatNames[3][TopInds[0, nnn]] == 'C', RealFeats[0, TopInds[0, nnn]] == 0):
                plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn])
            else:

                if ToppiesRealValsU[0, nnn] <= np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]):
                    plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
                        np.round(ToppiesRealValsU[0, nnn], 1)) + ' lower than the population median of ' +
                             str(np.round(np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]), 1)))
                else:
                    plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
                        np.round(ToppiesRealValsU[0, nnn], 1)) + ' higher than the population median of ' +
                             str(np.round(np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]), 1)))

    else:

        plt.text(0, 1, 'Personalized Low-Risk Factors for Patient: ' + str(PData_CV), weight="bold")
        CurrXPos = 1
        for nnn in range(4, -1, -1):
            CurrXPos = CurrXPos - 0.2
            PCnt = PCnt + 1
            if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeats[0, TopInds[0, nnn]])):

                plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn])
            elif np.logical_and(FuncFeatNames[3][TopInds[0, nnn]] == 'C', RealFeats[0, TopInds[0, nnn]] == 0):
                plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn])
            else:

                if ToppiesRealValsU[0, nnn] <= np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]):
                    plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
                        np.round(ToppiesRealValsU[0, nnn], 1)) + ' lower than the population median of ' +
                             str(np.round(np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]), 1)))
                else:
                    plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
                        np.round(ToppiesRealValsU[0, nnn], 1)) + ' higher than the population median of ' +
                             str(np.round(np.nanmedian(PopMatrixU[:, TopInds[0, nnn]]), 1)))

    pdf_path = os.path.join(settings.MEDIA_ROOT, 'predictions', str(prediction.id), '{0}.pdf'.format(str(prediction.id)))
    png_path = os.path.join(settings.MEDIA_ROOT, 'predictions', str(prediction.id), '{0}.png'.format(str(prediction.id)))
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close('all')

    logger.debug('Phase 5/5: Generating Results are completed')

    prediction.refresh_from_db()
    prediction.status = PredictionStatus.DONE.name
    prediction.save()

    end = timezone.now()
    delta = relativedelta(end, start)
    logger.debug("ML Process took : {:02}:{:02}:{:02} to finish".format(delta.hours, delta.minutes, delta.seconds))
