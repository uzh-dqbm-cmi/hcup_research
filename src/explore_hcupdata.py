'''
@author: ahmed allam
'''
import os
from collections import namedtuple
import pandas as pd
import numpy as np
from utilities import ReaderWriter, create_directory

def generate_elms_info(elm_desc):
    elm_info = {'colnames':[], 'coldesc':[], 'coltype':[]}
    for elm_tuple in elm_desc:
        prefix, num_cols, desc, coltype = elm_tuple
        if(num_cols):
            for i in range(1, num_cols+1):
                elm_info['colnames'].append(prefix+str(i))
                elm_info['coldesc'].append(desc + " " + str(i))
                elm_info['coltype'].append(coltype)
        else:
            elm_info['colnames'].append(prefix)
            elm_info['coldesc'].append(desc)
            elm_info['coltype'].append(coltype)
    return(elm_info)

# NRD Core database
NRDCore_namedesc = [("age", 0, "Age in years at admission","int32"),
                    ("aweekend", 0, "Admission day is a weekend","int32"),
                    ("died", 0, "Died during hospitalization","float64"),
                    ("discwt", 0, "Weight to discharges in AHA universe","float64"),
                    ("dispuniform", 0, "Disposition of patient (uniform)","int32"),
                    ("dmonth", 0, "Discharge month","int32"),
                    ("dqtr", 0, "Discharge quarter","int32"),
                    ("drg", 0, "DRG in effect on discharge date","int32"),
                    ("drgver", 0, "DRG grouper version used on discharge date","int32"),
                    ("drg_nopoa", 0, "DRG in use on discharge date, calculated without POA","int32"),
                    ("dx", 25, "Diagnosis","object"),
                    ("dxccs", 25, "CCS: diagnosis","int32"),
                    ("ecode", 4, "E code","object"),
                    ("elective", 0, "Elective versus non-elective admission","int32"),
                    ("e_ccs", 4, "CCS: E Code","int32"),
                    ("female", 0, "Indicator of sex","int32"),
                    ("hcup_ed", 0,  "HCUP Emergency Department service indicator","int32"),
                    ("hosp_nrd", 0, "NRD hospital identifier","int32"),
                    ("key_nrd", 0, "NRD record identifier","int32"),
                    ("los", 0, "Length of stay (cleaned)","int32"),
                    ("mdc", 0, "MDC in effect on discharge date","int32"),
                    ("mdc_nopoa", 0, "MDC in use on discharge date, calculated without POA","int32"),
                    ("nchronic", 0, "Number of chronic conditions","int32"),
                    ("ndx", 0, "Number of diagnoses on this record","int32"),
                    ("necode", 0, "Number of E codes on this record","int32"),
                    ("npr", 0, "Number of procedures on this record","int32"),
                    ("nrd_daystoevent", 0, "Timing variable used to identify days between admissions","int32"),
                    ("nrd_stratum", 0, "NRD stratum used for weighting","int32"),
                    ("nrd_visitlink", 0, "NRD visitlink","object"),
                    ("orproc", 0, "Major operating room procedure indicator","int32"),
                    ("pay1", 0, "Primary expected payer (uniform)","int32"),
                    ("pl_nchs", 0, "Patient Location: NCHS Urban-Rural Code","int32"),
                    ("pr", 15, "Procedure","object"),
                    ("prccs", 15, "CCS: procedure","int32"),
                    ("rehabtransfer", 0, "A combined record involving rehab transfer","int32"),
                    ("resident", 0, "Patient State is the same as Hospital State","int32"),
                    ("samedayevent", 0, "Transfer flag indicating combination of discharges involve same day events","int32"),
                    ("totchg", 0, "Total charges (cleaned)","int32"),
                    ("year", 0, "Calendar year","int32"),
                    ("zipinc_qrtl", 0, "Median household income national quartile for patient ZIP Code","int32")]
# NRD Severity database
NRDSeverity_namedesc = [("aprdrg", 0, "All Patient Refined DRG", ""),
                        ("aprdrg_risk_mortality", 0, "All Patient Refined DRG: Risk of Mortality Subclass",""),
                        ("aprdrg_severity", 0, "All Patient Refined DRG: Severity of Illness Subclass",""),
                        ("cm_aids", 0, "AHRQ comorbidity measure: Acquired immune deficiency syndrome",""),
                        ("cm_alcohol", 0, "AHRQ comorbidity measure: Alcohol abuse",""),
                        ("cm_anemdef", 0, "AHRQ comorbidity measure: Deficiency anemias",""),
                        ("cm_arth", 0, "AHRQ comorbidity measure: Rheumatoid arthritis/collagen vascular diseases",""),
                        ("cm_bldloss", 0, "AHRQ comorbidity measure: Chronic blood loss anemia",""),
                        ("cm_chf", 0, "AHRQ comorbidity measure: Congestive heart failure",""), 
                        ("cm_chrnlung", 0, "AHRQ comorbidity measure: Chronic pulmonary disease",""),
                        ("cm_coag", 0, "AHRQ comorbidity measure: Coagulopathy",""),
                        ("cm_depress", 0, "AHRQ comorbidity measure: Depression",""),
                        ("cm_dm", 0, "AHRQ comorbidity measure: Diabetes, uncomplicated",""),
                        ("cm_dmcx", 0, "AHRQ comorbidity measure: Diabetes with chronic complications",""),
                        ("cm_drug", 0, "AHRQ comorbidity measure: Drug abuse",""),
                        ("cm_htn_c", 0, "AHRQ comorbidity measure: Hypertension (combine uncomplicated and complicated)",""), 
                        ("cm_hypothy", 0, "AHRQ comorbidity measure: Hypothyroidism",""),
                        ("cm_liver", 0, "AHRQ comorbidity measure: Liver disease",""),
                        ("cm_lymph", 0, "AHRQ comorbidity measure: Lymphoma",""), 
                        ("cm_lytes", 0, "AHRQ comorbidity measure: Fluid and electrolyte disorders",""),
                        ("cm_mets", 0, "AHRQ comorbidity measure: Metastatic cancer",""),
                        ("cm_neuro", 0, "AHRQ comorbidity measure: Other neurological disorders",""),
                        ("cm_obese", 0, "AHRQ comorbidity measure: Obesity",""),
                        ("cm_para", 0, "AHRQ comorbidity measure: Paralysis",""),
                        ("cm_perivasc", 0, "AHRQ comorbidity measure: Peripheral vascular disorders",""),
                        ("cm_psych", 0, "AHRQ comorbidity measure: Psychoses",""),
                        ("cm_pulmcirc", 0, "AHRQ comorbidity measure: Pulmonary circulation disorders",""),
                        ("cm_renlfail", 0, "AHRQ comorbidity measure: Renal failure",""),
                        ("cm_tumor", 0, "AHRQ comorbidity measure: Solid tumor without metastasis",""),
                        ("cm_ulcer", 0, "AHRQ comorbidity measure: Peptic ulcer disease excluding bleeding",""),
                        ("cm_valve", 0, "AHRQ comorbidity measure: Valvular disease",""),
                        ("cm_wghtloss", 0, "AHRQ comorbidity measure: Weight loss",""),
                        ("hosp_nrd", 0, "NRD hospital identifier",""), 
                        ("key_nrd", 0, "NRD record identifier","")]

# NRD Dxpr database                    
NRDDxpr_namedesc= [('chron', 25, 'Chronic condition indicator', 'int32'),
                    ('chronb',25, 'Chronic condition body system', 'int32'),
                    ('dxmccs1', 0, 'Multi-Level CCS:  Diagnosis 1', 'str'),
                    ('e_mccs1', 0, 'Multi-Level CCS:  E Code 1', 'str'),
                    ('hosp_nrd', 0, 'NRD hospital identifier', 'int32'),
                    ('key_nrd', 0, 'NRD record identifier', 'str'),
                    ('pclass', 15, 'Procedure classs', 'int32'),
                    ('prmccs1', 0, 'Multi-Level CCS:  Procedure 1', 'str')
                ]
"""
datatype for NRD Core data is identified using the `description file of NRD 2013 database <https://www.hcup-us.ahrq.gov/db/nation/nrd/stats/FileSpecifications_NRD_2013_Core.TXT>`__

Reading the NRD Core data into a dataframe will capture the datatypes correctly except for couple of data elements. Hence, below we supply the expected datatypes following the 
``description file of NRD 2013``

Usually, the transformation/mapping of python datatypes to the pandas dataframe datatypes is:
::
    
    Python datatype        Pandas datatype
    string                 object
    int                    int64
    float                  float64
"""
# use the below script to identify the datatypes coarced by pandas.read_csv()
# with open('nrd_datatypes.txt', 'w') as f:
#     for col in nrd.columns:
#         f.write('"'+col+'"' + ":" + '"'+str(nrd[col].dtypes)+'"'+ ",\n")



# load the data into a dataframe 
def load_nrd(dataset_type, dataset_dir):
    if(dataset_type == 'Core'):
        # "NRD_2013_Core.CSV"
        nrdcore_info = generate_elms_info(NRDCore_namedesc)
        fpath = os.path.join(dataset_dir, "NRD_2013_Core.CSV")
        nrd = pd.read_csv(fpath, dtype= dict(zip(nrdcore_info['colnames'], nrdcore_info['coltype'])), names = nrdcore_info['colnames'])
    elif(dataset_type == 'Severity'):
        # "NRD_2013_Severity.CSV"
        nrdseverity_info = generate_elms_info(NRDSeverity_namedesc)
        fpath = os.path.join(dataset_dir, "NRD_2013_Severity.CSV")
        nrd = pd.read_csv(fpath, names = nrdseverity_info['colnames'])
    elif(dataset_type == 'Dxpr'):
        # preparing elements of nrd_dxpr database
        nrd_dxpr_info = generate_elms_info(NRDDxpr_namedesc)
        # load DX_PR_Grps database
        fpath = os.path.join(dataset_dir, "NRD_2013_DX_PR_GRPS.CSV")
        nrd = pd.read_csv(fpath, dtype= dict(zip(nrd_dxpr_info['colnames'], nrd_dxpr_info['coltype'])), names = nrd_dxpr_info['colnames'])
    return(nrd)

"""
missing and invalid codes:

    - For numeric data elements, it is always starting with minus sign (i.e. -9x, -8x, -7x, -6x, etc. where x means the decimal is repeated).
      To differentiate between the types of the missing or invalid codes check `data coding pdf from HCUP <https://www.hcup-us.ahrq.gov/db/coding.pdf>`__
      -9x : missing
      -8x : invalid
      -7x : data unavailable from source
      -6x : inconsistent data
      -5x : not applicable data
      
    - For char data elements, it is blank 
    
    .. note ::
    
       numeric data element represent also categorical elements. The definition of numeric in this context means elements coded with numbers which might refer to also categories.
       Check ``Attributes of Data Elements`` section of the `coding pdf on HCUP <https://www.hcup-us.ahrq.gov/db/coding.pdf>`__
   
   After checking and exploring data, here is the conclusion:
   
       1- numeric data would have negative numbers as stated above referring to different types of missing (i.e. missing, invalid, inconsistent, etc.)
       2- string/char data have blank, invl, incn as values for missing, invalid or inconsistent respectively
"""

def report_missing(df, out_fpath):
    """check missing or invalid codes in the `nrd dataframe`
    
       Args:
           df: pandas dataframe
           out_fpath: output file path to write the results to 
    """
    f_out = open(out_fpath, 'a')
    # check missing values in every column
    for col in df.columns:
        st = ""
        # check negative values and describe using value_counts function
        st += 'column name: {}, column dtype: {} \n'.format(col, df[col].dtype)
        st += 'incorrect, invalid, inconsistent values \n'
        # check if the datatype of the column is numeric
        if(df[col].dtype != 'object'):
            cond_isneg = df[col]<0
            st += "{} \n".format(df.loc[cond_isneg, col].value_counts())
            st += "-"*20 + " \n"
        # if the datatype of the column is object/string
        else:
            for code in ('invl', 'incn'):
                st += "condition: {} \n".format(code)
                cond = df[col] == code
                st += "{} \n".format(df.loc[cond, col].value_counts())
            st += "-"*20 + " \n"
        # check blank/NaN values
        st += "Blank/NaN count values \n"
        st += "{} \n".format(df[col].isnull().sum())
        st += "*"*40 + "\n"
        print(st)
        f_out.write(st)
    f_out.close()


def fill_missing(df):
    """fill missing or invalid codes in the dataframe with a default code NaN
    
       Args:
           df: pandas dataframe
           
    """
    missing_codes = ('invl', 'incn')
    # fill missing values with NaN
    for col in df.columns:
        print("fixing col_name: {}, dtype: {}".format(col, df[col].dtype))
        # check if the datatype of the column is numeric
        if(df[col].dtype != 'object' and df[col].dtype != 'str'):
            cond_isneg = df[col]<0
            count_viol = cond_isneg.sum()
            if(count_viol):
                print("found {} violations".format(count_viol))
                # assign NaN values
                df.loc[cond_isneg, col] = np.nan
        # if the datatype of the column is object/string
        else:
            cond = df[col].isin(missing_codes)
            count_viol = cond.sum()
            if(cond.sum()):
                print("found {} violations".format(count_viol))
                df.loc[cond, col] = np.nan  

def convert_dtypes(df): # to integrate this function with :func:`fill_missing` to save resources
    """convert the dtypes of columns in a data frame in place"""
    for col in df.columns:
        coltype = df[col].dtype
        print('colname: {}, coltype: {}'.format(col, coltype))
        if(coltype == 'int64' or coltype == 'float64'):   
            flag = df[col].isnull().values.any()
            if(flag): # default when a columns has NaN values
                coltype = 'float32'
            else:
                if(coltype == 'int64'):
                    coltype = 'int32'
                else:
                    coltype = 'float32'
            df[col] = df[col].astype(coltype)
            print('colname: {}, coltype: {}'.format(col, coltype))
            print('-'*5)

def check_percent_missing(df, cols):
    for col in cols:
        num_null = df[col].isnull().sum()
        print("the % of missing values in {} is {}".format(col, 100*num_null/df.shape[0]))

#####################################
"""
replicating the analysis of page <ref>
"""
def readmission_example_indexevent(nrd):
    # create a pseudo discharge date
    nrd['pseudoddate'] = nrd['nrd_daystoevent'] + nrd['los']
    # create the agegroup variable 
    # age groups are 1: 1-17; 2: 18-64; 3: 65-120
    nrd['agegroup'] = pd.cut(nrd['age'], [1, 17, 64, 120], labels = [1,2,3], include_lowest=True) 
    heartattack_dx1code = [str(i) for i in range(41000,41092) if(i%10!=2)]
    cond_heartattack = nrd['dx1'].isin(heartattack_dx1code)
    cond_age = nrd['age'] >= 18
    cond_died = nrd['died'] == 0
    cond_dmonth = (nrd['dmonth'] >=1) & (nrd['dmonth'] <= 11)
    # this cond will check simultaneously the nrd_daystoevent and los
    cond_pseudoddate_valid = pd.notnull(nrd['pseudoddate'])
    # create indexevent
    nrd['indexevent'] = 0
    # assign 1 to all rows satisfying the above conditions
    overall_cond = (cond_heartattack) & (cond_age) & (cond_died) & (cond_dmonth) & (cond_pseudoddate_valid)
    nrd.loc[overall_cond, 'indexevent'] = 1
    
    print("Index event stats:")
    print(nrd['indexevent'].describe())
    print("total sum: {}".format(nrd['indexevent'].sum()))
    print("-"*20)
    print("Pseudo discharge date stats:")
    print(nrd['pseudoddate'].describe())
    print("total sum: {}".format(nrd['pseudoddate'].sum()))
    print("-"*20)
    
def generate_dignosis_condition(df, dxs, dx_val=108):
    """get the rows of the dataframe where dxccs columns are equal to the specified dx_val
    
       Args:
           df: dataframe
           dxs: tuple/list of diagnosis order
                   -(1,) for considering primary diagnosis
                   -(1,2) for considering primary and secondary diagnosis
                   -(1,2,3) for considering first, second and third diagnosis
                   -etc..
           dx_val: the target dxccs code 
    """
    for i, dx_order in enumerate(dxs):
        curr_cond = df['dxccs{}'.format(dx_order)] == dx_val
        if(i == 0):
            dx_cond = curr_cond
        else: 
            dx_cond = dx_cond | curr_cond
    return(dx_cond)


    
def get_ordered_cols(df, y_cols):
    all_cols = df.columns.tolist()
    ordered_cols = sorted(set(all_cols) - set(y_cols)) + y_cols
    return(ordered_cols)

def inner_join(df_l, df_r):
    # remember to make sure the data types of the keys must be the same to have successful join
    df_merge = pd.merge(left=df_l, right=df_r, how='inner', on=['key_nrd', 'hosp_nrd'])
    return(df_merge)

RANDOM_STATE = 123
# Note to self: both functions :func:`get_datasubset` and :func:`get_datafolds` can be merged in one
def get_datasubset(datasplit, traj_info, frac):
    # get a small stratified sample from one of the training folds
    # to do hyper parameter optimization and feature selection options
    np.random.seed(RANDOM_STATE) # for reproducibility
    fold_id = np.random.randint(0,5)
    print("chosen fold: ", fold_id)
    train_set = traj_info.loc[traj_info['nrd_visitlink'].isin(datasplit[fold_id][0])].copy()
    print("total number of samples in this training fold ", len(train_set))
    # get a stratified small fraction from the chosen training fold 
    # group_keys=False does not add index for the grouped variable -- to reuse 
    # we take 30% fraction and set random_state for reproducibility
    train_subset = train_set.groupby('allcause_readmit', group_keys=False).apply(lambda x: x.sample(frac=frac, 
                                                                                                    random_state=RANDOM_STATE))
    print("total number of samples in the subset ", train_subset.shape[0])
    print("average sequence length: ", train_subset['seq_len'].mean())
    print("average readmission rate: ", train_subset['allcause_readmit_rate'].mean())
    print("average readmission rate for last event only: ", train_subset['allcause_readmit'].mean())
    print()
    # get a stratified subsample of the above subset that will be the validation set
    # in other words, split the subset into training and validation 80-20%
    val_set = train_subset.groupby('allcause_readmit', group_keys=False).apply(lambda x: x.sample(frac=0.2, 
                                                                                                  random_state=RANDOM_STATE))
    print("total number of samples in validation: ", len(val_set))
    print("average sequence length: ", val_set['seq_len'].mean())
    print("average readmission rate: ", val_set['allcause_readmit_rate'].mean())
    val_idx =set(val_set['nrd_visitlink'].unique())
    train_idx = set(train_subset['nrd_visitlink'].unique()) - val_idx
    print("Final train size: ", len(train_idx))
    print("Final validation size: ", len(val_idx))
    return(train_idx, val_idx)

def get_datafolds(datasplit, traj_info, frac):
    datafolds = {}
    for fold_id in range(0, 5):
        print("chosen fold: ", fold_id)
        train_set = traj_info.loc[traj_info['nrd_visitlink'].isin(datasplit[fold_id][0])].copy()
        test_set = traj_info.loc[traj_info['nrd_visitlink'].isin(datasplit[fold_id][1])].copy()
        print("total number of training samples in this fold ", len(train_set))
        print("total number of testing samples in this fold ", len(test_set))
        # group_keys=False does not add index for the grouped variable -- to reuse
        # we take frac representing fraction% of the training set and set random_state for reproducibility
        val_set = train_set.groupby('allcause_readmit', group_keys=False).apply(lambda x: x.sample(frac=frac, 
                                                                                                   random_state=RANDOM_STATE))
        print("total number of samples in the validation subset ", val_set.shape[0])
        print("average sequence length: ", val_set['seq_len'].mean())
        print("average readmission rate: ", val_set['allcause_readmit_rate'].mean())
        print("average readmission rate for last event only: ", val_set['allcause_readmit'].mean())

        val_idx =set(val_set['nrd_visitlink'].unique())
        train_idx = set(train_set['nrd_visitlink'].unique()) - val_idx
        test_idx = set(test_set['nrd_visitlink'].unique())
        print("Final train size: ", len(train_idx))
        print("Final validation size: ", len(val_idx))
        print("Final test size: ", len(test_idx))
        datafolds['fold_{}'.format(fold_id)] = (train_idx, val_idx, test_idx)
    return(datafolds)

GaussianNormalizerInfo = namedtuple('GaussianNormalizerInfo', ['mean_cols', 'std_cols'])
MeanRangeNormalizerInfo = namedtuple('MeanRangeNormalizerInfo', ['mean_cols', 'range_cols'])
RescaleNormalizerInfo = namedtuple('RescaleNormalizerInfo', ['min_cols', 'range_cols'])

def get_feature_normalizer(dset, norm_features, norm_option):
    if(norm_option == 'standardize'):
        fmean = dset[norm_features].mean()
        print("mean of features:")
        print(dset[norm_features].mean())
        print()
        fstd = dset[norm_features].std()
        print("std of features:")
        print(fstd)
        print("-"*30)
        return(fmean, fstd)
    elif(norm_option=='meanrange'):
        fmean = dset[norm_features].mean()
        print("mean of features:")
        print(dset[norm_features].mean())
        print()
        fmax = dset[norm_features].max()
        fmin = dset[norm_features].min()
        frange = fmax-fmin
        print("range of features:")
        print(frange)
        print("-"*30)  
        return(fmean, frange)
    else:
        fmax = dset[norm_features].max()
        fmin = dset[norm_features].min()
        frange = fmax-fmin
        print("min of features:")
        print(fmin)
        print()
        print("range of features:")
        print(frange)
        print("-"*30)  
        return(fmin, frange)
    
def generate_normalizers(datafolds, fsample, dataset_dir, cont_cols, normalize_options = ('standardize', 'rescale')):
    for fold_name in datafolds:
        print("fold ", fold_name)
        train_idx = datafolds[fold_name][0]
        train_sample = fsample.loc[fsample['nrd_visitlink'].isin(train_idx)].copy()
        dsets = (train_sample, )
        for norm_option in normalize_options:
            print("norm_option: ", norm_option)
            dirname = "{}_{}".format(fold_name, norm_option)
            cdir = create_directory(dirname, dataset_dir)
            if(norm_option=='standardize'):
                normalizer = GaussianNormalizerInfo
            elif(norm_option=='meanrange'):
                normalizer = MeanRangeNormalizerInfo
            elif(norm_option == 'rescale'):
                normalizer = RescaleNormalizerInfo
            for dset in dsets:
                a, b = get_feature_normalizer(dset, cont_cols, norm_option)
                ReaderWriter.dump_data(normalizer(a,b), os.path.join(cdir, ("{}_info.pkl".format(norm_option))))
                
def apply_normalization(dset, norm_features, normalizer):
    """applies in-place normalization for norm_features columns in dataset (dset dataframe)
    """
    print('applying normalization')
    print(normalizer.__class__)
    a, b = normalizer
    flag = (b==0).sum()
    if(flag):
        norm_features = b[b!=0].index.tolist()
    dset[norm_features] = (dset[norm_features] - a[norm_features])/b[norm_features]                
    norm_option = str(normalizer.__class__).split('.')[-1]
    if(norm_option.startswith('Gauss')):
        print("updated mean:")
        print(dset[norm_features].mean())
        print("updated std:")
        print(dset[norm_features].std())
    elif(norm_option.startswith('Rescale')):
        print("updated max:")
        print(dset[norm_features].max())
        print("updated min:")
        print(dset[norm_features].min())
    print()
