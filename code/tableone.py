import pandas as pd
import numpy as np
from tableone import TableOne


import os
print(os.getcwd())
path = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1"
icu_path = "/oscar/data/shared/ursa/mimic-iv/icu/3.1"
#print(os.listdir(path))
admissions_path = path + "/admissions.csv"
diagnoses_icd = path + "/diagnoses_icd.csv"
d_icd = path + "/d_icd_diagnoses.csv"
patients = path + "/patients.csv"
omr = path + "/omr.csv"
emar = path + "/emar.csv"
admissions = icu_path + "/chartevents.csv"


#print(d_icd)
df = pd.read_csv(d_icd)
#print(df.columns.tolist())
filtered = df[df["icd_code"].str.contains("F03")]
filtered = np.asarray(filtered)
#print(filtered)

#Getting subject ids
df = pd.read_csv(diagnoses_icd)
filtered = df[df["icd_code"].str.contains("F03")]
print(filtered)
subject_ids = np.asarray(filtered)[:,0]
print("subject ids shape is", subject_ids.shape)


#Filter demographics by subject id 
unique_subjects = np.unique(subject_ids)


#Get patients
print(patients)
df = pd.read_csv(patients)
print(df.columns.tolist())
patients_with_dementia = df[df["subject_id"].isin(unique_subjects)]
patients_with_dementia = np.array(patients_with_dementia)

print(os.getcwd())
path = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1"
icu_path = "/oscar/data/shared/ursa/mimic-iv/icu/3.1"
#print(os.listdir(path))
admissions_path = path + "/admissions.csv"


print("Reading the admissions...")
df = pd.read_csv(admissions_path, usecols=["subject_id", "race"],
                    dtype={"subject_id": "int32"})
dementia_patients = np.asarray(df[df["subject_id"].isin(unique_subjects)])

print(dementia_patients)
print(dementia_patients.shape)
subject_ids = dementia_patients[:,0]
unique_subjects = np.unique(subject_ids)
print("There are", len(unique_subjects), "unique subjects")


print("Total unique patients:",len(patients_with_dementia))
total = int(len(patients_with_dementia))
#Female vs. male patients
female = np.where(patients_with_dementia[:,1] == "F", 1, 0)
f_num = np.sum(female)
print("Female:", f_num)
print(f_num/total)
male = np.where(patients_with_dementia[:,1] == "M", 1, 0)
m_num = np.sum(male)
print("Male:", m_num)
print(m_num/total)

#Ages 0-18, 18-30, 30-50, 50-70, 70+
tier_1 = np.where(patients_with_dementia[:,2] <= 18, 1,0)
tier_1_sum = np.sum(tier_1)
print("Less than 18:", tier_1_sum,";", tier_1_sum/total)
tier_2 = np.where((patients_with_dementia[:,2] > 18) & (patients_with_dementia[:,2] < 30), 1,0)
tier_2_sum = np.sum(tier_2)
print("18-30yo:", tier_2_sum,";", tier_2_sum/total)
tier_3 = np.where((patients_with_dementia[:,2] > 30) & (patients_with_dementia[:,2] < 50), 1,0)
tier_3_sum = np.sum(tier_3)
print("30-50yo:", tier_3_sum,";", tier_3_sum/total)
tier_4 = np.where((patients_with_dementia[:,2] > 50) & (patients_with_dementia[:,2] < 70), 1,0)
tier_4_sum = np.sum(tier_4)
print("50-70yo:", tier_4_sum,";", tier_4_sum/total)
tier_5 = np.where(patients_with_dementia[:,2] > 70, 1,0)
tier_5_sum = np.sum(tier_5)
print("70+:", tier_5_sum,";", tier_5_sum/total)


patients_df = pd.DataFrame(patients_with_dementia, columns=[
    "subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"
])


patients_df["anchor_age"] = patients_df["anchor_age"].astype(int)

patients_df["deceased"] = patients_df["dod"].notna().astype(int)

patients_df["age_group"] = pd.cut(
    patients_df["anchor_age"],
    bins=[30, 50, 70, 120],
    labels=["30–50", "50–70", "70+"],
    right=True
)

columns = ["gender", "anchor_age", "age_group", "anchor_year_group"]
categorical = ["gender", "age_group", "anchor_year_group"]
nonnormal = ["anchor_age"]  # will use median [IQR] instead of mean ± SD

rename = {
    "gender": "Sex",
    "anchor_age": "Age (years)",
    "age_group": "Age Group",
    "anchor_year_group": "Year Group",
}

# Generate Table 1
t1 = TableOne(
    patients_df,
    columns=columns,
    categorical=categorical,
    nonnormal=nonnormal,
    rename=rename,
    missing=True,       # show missing data counts
    smd=False
)

print(t1.tabulate(tablefmt="grid"))

