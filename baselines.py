import numpy as np
import pandas as pd

from constants import ENZYME_COLS
from utils import dose2str

WT_COL = 'Weight (kg)'
HEIGHT_COL = 'Height (cm)'


def compute_fixed_dose_acc(warfarin):
    arr = np.array(warfarin) == 'medium'
    print(f'Fixed dose: {arr.mean()}')
    return arr.mean()


def compute_clinical_dose_acc(data, warfarin):
    dec_age = [int(a[0]) for a in list(data['Age'])]
    height = list(data['Height (cm)'])
    weight = list(data['Weight (kg)'])
    asian = [1 if r == "Asian" else 0 for r in list(data['Race'])] 
    black = [1 if r == "Black or African American" else 0 for r in list(data['Race'])] 
    missing = [1 if r == "Unknown" else 0 for r in list(data['Race'])]
    enzyme = data[ENZYME_COLS].fillna(0).sum(1).clip(0, 1).values
    amiodarone = data['Amiodarone (Cordarone)'].fillna(0).values
    num_correct = 0
    for i in range(len(warfarin)):
        dose = 4.0376 - 0.2546 * dec_age[i] + 0.0118 * height[i] + 0.0134 * weight[i]
        dose = dose - 0.6752 * asian[i] + 0.4060 * black[i] + 0.0443 * missing[i]
        dose = dose + 1.2799 * enzyme[i] - 0.5695 * amiodarone[i]
        dose = dose ** 2
        if dose2str(dose) == warfarin[i]:
            num_correct += 1

    print(f'Clinical Dose Algorithm accuracy: { (num_correct / len(warfarin))}')
    return num_correct / len(warfarin)

VK_COL = 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'


def preprocess(data):
    data['enzyme_sum'] = data[ENZYME_COLS].fillna(0).sum(1).clip(1, None).values
    data['Age'] = data['Age'].str.partition('-')[0].str.strip('+').astype(int)
    return data

def compute_pharma_dose_acc(data, warfarin):
    dec_age = [int(str(a)[0]) for a in list(data['Age'])]
    height = list(data[HEIGHT_COL])
    weight = list(data[WT_COL])
    asian = [1 if r == "Asian" else 0 for r in list(data['Race'])] 
    black = [1 if r == "Black or African American" else 0 for r in list(data['Race'])] 
    missing = [1 if r == "Unknown" else 0 for r in list(data['Race'])]
    enzyme = data[ENZYME_COLS].fillna(0).sum(1).clip(0, 1).values
    amiodarone = data['Amiodarone (Cordarone)'].fillna(0).values
    vk_ag = (data[VK_COL] == 'A/G').values
    vk_aa = (data[VK_COL] == 'A/A').values
    vk_unknown = data[VK_COL].isnull().values
    CYP_COL = 'Cyp2C9 genotypes'
    cyp_1_2 = (data[CYP_COL] == "*1/*2").values
    cyp_1_3 = (data[CYP_COL] == "*1/*3").values
    cyp_2_2 = (data[CYP_COL] == "*2/*2").values
    cyp_2_3 = (data[CYP_COL] == "*2/*3").values
    cyp_3_3 = (data[CYP_COL] == "*3/*3").values
    cyp_unknown = data[CYP_COL].isnull().values

    num_correct = 0
    for i in range(len(warfarin)):
        dose = 5.6044 - 0.2614 * dec_age[i] + 0.0087 * height[i] + 0.0128 * weight[i]
        dose = dose - 0.8677 * vk_ag[i] - 1.6974 * vk_aa[i] - 0.4854 * vk_unknown[i]
        dose = dose - 0.5211 * cyp_1_2[i] - 0.9357 * cyp_1_3[i] - 1.0616 * cyp_2_2[i] - 1.9206 * cyp_2_3[i] - 2.3312 * cyp_3_3[i] - 0.2188 * cyp_unknown[i]
        dose = dose - 0.1092 * asian[i] - 0.2760 * black[i] - 0.1032 * missing[i]
        dose = dose + 1.1816 * enzyme[i] - 0.5503 * amiodarone[i]
        dose = dose ** 2
        dose = dose2str(dose)
        if dose == warfarin[i]:
            num_correct += 1

    print(f"Pharmacogenetic Dose Algorithm accuracy:  {(num_correct / len(warfarin))}")
    return (num_correct / len(warfarin))

if __name__ == "__main__":
    data = pd.read_csv("warfarin_data/warfarin.csv")
    data = data.dropna(subset=['Age', 'Therapeutic Dose of Warfarin'])
    data.rename(columns={"Therapeutic Dose of Warfarin": "warfarin", }, inplace=True)
    data['warfarin_str'] = data['warfarin'].apply(dose2str)
    targ_str_array = data['warfarin_str'].values
    fixed_dose = compute_fixed_dose_acc(targ_str_array)
    clin_dose = compute_clinical_dose_acc(data, targ_str_array)
    pharma_dose = compute_pharma_dose_acc(data, targ_str_array)
