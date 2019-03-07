import pandas as pd
import numpy as np

def compute_fixed_dose_acc(warfarin):
    meds = [1 if d == "medium" else 0 for d in warfarin]
    print("Fixed Dose accuracy: %s" % (sum(meds) / len(meds)))


def compute_clinical_dose_acc(data, warfarin):
    dec_age = [int(a[0]) for a in list(data['Age'])]
    height = list(data['Height (cm)'])
    weight = list(data['Weight (kg)'])
    asian = [1 if r == "Asian" else 0 for r in list(data['Race'])] 
    black = [1 if r == "Black or African American" else 0 for r in list(data['Race'])] 
    missing = [1 if r == "Unknown" else 0 for r in list(data['Race'])] 
    carbamazepine = [1 if c == 1 else 0 for c in list(data["Carbamazepine (Tegretol)"])]
    phenytoin = [1 if p == 1 else 0 for p in list(data["Phenytoin (Dilantin)"])]
    rifampin = [1 if r == 1 else 0 for r in list(data["Rifampin or Rifampicin"])]
    enzyme = [max(1, carbamazepine[i] + phenytoin[i] + rifampin[i]) for i in range(len(rifampin))]
    amiodarone = list(data['Amiodarone (Cordarone)'])

    num_correct = 0
    for i in range(len(warfarin)):
        dose = 4.0376 - 0.2546 * dec_age[i] + 0.0118 * height[i] + 0.0134 * weight[i]
        dose = dose - 0.6752 * asian[i] + 0.4060 * black[i] + 0.0443 * missing[i]
        dose = dose + 1.2799 * enzyme[i] - 0.5695 * amiodarone[i]
        dose = dose ** 2
        dose = "low" if dose < 21 else ("high" if dose > 49 else "medium")
        if dose == warfarin[i]:
            num_correct += 1

    print("Clinical Dose Algorithm accuracy: %s" % (num_correct / len(warfarin)))


def compute_pharma_dose_acc(data, warfarin):
    dec_age = [int(str(a)[0]) for a in list(data['Age'])]
    height = list(data['Height (cm)'])
    weight = list(data['Weight (kg)'])
    asian = [1 if r == "Asian" else 0 for r in list(data['Race'])] 
    black = [1 if r == "Black or African American" else 0 for r in list(data['Race'])] 
    missing = [1 if r == "Unknown" else 0 for r in list(data['Race'])] 
    carbamazepine = [1 if c == 1 else 0 for c in list(data["Carbamazepine (Tegretol)"])]
    phenytoin = [1 if p == 1 else 0 for p in list(data["Phenytoin (Dilantin)"])]
    rifampin = [1 if r == 1 else 0 for r in list(data["Rifampin or Rifampicin"])]
    enzyme = [max(1, carbamazepine[i] + phenytoin[i] + rifampin[i]) for i in range(len(rifampin))]
    amiodarone = list(data['Amiodarone (Cordarone)'])
    vk_ag = [1 if g == "A/G" else 0 for g in list(data['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'])]
    vk_aa = [1 if g == "A/A" else 0 for g in list(data['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'])]
    vk_unknown = [1 if pd.isnull(g) else 0 for g in list(data['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'])]
    cyp_1_2 = [1 if g == "*1/*2" else 0 for g in list(data['Cyp2C9 genotypes'])]
    cyp_1_3 = [1 if g == "*1/*3" else 0 for g in list(data['Cyp2C9 genotypes'])]
    cyp_2_2 = [1 if g == "*2/*2" else 0 for g in list(data['Cyp2C9 genotypes'])]
    cyp_2_3 = [1 if g == "*2/*3" else 0 for g in list(data['Cyp2C9 genotypes'])]
    cyp_3_3 = [1 if g == "*3/*3" else 0 for g in list(data['Cyp2C9 genotypes'])]
    cyp_unknown = [1 if pd.isnull(g) else 0 for g in list(data['Cyp2C9 genotypes'])]

    num_correct = 0
    for i in range(len(warfarin)):
        dose = 5.6044 - 0.2614 * dec_age[i] + 0.0087 * height[i] + 0.0128 * weight[i]
        dose = dose - 0.8677 * vk_ag[i] - 1.6974 * vk_aa[i] - 0.4854 * vk_unknown[i]
        dose = dose - 0.5211 * cyp_1_2[i] - 0.9357 * cyp_1_3[i] - 1.0616 * cyp_2_2[i] - 1.9206 * cyp_2_3[i] - 2.3312 * cyp_3_3[i] - 0.2188 * cyp_unknown[i]
        dose = dose - 0.1092 * asian[i] - 0.2760 * black[i] - 0.1032 * missing[i]
        dose = dose + 1.1816 * enzyme[i] - 0.5503 * amiodarone[i]
        dose = dose ** 2
        dose = "low" if dose < 21 else ("high" if dose > 49 else "medium")
        if dose == warfarin[i]:
            num_correct += 1

    print("Pharmacogenetic Dose Algorithm accuracy: %s" % (num_correct / len(warfarin)))


if __name__ == "__main__":
    data = pd.read_csv("data/warfarin.csv")
    data = data.dropna(subset=['Age'])
    data.rename(columns={"Therapeutic Dose of Warfarin": "warfarin"}, inplace=True)

    warfarin = list(data.warfarin)
    for i in range(len(warfarin)):
        if warfarin[i] < 21:
            warfarin[i] = "low"
        elif warfarin[i] > 49:
            warfarin[i] = "high"
        else:
            warfarin[i] = "medium"

    compute_fixed_dose_acc(warfarin)
    compute_clinical_dose_acc(data, warfarin)
    compute_pharma_dose_acc(data, warfarin)
