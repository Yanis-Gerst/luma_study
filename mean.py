import pandas as pd
modality = ["image", "audio", "text", "multimodal"]

mean_acc = {}

for m in modality:
    acc = 0
    for j in range(1, 11):
        file = f"unimodal_results/{m}_{j}.csv"
        csv = pd.read_csv(file)
        acc += float(csv.iloc[0, 1])
    acc /= 10
    mean_acc[m] = [acc]

mean_acc = pd.DataFrame.from_dict(
    mean_acc)
mean_acc.to_csv("unimodal_mean_acc.csv", index=False)
