import json
import csv 

with open('preds_0.json', 'r') as f:
    pred_0 = json.load(f)
with open('preds_50.json', 'r') as f:
    pred_50 = json.load(f)
with open('preds_100.json', 'r') as f:
    pred_100 = json.load(f)
with open('preds_500.json', 'r') as f:
    pred_500 = json.load(f)
with open('preds_1000.json', 'r') as f:
    pred_1000 = json.load(f)
with open('preds_10000.json', 'r') as f:
    pred_10000 = json.load(f)
with open('labels_0.json', 'r') as f:
    labels = json.load(f)


with open('result.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['building', 'true', 'TS0', 'TS50', 'TS100', 'TS500', 'TS1000', 'TS10000'])
    for idx in range(len(pred_0)):
        l = labels[idx]
        writer.writerow([idx, 1-l, pred_0[idx][0], pred_50[idx][0], pred_100[idx][0], pred_500[idx][0], pred_1000[idx][0], pred_10000[idx][0]])