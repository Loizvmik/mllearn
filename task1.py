import pandas as pd

data=pd.read_csv("data/classification_predictions.csv")
gold=data["y_true"]
sc1=data["model_a_score"]
sc2=data["model_b_score"]
pred1tp,pred1tn,pred1fp,pred1fn=[],[],[],[]
pred2tp,pred2tn,pred2fp,pred2fn=[],[],[],[]
pred_gold=[]
t=0.5
def evaluate_models(gold, sc1, sc2, threshold=0.5):
    tp1 = tp2 = tn1 = tn2 = fp1 = fp2 = fn1 = fn2 = 0
    
    for i in range(len(gold)):
        if sc1[i] >= threshold:
            if gold[i] == 1:
                tp1 += 1
            else:
                fp1 += 1
        else:
            if gold[i] == 1:
                fn1 += 1
            else:
                tn1 += 1
        
        if sc2[i] >= threshold:
            if gold[i] == 1:
                tp2 += 1
            else:
                fp2 += 1
        else:
            if gold[i] == 1:
                fn2 += 1
            else:
                tn2 += 1
    
    pos = sum(gold)
    neg = len(gold) - pos
    
    return {
        "model_a": {"tp": tp1, "tn": tn1, "fp": fp1, "fn": fn1, "accuracy": (tp1 + tn1) / len(gold), "precision": tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0, "recall": tp1 / (tp1 + fn1) if pos > 0 else 0},
        "model_b": {"tp": tp2, "tn": tn2, "fp": fp2, "fn": fn2, "accuracy": (tp2 + tn2) / len(gold), "precision": tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0, "recall": tp2 / (tp2 + fn2) if pos > 0 else 0}
    }
results = evaluate_models(gold, sc1, sc2, threshold=t)
print(results)
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    results = evaluate_models(gold, sc1, sc2, threshold=t)
    if results["model_a"]["recall"] >= 0.8 and results["model_b"]["recall"] >= 0.8:
        print(f"Threshold: {t}")
        break
