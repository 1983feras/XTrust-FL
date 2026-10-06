"""Centralized CICIoT2023 baseline. Outputs real metrics only from supplied data."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xtrust_fl.ciciot2023 import load_ciciot2023, split_and_scale, audit_dataset
from xtrust_fl.model import XTrustMLP


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-rows-per-file", type=int, default=None)
    p.add_argument("--out", default="results/centralized_seed42.json")
    a=p.parse_args(); seed_all(a.seed)
    X,y,labels=load_ciciot2023(a.data,a.max_rows_per_file,a.seed)
    audit=audit_dataset(X,y,labels)
    train,val,test,_,_=split_and_scale(X,y,a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=XTrustMLP(train[0].shape[1],2).to(device)
    loader=DataLoader(TensorDataset(torch.tensor(train[0]),torch.tensor(train[1])),batch_size=a.batch_size,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr); loss_fn=torch.nn.CrossEntropyLoss()
    history=[]; best=None; best_state=None
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.0
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total += loss.item()*len(yb)
        vm=evaluate(model,*val,device); history.append({"epoch":epoch,"train_loss":total/len(train[1]),"val_macro_f1":vm["macro_f1"]})
        if best is None or vm["macro_f1"]>best:
            best=vm["macro_f1"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    result={"dataset_audit":audit,"seed":a.seed,"epochs":a.epochs,"device":str(device),"best_val_macro_f1":best,"test":evaluate(model,*test,device),"history":history}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["test"],indent=2))

if __name__=="__main__": main()
