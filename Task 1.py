import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
 
# ── 1. Generate Dataset (embedded — no CSV needed) ───────────
np.random.seed(42)
n = 1000
 
income             = np.random.normal(55000, 20000, n).clip(10000, 150000)
debt               = np.random.normal(15000, 10000, n).clip(0, 80000)
payment_history    = np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.2, 0.3, 0.4])
credit_utilization = np.random.beta(2, 5, n)
 
score = (
    (income / 150000) * 0.35 +
    (1 - debt / 80000) * 0.25 +
    (payment_history / 3) * 0.30 +
    (1 - credit_utilization) * 0.10
)
creditworthy = (score + np.random.normal(0, 0.05, n) > 0.5).astype(int)
 
df = pd.DataFrame({
    'Income'            : income.round(2),
    'Debt'              : debt.round(2),
    'Payment_History'   : payment_history,
    'Credit_Utilization': credit_utilization.round(4),
    'Creditworthy'      : creditworthy
})
 
# Also save CSV alongside this script
df.to_csv('credit_dataset.csv', index=False)
 
print("=" * 60)
print("  CREDIT SCORING MODEL — HARSHIT (25AM031)")
print("=" * 60)
print(f"\n📂 Dataset Shape : {df.shape}")
print(f"   Features      : {list(df.columns[:-1])}")
print(f"\n📊 Class Distribution:")
vc = df['Creditworthy'].value_counts()
print(f"   Creditworthy (1)     : {vc.get(1, 0)}")
print(f"   Not Creditworthy (0) : {vc.get(0, 0)}")
 
# ── 2. Preprocessing ─────────────────────────────────────────
print("\n── Preprocessing ──")
print(f"   Missing values : {df.isnull().sum().sum()}")
 
X = df.drop('Creditworthy', axis=1)
y = df['Creditworthy']
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size : {len(X_train)} | Test size : {len(X_test)}")
 
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
 
# ── 3. Train Models ──────────────────────────────────────────
models = {
    "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}
 
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),
        "F1-Score" : f1_score(y_test, y_pred),
        "ROC-AUC"  : roc_auc_score(y_test, y_prob),
    }
 
# ── 4. Print Metrics ─────────────────────────────────────────
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
print("\n" + "=" * 60)
print("  PERFORMANCE METRICS")
print("=" * 60)
header = f"\n{'Model':<22} " + "  ".join(f"{m:>10}" for m in metric_cols)
print(header)
print("-" * len(header))
for name, r in results.items():
    vals = f"{name:<22} " + "  ".join(f"{r[m]:>10.4f}" for m in metric_cols)
    print(vals)
 
best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
best      = results[best_name]
print(f"\n🏆 Best Model (by ROC-AUC): {best_name}")
 
# ── 5. Sample Predictions ────────────────────────────────────
print("\n" + "=" * 60)
print("  SAMPLE PREDICTIONS (Best Model — first 10 test rows)")
print("=" * 60)
print(f"\n{'#':<4} {'Income':>10} {'Debt':>10} {'PayHist':>8} {'Util':>8}  {'Actual':>8} {'Predicted':>10} {'Prob':>8}")
print("-" * 70)
for i in range(10):
    row    = X_test.values[i]
    actual = y_test.values[i]
    pred   = best["y_pred"][i]
    prob   = best["y_prob"][i]
    print(f"{i+1:<4} {row[0]:>10.0f} {row[1]:>10.0f} {row[2]:>8.0f} {row[3]:>8.3f}"
          f"  {'Yes' if actual==1 else 'No':>8}"
          f"  {'✅ Yes' if pred==1 else '❌ No':>10} {prob:>8.3f}")
 
# ── 6. Visualizations ────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f172a')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
 
ACCENT = '#38bdf8'; GOLD = '#fbbf24'; GREEN = '#4ade80'; RED = '#f87171'
BG = '#1e293b'; TXT = '#e2e8f0'
plt.rcParams.update({'text.color': TXT, 'axes.labelcolor': TXT,
                     'xtick.color': TXT, 'ytick.color': TXT})
colors = [ACCENT, GOLD, GREEN]
 
# Metric bar chart
ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor(BG)
x   = np.arange(len(metric_cols)); bar_w = 0.25
for i, (name, r) in enumerate(results.items()):
    ax1.bar(x + i*bar_w, [r[m] for m in metric_cols], bar_w,
            label=name, color=colors[i], alpha=0.88)
ax1.set_xticks(x + bar_w); ax1.set_xticklabels(metric_cols, fontsize=11)
ax1.set_ylim(0, 1.15); ax1.set_ylabel("Score", fontsize=11)
ax1.set_title("Model Performance Comparison", fontsize=13, fontweight='bold', color=GOLD)
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.2)
 
# ROC curves
ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor(BG)
for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    ax2.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name} ({r['ROC-AUC']:.3f})")
ax2.plot([0,1],[0,1],'--', color='#64748b', lw=1)
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("ROC Curves", fontsize=13, fontweight='bold', color=GOLD)
ax2.legend(fontsize=8); ax2.grid(alpha=0.15)
 
# Confusion matrix
ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor(BG)
cm  = confusion_matrix(y_test, best["y_pred"])
ax3.imshow(cm, cmap='Blues')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Not CW','Creditworthy']); ax3.set_yticklabels(['Not CW','Creditworthy'])
for ii in range(2):
    for jj in range(2):
        ax3.text(jj, ii, str(cm[ii,jj]), ha='center', va='center',
                 fontsize=14, fontweight='bold', color='white')
ax3.set_title(f"Confusion Matrix\n({best_name})", fontsize=11, fontweight='bold', color=GOLD)
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
 
# Feature importance
ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor(BG)
rf  = results["Random Forest"]["model"]
imp = rf.feature_importances_
idx = np.argsort(imp)
ax4.barh([X.columns[i] for i in idx], imp[idx], color=ACCENT, alpha=0.85)
ax4.set_title("Feature Importance\n(Random Forest)", fontsize=11, fontweight='bold', color=GOLD)
ax4.set_xlabel("Importance"); ax4.grid(axis='x', alpha=0.2)
 
# Probability distribution
ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor(BG)
ax5.hist(best["y_prob"][y_test==0], bins=25, alpha=0.7, color=RED,   label='Not Creditworthy')
ax5.hist(best["y_prob"][y_test==1], bins=25, alpha=0.7, color=GREEN, label='Creditworthy')
ax5.axvline(0.5, color=GOLD, linestyle='--', lw=1.5, label='Threshold 0.5')
ax5.set_xlabel("Predicted Probability"); ax5.set_ylabel("Count")
ax5.set_title(f"Probability Distribution\n({best_name})", fontsize=11, fontweight='bold', color=GOLD)
ax5.legend(fontsize=8); ax5.grid(alpha=0.15)
 
fig.suptitle("Credit Scoring Model — Full Report", fontsize=16,
             fontweight='bold', color=GOLD, y=1.01)
 
plt.savefig('credit_scoring_results.png', dpi=150, bbox_inches='tight',
            facecolor='#0f172a')
print("\n📊 Chart saved → credit_scoring_results.png")
print("📁 Dataset saved → credit_dataset.csv")
print("✅ Done.")
 
