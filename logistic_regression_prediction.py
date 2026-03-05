import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import roc_curve

### Load the data
train_df = pd.read_csv("training_data.csv")
test_df  = pd.read_csv("la_count_data.csv")

train_df["LA_provider"] = train_df["LA_provider"].astype(int)
test_df["LA_provider"]  = test_df["LA_provider"].astype(int)

full_df = pd.concat([train_df, test_df], ignore_index=True)

### Fit on the training data
model_train = smf.glm(
    formula="LA_provider ~ PROP",
    data=train_df,
    family=sm.families.Binomial()
).fit()


### Predict on test and find optimal probability threshold (Youden's J)
test_prob = model_train.predict(test_df)
y_true = test_df["LA_provider"].values

fpr, tpr, thresholds = roc_curve(y_true, test_prob)
j = tpr - fpr
best_idx = np.argmax(j)
best_threshold = thresholds[best_idx]

print("Optimal probability threshold (Youden's J):", round(best_threshold,3))

### Refit on FULL for final analysis
model_full = smf.glm(
    formula="LA_provider ~ PROP",
    data=full_df,
    family=sm.families.Binomial()
).fit()

###  Convert best_threshold -> PROP cutoff using FULL model coefficients
b0 = model_full.params["Intercept"]
b1 = model_full.params["PROP"]

# logit(t) = ln(t/(1-t))
logit_t = np.log(best_threshold / (1 - best_threshold))

prop_cutoff = (logit_t - b0) / b1
print("Implied optimal PROP cutoff:", round(prop_cutoff, 3))

### Score full for plotting & misclassification coloring
plot_df = full_df.copy()
plot_df["p_hat"] = model_full.predict(plot_df)
plot_df["y_hat"] = (plot_df["p_hat"] >= best_threshold).astype(int)
plot_df["is_misclassified"] = (plot_df["y_hat"] != plot_df["LA_provider"])

# Curve grid
x_min, x_max = plot_df["PROP"].min(), plot_df["PROP"].max()
x_grid = np.linspace(x_min, x_max, 400)
p_grid = model_full.predict(pd.DataFrame({"PROP": x_grid}))

# Jitter y for visualization
rng = np.random.default_rng(0)
y_jitter = plot_df["LA_provider"].astype(float) + rng.normal(0, 0.03, size=len(plot_df))

correct = plot_df[~plot_df["is_misclassified"]]
wrong   = plot_df[ plot_df["is_misclassified"]]

### Plot
plt.figure(figsize=(9, 6))

plt.plot(x_grid, p_grid, linewidth=2, label="Fitted P(LA_provider=1 | PROP)")

plt.scatter(correct["PROP"], y_jitter.loc[correct.index], alpha=0.6, label="Correctly classified")
plt.scatter(wrong["PROP"],   y_jitter.loc[wrong.index],   alpha=0.9, label="Misclassified")

# Draw the chosen probability threshold and the implied PROP cutoff
plt.axhline(best_threshold, linestyle="--", linewidth=1, label=f"Optimal p threshold = {best_threshold:.3f}")
plt.axvline(prop_cutoff,     linestyle="--", linewidth=1, label=f"Implied PROP cutoff = {prop_cutoff:.6g}")

plt.ylim(-0.15, 1.15)
plt.xlabel("PROP")
plt.ylabel("LA_provider (0/1) with jitter")
plt.title("Logistic regression on full data with optimal PROP cutoff")
plt.legend()
plt.tight_layout()
plt.savefig("results_figure.png", dpi=300)
plt.show()