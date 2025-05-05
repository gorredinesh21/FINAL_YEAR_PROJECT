from statsmodels.stats.contingency_tables import mcnemar

def perform_mcnemars_test(y_test, y_pred_original, y_pred_augmented):
    b = ((y_pred_original == 0) & (y_pred_augmented == 1) & (y_test == 1)).sum()
    c = ((y_pred_original == 1) & (y_pred_augmented == 0) & (y_test == 1)).sum()

    table = [[0, b], [c, 0]]

    result = mcnemar(table, exact=False, correction=True)

    print("----- McNemar’s Test -----")
    print(f"Statistic = {result.statistic}, p-value = {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print(" The difference is statistically significant (p < 0.05)")
    else:
        print(" No significant difference between the two models")
