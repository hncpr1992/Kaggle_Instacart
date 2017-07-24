def f_score(y_pred, y_true):
    """
    y_pred: python list, eg: [12345, 12321, 22221]
    y_true: string, eg: [12345, 12321, 22221]
    """
    y_pred_set = set(y_pred)
    if len(y_true) > 0:
        y_true_set = set([int(x) for x in y_true])
    else:
        y_true_set = set([])
    TP = len(y_pred_set.intersection(y_true_set))
    FP = len(y_pred_set) - TP
    FN = len(y_true_set) - TP
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f_score = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        return 0
    else:
        return f_score