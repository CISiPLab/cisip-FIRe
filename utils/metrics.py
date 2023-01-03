def calculate_accuracy(logits, labels, onehot=True, multiclass=False, topk=1):
    if multiclass or topk != 1:
        if topk == 1:
            topk = 5
        pred = logits.topk(topk, 1, True, True)[1].t()
        if onehot:
            labels = labels.argmax(1)
        correct = pred.eq(labels.reshape(1, -1).expand_as(pred))
        acc = correct[:topk].reshape(-1).float().sum(0, keepdim=True) / logits.size(0)
    else:
        if len(labels.size()) == 2:
            labels = labels.argmax(1)

        acc = (logits.argmax(1) == labels).float().mean()
    return acc


def calculate_accuracy_hamm_dist(hamm_dist, labels, onehot=True, multiclass=False):
    if multiclass:
        pred = hamm_dist.topk(5, 1, False, True)[1].t()
        correct = pred.eq(labels.argmax(1).reshape(1, -1).expand_as(pred))
        cbacc = correct[:5].reshape(-1).float().sum(0, keepdim=True) / hamm_dist.size(0)
    else:
        if len(labels.size()) == 2:
            labels = labels.argmax(1)

        cbacc = (hamm_dist.argmin(1) == labels).float().mean()

    return cbacc
