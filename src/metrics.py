import torch


def intersection_over_union(
    predictions,
    targets,
    num_classes
):

    ious = []

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    for class_id in range(num_classes):

        predicted_class = predictions == class_id
        target_class = targets == class_id

        intersection = (
            predicted_class & target_class
        ).sum().float()

        union = (
            predicted_class | target_class
        ).sum().float()

        if union == 0:
            iou = float("nan")
        else:
            iou = (
                intersection / union
            ).item()

        ious.append(iou)

    return ious


def mean_iou(ious):

    valid_ious = [
        value for value in ious
        if value == value
    ]

    if not valid_ious:
        return 0.0

    return sum(valid_ious) / len(valid_ious)