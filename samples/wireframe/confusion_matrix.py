import mrcnn.model as modellib
import random
import samples.wireframe.plots as plots

def confusion_matrix(EVALUATE, NORMALIZE, dataset_val, dataset_train, inference_config, model):
    class_names = dataset_train.class_names
    real_label = []
    pred_label = []

    for _ in range(EVALUATE):
        image_id = random.choice(dataset_val.image_ids)
        original_image, _, gt_class_id, _, _ =\
            modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
        r, _ = model.OneShotDetect(original_image)
        r = list(r.values())
        r = [i[0][0].decode("utf-8") for i in r]
        foo = [class_names[i] for i in gt_class_id]
        r.sort()
        foo.sort()
        if len(r) != len(foo):
            continue
        elif set(r) == set(foo):
            real_label.extend(foo)
            pred_label.extend(r)
        else:
            real_label.extend(foo)
            pred_label.extend(r)

    confusion_matrix = plots.plot_confusion_matrix(real_label, pred_label, normalize=NORMALIZE)
    return confusion_matrix