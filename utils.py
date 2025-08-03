from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from csv import DictReader
import config


def read_annotations_csv(filename: str):
    res = {}
    with open(filename) as fhandle:
        reader = DictReader(fhandle)
        for row in reader:
            res[row["filename"]] = row["class"]
    return res


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == "all" or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt), all_cnt, ok_cnt


def warmup_then_cosine_annealing_lr(
    optimizer,
    start_factor,
    total_steps,
    warmup_duration,
):
    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_duration,
    )
    cos_annealing = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_duration,
    )
    warmup_then_cos_anneal = SequentialLR(
        optimizer,
        [warmup, cos_annealing],
        milestones=[warmup_duration],
    )
    return warmup_then_cos_anneal


def format_test_results(total_tuple, rare_tuple, freq_tuple):
    def format_result(name, res_tuple):
        acc, cnt, ok = res_tuple
        print(f'Recall for type={name}: {acc:.3f}, Total signs of this type: {cnt}, TP signs: {ok}')

    for res_tuple, type_name in zip([total_tuple, rare_tuple, freq_tuple], ['all', 'rare', 'freq']):
        format_result(type_name, res_tuple)


def update_global_vars(config_data):
    for k in config_data.keys():
        if k.upper() in config.__all__:
            setattr(config, k.upper(), config_data[k])
