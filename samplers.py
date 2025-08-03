import torch
import numpy as np

from dataset import DatasetRTSD


class IndexSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source: DatasetRTSD, examples_per_class: int | None) -> None:
        self.data_source = data_source
        self.elems_per_class = examples_per_class

    def __iter__(self):
        def class_cycle(class_i):
            class_samples = np.array(self.data_source.classes_to_samples[class_i])
            np.random.shuffle(class_samples)
            elem_iter = 0
            while True:
                elem_list = list()
                if self.elems_per_class is None:
                    limit = len(class_samples)
                else:
                    limit = self.elems_per_class
                while len(elem_list) < limit:
                    elem_list.append(class_samples[elem_iter])
                    elem_iter = (elem_iter + 1) % len(class_samples)
                yield elem_list

        ret_list = list()
        class_cycles = [class_cycle(cls_i) for cls_i in range(len(self.data_source.classes_to_samples))]
        for class_i in range(len(self.data_source.classes_to_samples)):
            if len(self.data_source.classes_to_samples[class_i]) == 0:
                continue
            ret_list += next(class_cycles[class_i])

        return iter(ret_list)

    def __len__(self) -> int:
        N_classes = 0
        for cls in range(len(self.data_source.classes)):
            if len(self.data_source.classes[cls]) > 0:
                N_classes += 1
        if self.elems_per_class is not None:
            return N_classes * self.elems_per_class
        else:
            return len(self.data_source)


class CustomBatchSampler(torch.utils.data.sampler.Sampler):

    def __init__(
        self,
        data_source: DatasetRTSD,
        elems_per_class: int,
        classes_per_batch: int
    ) -> None:
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch

    def __iter__(self):
        def sampler():
            data_classes_to_sample = self.data_source.classes_to_samples

            def class_cycle(class_i):
                elem_iter = 0
                class_samples = np.array(data_classes_to_sample[class_i])
                elem_iter = 0
                cycle_start = elem_iter
                while True:
                    passed_cycle = False
                    elem_list = list()
                    while len(elem_list) < self.elems_per_class:
                        elem_list.append(class_samples[elem_iter])
                        elem_iter = (elem_iter + 1) % len(class_samples)
                        passed_cycle = elem_iter == cycle_start

                    if passed_cycle:
                        cycle_start = elem_iter
                        np.random.shuffle(class_samples)
                        
                    yield elem_list
            cls_cycles = [class_cycle(cls) for cls in range(len(data_classes_to_sample))]

            unused_classes = set(range(len(data_classes_to_sample)))
            while len(unused_classes) > 0:
                classes = np.random.choice(np.array(list(unused_classes)), replace=False, size=min(len(unused_classes), self.classes_per_batch))
                if len(unused_classes) < self.classes_per_batch:
                    used_classes = [i for i in range(len(data_classes_to_sample)) if i not in unused_classes]
                    extra_classes = np.random.choice(np.array(used_classes), replace=False, size=self.classes_per_batch - len(unused_classes))
                    classes = np.concatenate([classes, extra_classes])
                ret_list = list()
                for cls in classes:
                    ret_list += next(cls_cycles[cls])
                yield from ret_list
                unused_classes = unused_classes - set(classes)
        
        return sampler()

    def __len__(self) -> int:
        N_classes = len(self.data_source.classes_to_samples)
        return self.elems_per_class * self.classes_per_batch * (N_classes + int(N_classes % self.classes_per_batch != 0))