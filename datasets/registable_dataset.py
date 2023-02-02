class RegisterableDataset():
    def get_dataset_dicts(self):
        return NotImplementedError
    @property
    def name(self):
        return NotImplementedError

    @property
    def json_file(self):
        return NotImplementedError

    @property
    def image_root(self):
        return NotImplementedError

    @property
    def evaluator_type(self):
        return NotImplementedError

    @property
    def thing_colors(self):
        return NotImplementedError

    @property
    def thing_classes(self):
        return NotImplementedError

    @property
    def thing_dataset_id_to_contiguous_id(self):
        return NotImplementedError
