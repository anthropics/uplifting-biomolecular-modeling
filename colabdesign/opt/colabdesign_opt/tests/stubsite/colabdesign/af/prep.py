"""stand-in colabdesign.af.prep (tests only): what nosub.py's _prep_model replacement reads."""
import copy


def copy_dict(d):
    return copy.deepcopy(d)


class _af_prep:
    def _prep_model(self, **kwargs):
        self._cfg.model.global_config.subbatch_size = None
        self._model = self._get_model(self._cfg)
        if sum(self._lengths) > 384:
            self._cfg.model.global_config.subbatch_size = 4
            self._model["fn"] = self._get_model(self._cfg)["fn"]
        self._opt = copy_dict(self.opt)
        self.restart(**kwargs)
