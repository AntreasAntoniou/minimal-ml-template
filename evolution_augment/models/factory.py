import dataclasses


class ModelFactory:
    _REGISTRY = {}

    @classmethod
    def register(cls, name=None):

        def _wrapper(model_cls):
            model_name = name if name else model_cls.__name__

            if model_name in cls._REGISTRY:
                raise ValueError(
                    'Model with name: {} already registered'.format(
                        model_name))

            cls._REGISTRY[model_name] = model_cls
            print('Registered model: {} from {}'.format(model_name, model_cls))
            return model_cls

        return _wrapper

    @classmethod
    def build(cls, config):
        name = config.__class__.__name__
        if name not in cls._REGISTRY:
            raise ValueError(
                'Model with name: {} is not registered'.format(name))

        print('Building {} model with config:\n{}'.format(name, config))
        model = cls._REGISTRY[name](**dataclasses.asdict(config))
        return model