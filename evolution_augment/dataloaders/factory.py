import dataclasses


class DataloaderFactory:
    _REGISTRY = {}

    @classmethod
    def register(cls, name=None):

        def _wrapper(dataloader_cls):
            dataloader_name = name if name else dataloader_cls.__name__

            if dataloader_name in cls._REGISTRY:
                raise ValueError(
                    'Dataloader with name: {} already registered'.format(
                        dataloader_name))

            cls._REGISTRY[dataloader_name] = dataloader_cls
            print('Registered dataloader: {} from {}'.format(dataloader_name, dataloader_cls))
            return dataloader_cls

        return _wrapper

    @classmethod
    def build(cls, config):
        name = config.dataset.name
        if name not in cls._REGISTRY:
            raise ValueError(
                'Dataloader with name: {} is not registered'.format(name))

        config =  dataclasses.asdict(config)
        dataset_config = config.pop('dataset')
        config.update(dataset_config)
        print('Building {} dataloader with config:\n{}'.format(name, config))
        model = cls._REGISTRY[name](config).build()
        return model