import yaml


def yamlobj(tag):
    def wrapper(cls):
        def constructor(loader, node):
            fields = loader.construct_mapping(node, deep=True)
            obj = cls(**fields)
            obj._yaml_fields = fields
            return obj

        yaml.add_constructor(tag, constructor)
        return cls

    return wrapper
