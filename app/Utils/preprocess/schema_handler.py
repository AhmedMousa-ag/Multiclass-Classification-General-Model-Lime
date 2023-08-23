


def __get_prep_param(data_schema):
    pp_params = {}
    pp_params["id"] = data_schema["id"]["name"]
    pp_params["target_col"] = data_schema["target"]["name"]
    pp_params["target_classes"] = data_schema["target"]["classes"]
    features= []
    for feature in data_schema["features"]:
        feat = {}
        feat["name"] = feature["name"]
        feat["data_type"] = feature["dataType"]
        if feature["dataType"] == "CATEGORICAL":
            feat["categories"] = feature["categories"]
        features.append(feat)
    pp_params["features"] = features
    return pp_params


def produce_schema_param(data_schema):
    # initiate the pp_params dict
    pp_params = __get_prep_param(data_schema)
    return pp_params

