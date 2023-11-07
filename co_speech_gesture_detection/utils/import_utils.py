def import_class(name, base_module=None):
    if base_module is None:
        base_module = "co_speech_gesture_detection" 
    components = name.split('.')
    mod = __import__(base_module + "." + components[0]) 
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod