import re

class lazyval:
    """Decorator to lazily compute and cache a value.
    """
    def __init__(self, fget):
        self._fget = fget
        self._name = None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self

        value = self._fget(instance)
        vars(instance)[self._name] = value
        return value

    def __set__(self, instance, value):
        vars(instance)[self._name] = value


def parseOsuMp3Filename(name):
    '''Uses regex to find filename info'''
    match = re.search('^[0-9]*\s*(.*)\s+\-\s+(.*)\s+',name)
    artist = match.group(1)
    song = match.group(2)
    if '[' in name:
        # get diff name from enclosed [] brackets, choose closest to end
        diff = re.findall('\[([^\[]]*)\]',name)
        if diff:
            diff = diff[-1]
    else:
        diff = None
    return artist,song,diff