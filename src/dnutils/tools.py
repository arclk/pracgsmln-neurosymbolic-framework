'''
Created on May 22, 2017

@author: nyga
'''
import re


def ifnone(if_, else_, transform=None):
    '''Returns the condition ``if_`` iff it is not ``None``, or if a transformation is
    specified, ``transform(if_)``. Returns ``else_`` if the condition is ``None``.
    ``transform`` can be any callable, which will be passed ``if_`` in case ``if_`` is not ``None``.'''
    if if_ is None:
        return else_
    else:
        if transform is not None: return transform(if_)
        else: return if_


def ifnot(if_, else_, transform=None):
    '''Returns the condition ``if_`` iff it evaluates to ``True``, or if a transformation is
    specified, ``transform(if_)``. Returns ``else_`` if the condition is ``False``.
    ``transform`` can be any callable, which will be passed ``if_`` in case ``if_`` is not ``False``.'''
    if not bool(if_):
        return else_
    else:
        if transform is not None: return transform(if_)
        else: return if_


def ifstr(arg, transform):
    '''
    Returns ``transform(arg)`` if ``arg`` is a string, or returns ``arg``, otherwise
    :param arg:
    :param transform:
    :return:
    '''
    return transform(arg) if type(arg) is str else arg


def allnone(it):
    '''Returns True iff all elements in the iterable ``it`` are ``None``, and ``False`` otherwise.'''
    return not ([1 for e in it if e is not None])


def allnot(it):
    '''Returns True iff all elements in the iterable ``it`` evaluate to ``False``, and ``False`` otherwise.'''
    return not ([1 for e in it if bool(e) is True])


def idxif(it, idx, transform=None):
    '''Returns the element with the specified index of the iterable ``it``. If a ``transformation`` is specified,
    the result of the ``transformation`` will be returned applied to the element.
    If the iterable is ``None``, or ``it`` does not have enough elements, ``None`` is returned.'''
    try:
        it[idx]
    except (IndexError, TypeError):
        return None
    el = it[idx]
    if transform is not None:
        return transform(el)
    else:
        return el


def first(it, transform=None, else_=None):
    '''
    Returns the first element of the iterable ``it``, if it has any.
    Returns ``None``, if ``it`` is ``None`` or ``it` does not contain any elements. If a transformation is
    specified, the result of the transformation applied to the first element is returned.
    :param transform:
    :param it:
    :return:
    '''
    if it is None:
        return else_
    try:
        el = next(iter(it))
        if transform is not None:
            return transform(el)
        else:
            return el
    except StopIteration:
        pass
    return else_


def last(it, transform=None):
    '''
    Same as :func:`dnutils.tools.first`, but returns the last element.
    :param it:
    :param transform:
    :return:
    '''
    return idxif(it, -1, transform=transform)


sqbrpattern = re.compile(r'\[(-?\d+)\]')


class edict(dict):
    '''
    Enhanced ``dict`` with some convenience methods such as dict addition and
    subtraction.

    Warning: The constructor using keyword arguments, ie. ``dict(one=1, two=2, ...)`` does not work
    with the edict dictionaries. Instead, ``edict``s support default values corresponding to the
    ``defaultdict`` class from the ``itertools`` package.

    :Example:

    >>> s = edict({'a':{'b': 1}, 'c': [1,2,3]})
    >>> r = edict({'x': 'z', 'c': 5})
    >>> print s
    {'a': {'b': 1}, 'c': [1, 2, 3]}
    >>> print r
    {'x': 'z', 'c': 5}
    >>> print s + r
    {'a': {'b': 1}, 'x': 'z', 'c': 5}
    >>> print s - r
    {'a': {'b': 1}}
    >>> print r
    {'x': 'z', 'c': 5}
    '''
    def __init__(self, d=None, default=None, recursive=False):
        if d is None:
            dict.__init__(self)
        else:
            dict.__init__(self, dict(d))
        self._default = default
        if recursive:
            self._recurse()

    def __iadd__(self, d):
        self.update(d)
        return self

    def __isub__(self, d):
        for k in d:
            if k in self: del self[k]
        return self

    def __add__(self, d):
        return type(self)({k: v for items in (self.items(), d.items())for k, v in items})

    def __sub__(self, d):
        return type(self)({k: v for k, v in self.items() if k not in d})

    def __getitem__(self, key):
        if self._default is not None and key not in self:
            self[key] = self._default()
            return self[key]
        else:
            return dict.__getitem__(self, key)

    def _recurse(self):
        for key, value in self.items():
            if type(value) is list:
                self[key] = [edict(v) if hasattr(v, '__getitem__') else v for v in value]
            elif hasattr(value, '__getitem__'):  #type(value) is dict:
                self[key] = edict(value, default=self._default, recursive=True)

    @staticmethod
    def _todict(d, recursive=True):
        d = dict(d)
        if recursive:
            for key, value in d.items():
                if type(value) is edict:
                    d[key] = edict._todict(value, recursive=True)
        return d

    @staticmethod
    def _parse_xpath(selector):
        keys = map(str.strip, selector.split('/'))
        for key in keys:
            m = sqbrpattern.match(key)
            if m is not None:
                yield int(m.group(1))
            else:
                yield key

    def xpath(self, selector, insert=None, force=False):
        '''
        Allows a 'pseudo-xpath' query to a nested set of dictionaries.

        At the moment, only nested dict-selections separated by slashes (``/``) are supported.
        Allows to conveniently access hierarchical dictionart structures without the need
        of checking every key for existence.

        :param selector:    a slash-separated list of dict keys
        :param insert:
        :param force:
        :return:
        '''
        keys = edict._parse_xpath(selector)
        d = self
        for key in keys:
            if type(key) is int:
                d = None if key >= len(d) else d[key]
            else:
                d = d.get(key)
            if d is None:
                if insert is None:
                    return None
                return self.set_xpath(selector, insert, force=force)
        return d

    def set_xpath(self, selector, data, force=False):
        '''
        Creates the xpath structure represented by the selector string, if necessary, to
        set the data to the end point.
        :param selector:
        :param data:
        :return:
        '''
        keys = list(edict._parse_xpath(selector))
        d = self
        for key in keys[:-1]:
            if type(key) is int:
                raise ValueError('indexing in set_xpath() is not yet supported')
            else:
                d_ = d.get(key)
                if d_ is None or not isinstance(d_, dict) and force:
                    d[key] = edict()
                d = d[key]
        d[keys[-1]] = data
        return data

    def pprint(self):
        from pprint import pprint
        pprint(self)

    def project(self, *keys):
        '''
        Returns a copy of this edict that contains only the pairs whose key is in ``keys``.
        :param keys:
        :return:
        '''
        return edict({k: v for k, v in self.items() if k in keys})


class RStorage(edict, object):
    '''
    Recursive extension of web.util.Storage that applies the Storage constructor
    recursively to all value elements that are dicts.
    '''
    __slots__ = ['_utf8']

    def __init__(self, d=None, utf8=False):
        self._utf8 = utf8
        if d is not None:
            for k, v in d.iteritems(): self[k] = v

    def __setattr__(self, key, value):
        if key in self.__slots__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __setitem__(self, key, value):
        if self._utf8 and isinstance(key, str): key = key.encode('utf8')
        dict.__setitem__(self, key, rstorify(value, utf8=self._utf8))

    def __getattr__(self, key):
        if key in type(self).__slots__:
            return self.__dict__[key]
        else:
            try:
                return self[key]
            except KeyError as k:
                raise (AttributeError, k)

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise (AttributeError, k)

    def __repr__(self):
        return ('<%s ' % type(self).__name__) + dict.__repr__(self) + '>'


def rstorify(e):
    if type(e) is dict:
        return RStorage(d=e)
    elif type(e) in (list, tuple):
        return [rstorify(i) for i in e]
    else: return e


def jsonify(item, ignore_errors=False):
    '''
    Recursively construct a json representation of the argument ``item``.
    :param item:
    :return:
    '''
    if hasattr(item, 'json'):
        return item.json
    elif hasattr(item, 'tojson'):
        return item.tojson()
    elif isinstance(item, dict):
        return {str(k): jsonify(v, ignore_errors=ignore_errors) for k, v in item.items()}
    elif type(item) in (list, tuple):
        return [jsonify(e, ignore_errors=ignore_errors) for e in item]
    elif isinstance(item, (int, float, bool, str, type(None))):
        return item
    else:
        if not ignore_errors:
            raise TypeError('object of type "%s" is not jsonifiable: %s' % (type(item), repr(item)))
        else: return '%s (NOT JSONIFIABLE)' % str(item)


class LinearScale(object):
    '''
    Implementation of a linear mapping from one interval of real
    numbers [a,b] into another one [c,d] by linearly interpolating.

    Example:
        >>> scale = LinearScale((1, 2), (-2, 278))
        >>> scale(1.5)
        138.0
    '''
    def __init__(self, fromint, toint, strict=True):
        self._from = fromint
        self._to = toint
        self._fromrange = fromint[1] - fromint[0]
        self._torange = toint[1] - toint[0]
        self.strict = strict

    def _apply(self, value):
        if self.strict and not self._from[0] <= value <= self._from[1]:
            raise ValueError('value out of range [%s, %s], got %s' % (self._from[0], self._from[1], value))
        v = float((value-self._from[0])) / self._fromrange
        return v * self._torange + self._to[0]

    def __call__(self, value):
        return self._apply(value)


if __name__ == '__main__':
    d = edict({1:2,2:3})
    print(d.project(2))
