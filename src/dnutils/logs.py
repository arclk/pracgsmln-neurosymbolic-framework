import json
import logging
import os
import re
import sys
import tempfile

import atexit
import warnings

import colored

import datetime

from .tools import ifnone
from .debug import _caller
from .threads import RLock, interrupted, Lock
from .tools import jsonify

import portalocker
FLock = portalocker.Lock


DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


class FileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        logging.FileHandler.__init__(self, filename, mode=mode, encoding=encoding, delay=delay)
        self.timeformatstr = '%Y-%m-%d %H:%M:%S'

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def format(self, record):
        return '{} - {} - {}'.format(datetime.datetime.fromtimestamp(record.created).strftime(self.timeformatstr),
                                     record.levelname,
                                     ' '.join(' '.join(map(str, record.msg)).split('\n')))


StreamHandler = logging.StreamHandler


_expose_basedir = '.exposure'
_exposures = None
_writelockname = '.%s.lock'

_MAX_EXPOSURES = 9999

exposure_dir = None


def set_exposure_dir(d):
    global exposure_dir
    exposure_dir = d


def tmpdir():
    '''
    Returns the path for temporary files.

    On Unix systems, eg. mostly ``/tmp``
    :return:
    '''
    with tempfile.NamedTemporaryFile(delete=True) as f:
        return os.path.dirname(f.name)


class ExposureEmptyError(Exception): pass


class ExposureLockedError(Exception): pass


def active_exposures(name='/*'):
    '''
    Generates the names of all exposures that are currently active (system-wide).

    :param name:    a pattern that the list of exposure names can be filtered (supports the wildcard character *)
    :return:
    '''
    tmp = tmpdir()
    rootdir = ifnone(exposure_dir, tmp)
    rootdir = os.path.join(rootdir, _expose_basedir)
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            if re.match(r'\.\w+\.lock', f):  # skip file locks
                continue
            try:
                tmplock = FLock(os.path.join(root, _writelockname % f), timeout=0, fail_when_locked=True)
                tmplock.acquire()
            except portalocker.LockException:
                expname = '/'.join([root.replace(rootdir, ''), f])
                tokens = expname.split('/')
                patterns = name.split('/')
                ok = False
                for idx, pat in enumerate(patterns):
                    try:
                        repattern = '^%s$' % re.escape(pat).replace(r'\*', r'.*?')
                        ok = re.match(repattern, tokens[idx]) is not None
                    except IndexError:
                        ok = False
                    if not ok: break
                else:
                    if ok:
                        yield expname
            else:
                tmplock.release()


class ExposureManager:
    '''
    Manages all instances of exposures.
    '''

    def __init__(self, basedir=None):
        self.exposures = {}
        basedir = ifnone(basedir, tmpdir())
        self.basedir = os.path.join(basedir, _expose_basedir)
        atexit.register(_cleanup_exposures)
        self._lock = RLock()

    def _create(self, name):
        '''
        Create a new exposure with name ``name``.

        :param name:
        :return:
        '''
        e = Exposure(name, self.basedir)
        self.exposures[name] = e
        return e

    def get(self, name):
        with self._lock:
            if not name in self.exposures:
                self.exposures[name] = Exposure(name, self.basedir)
            return self.exposures.get(name)

    def delete(self):
        with self._lock:
            for name, exposure in self.exposures.items():
                exposure.delete()


def _cleanup_exposures(*_):
    _exposures.delete()


# def exposures(basedir='.'):
#     global _exposures
#     _exposures = ExposureManager(basedir)


def expose(name, *data, ignore_errors=False):
    '''
    Expose the data ``data`` under the exposure name ``name``.
    :param name:
    :param data:
    :return:
    '''
    global _exposures
    if _exposures is None:
        _exposures = ExposureManager(exposure_dir)
    e = _exposures.get(name)
    if data:
        if len(data) == 1:
            data = data[0]
        e.dump(data, ignore_errors=ignore_errors)
    return e.name


def inspect(name):
    '''
    Inspect the exposure with the name ``name``.
    :param name:
    :return:
    '''
    global _exposures
    if _exposures is None:
        _exposures = ExposureManager(exposure_dir)
    if name in _exposures.exposures:
        e = _exposures.exposures[name]
    else:
        e = _exposures.get(name)
    try:
        return e.load()
    except IOError:
        return None


def exposure(name):
    '''
    Get the exposure object with the given name.
    :param name:
    :return:
    '''
    global _exposures
    if _exposures is None:
        _exposures = ExposureManager(exposure_dir)
    e = _exposures.get(name)
    return e


class Exposure:
    '''
    This class implements a data structure for easy and lightweight exposure of
    parts of a program's state. An exposure is, in essence, a read/write
    wrapper around a regular file, which is being json data written to and read from.
    '''

    def __init__(self, name, basedir=None):
        self._lock = RLock()
        if sum([1 for c in name if c == '#']):
            raise ValueError('exposure name may contain maximally one hash symbol: "%s"' % name)
        self.flock = None
        self.counter = 0
        counter = 1
        while 1:
            name_ = name.replace('#', str(counter))
            self._init(name_, basedir)
            if not self.acquire(blocking=False):
                if '#' not in name or counter >= _MAX_EXPOSURES:
                    raise ExposureLockedError()
                counter += 1
            else:
                self.release()
                break

    def _init(self, name, basedir):
        if basedir is None:
            basedir = os.path.join(tmpdir(), _expose_basedir)
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        dirs = list(os.path.split(name))
        if not dirs[0].startswith('/'):
            raise ValueError('exposure names must start with "/"')
        else:
            dirs[0] = dirs[0].replace('/', '')
        fname = dirs[-1]
        fullpath = basedir
        for d in dirs[:-1]:
            fullpath = os.path.join(fullpath, d)
            if not os.path.exists(fullpath):
                os.mkdir(fullpath)
        self.abspath = os.path.abspath(fullpath)
        self.filepath = os.path.join(self.abspath, fname)
        self.filename = fname
        self.flockname = os.path.join(self.abspath, _writelockname % self.filename)
        # acquire the lock if write access is required
        self.flock = FLock(self.flockname, timeout=0, fail_when_locked=True)
        self.name = name

    def acquire(self, blocking=True, timeout=None):
        '''
        Acquire the exposure.

        An exposure may only be acquired by one process at a time and acts like a re-entrant lock.
        :param blocking:
        :param timeout:
        :return:
        '''
        with self._lock:
            if self.counter > 0:  # exposure can be re-entered
                self.counter += 1
                return True
            if not blocking:
                timeout = 0
            elif blocking and timeout is None:
                timeout = .5
            ret = None
            while ret is None and not interrupted():
                with warnings.catch_warnings():
                    try:
                        ret = self.flock.acquire(timeout, fail_when_locked=False)
                    except portalocker.LockException:
                        if not blocking: break
                    warnings.simplefilter("ignore")
            self.counter += 1
            return ret is not None

    def release(self):
        with self._lock:
            self.counter -= 1
            if self.counter == 0:
                self.flock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *args):
        self.release()

    def dump(self, item, ignore_errors=False):
        '''
        Write the item to the exposure.

        :param item:
        :return:
        '''
        with self._lock:
            jsondata = jsonify(item, ignore_errors=ignore_errors)
            gotit = self.acquire(blocking=False)
            if not gotit:
                raise ExposureLockedError()
            try:
                with open(self.filepath, 'w+') as f:
                    f.truncate(0)
                    f.seek(0)
                    json.dump(jsondata, f, indent=4)
                    f.write('\n')
                    f.flush()
            finally:
                self.release()

    def delete(self):
        '''
        Close this exposure.
        :return:
        '''
        with self._lock:
            try:
                # os.remove(self.filepath)
                os.remove(self.flockname)
            except FileNotFoundError: pass

    def load(self, block=1):
        '''
        Load the content exposed by this exposure.

        If ``block`` is ``True``, this methods blocks until the content of this exposure
        has been updated by the writer
        :return:
        '''
        with self._lock:
            with open(self.filepath, 'r') as f:
                f.seek(0)
                return json.load(f)


class _LoggerAdapter(object):
    def __init__(self, logger):
        self._logger = logger
        self._logger.findCaller = self._caller

    def _caller(self, *_):
        return _caller(4)

    def critical(self, *args, **kwargs):
        self._logger.critical(args, extra=kwargs)

    def exception(self, *args, **kwargs):
        self._logger.exception(args, extra=kwargs)

    def error(self, *args, **kwargs):
        self._logger.error(args, extra=kwargs)

    def warning(self, *args, **kwargs):
        self._logger.warning(args, extra=kwargs)

    def info(self, *args, **kwargs):
        self._logger.info(args, extra=kwargs)

    def debug(self, *args, **kwargs):
        self._logger.debug(args, extra=kwargs)

    def __getattr__(self, attr):
        return getattr(self._logger, attr)

    @property
    def level(self):
        return self._logger.level

    @level.setter
    def level(self, l):
        self._logger.setLevel(l)

    @property
    def name(self):
        return self._logger.name

    def add_handler(self, h):
        self._logger.addHandler(h)

    def rm_handler(self, h):
        self._logger.removeHandler(h)

    @property
    def handlers(self):
        return self._logger.handlers

    def new(self, name, level=None):
        '''
        Spawn a new logger with the given name and return it.

        The new logger will be a child logger of this logger, i.e. it will inherit all of its handlers and,
        if not specified by the level parameter, also the log level.

        :param name:
        :param level:
        :return:
        '''
        if level is None:
            level = self.level
        logger = logging.getLogger(name)
        logger.parent = self._logger
        logger._initialized = True
        logger.setLevel(level)
        return _LoggerAdapter(logger)

    def __str__(self):
        return '<LoggerAdapter name="%s", level=%s>' % (self.name, logging._levelToName[self.level])

def getloggers():
    with logging._lock:
        for name in logging.Logger.manager.loggerDict:
            yield getlogger(name)

def loglevel(level, name=None):
    if name is None:
        name = ''
    getlogger(name).level = level


ansi_escape = re.compile(r'\x1b[^m]*m')


def cleanstr(s):
    return ansi_escape.sub('', s)


class ColoredStreamHandler(logging.StreamHandler):
    def emit(self, record):
        self.stream.write(self.format(record))


colored_console = ColoredStreamHandler()


class ColoredFormatter(logging.Formatter):
    fmap = {
        logging.DEBUG: colored.fg('cyan') + colored.attr('bold'),
        logging.INFO: colored.fg('white') + colored.attr('bold'),
        logging.WARNING: colored.fg('yellow') + colored.attr('bold'),
        logging.ERROR: colored.fg('red') + colored.attr('bold'),
        logging.CRITICAL: colored.bg('dark_red_2') + colored.fg('white') + colored.attr('underlined') + colored.attr(
            'bold')
    }
    msgmap = {
        logging.DEBUG: colored.fg('cyan'),
        logging.INFO: colored.fg('white'),
        logging.WARNING: colored.fg('yellow'),
        logging.ERROR: colored.fg('red'),
        logging.CRITICAL: colored.fg('dark_red_2')
    }

    def __init__(self, formatstr=None):
        self.formatstr = formatstr

    def format(self, record):
        levelstr = colored.attr('reset')
        levelstr += ColoredFormatter.fmap[record.levelno]
        maxlen = max(map(len, logging._levelToName.values()))
        header = '%s - %s - ' % (datetime.datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                                 colored.stylize(record.levelname.center(maxlen, ' '), levelstr))
        return header + colored.stylize(('\n' + ' ' * len(cleanstr(header))).join(' '.join(map(str, record.msg)).split('\n')) + '\n',
                                        ColoredFormatter.msgmap[record.levelno])


colored_console.setFormatter(ColoredFormatter())

try:
    import pymongo
except ImportError:
    pass
else:
    class MongoHandler(logging.Handler):
        '''
        Log handler for logging into a MongoDB database.
        '''
        def __init__(self, collection, checkkeys=True):
            '''
            Create the handler.

            :param collection:  An accessible collection in a pymongo database.
            '''
            logging.Handler.__init__(self)
            self.checkkeys = checkkeys
            self.coll = collection
            self.setFormatter(MongoFormatter())

        def emit(self, record):
            try:
                self.coll.insert(self.format(record), check_keys=self.checkkeys)
            except pymongo.errors.ServerSelectionTimeoutError:
                sys.stderr.write('WARNING: Could not establish connection to mongo client to write log. Message:\n'
                                 '{} - {} - {}\n'.format(datetime.datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                                                         record.levelname,
                                                         ' '.join([str(s) for s in record.msg])))


    class MongoFormatter(logging.Formatter):

        def format(self, record):
            return {'message': record.msg , 'timestamp': datetime.datetime.utcfromtimestamp(record.created),
                    'module': record.module, 'lineno': record.lineno, 'name': record.name, 'level': record.levelname}


class LoggerConfig(object):
    '''
    Data structure for storing a configuration of a particular
    logger, such as its name and the handlers to be used.
    '''
    def __init__(self, level, *handlers):
        self.handlers = handlers
        self.level = level


def newlogger(*handlers, level=INFO):
    '''
    Create a new logger configuration.

    Takes a list of handlers and optionally a level specification.

    Example:
    >>> dnlog.newlogger(logging.StreamHandler(), level=ERROR)

    :param handlers:
    :param kwargs:
    :return:
    '''
    return LoggerConfig(level, *handlers)


def loggers(loggers=None):
    '''
    Initial setup for the logging of the current process.

    The root logger is identified equivalently by None or 'default'. If no specification for the root logger
    is provided, a standard console.rst handler will be automatically appended.

    :param loggers: a dictionary mapping the names of loggers to :class:`dnlog.LoggerConfig` instances.
    :return:
    '''
    if loggers is None:
        loggers = {}
    if not {None, 'default'} & set(loggers.keys()):
        loggers['default'] = newlogger(console)
    for name, config in loggers.items():
        logger = getlogger(name)
        for h in logger.handlers:
            logger.removeHandler(h)
        logger.level = config.level
        for handler in config.handlers:
            logger.add_handler(handler)
            logger._logger._initialized = True


def getlogger(name=None, level=None):
    '''
    Get the logger with the associated name.

    If name is None, the root logger is returned. Optionally, a level can be specified that the logger is autmatically
    set to.

    :param name:    the name of the desired logger
    :param level:   the log level
    :return:
    '''
    if name == 'default':
        name = None
    logger = logging.getLogger(name)
    defaultlevel = logging.getLogger().level
    adapter = _LoggerAdapter(logger)
    if not hasattr(logger, '_initialized') or not logger._initialized:
        logger.parent = None
        roothandlers = list(logging.getLogger().handlers)
        # clear all loggers first
        for h in logger.handlers:
            logger.removeHandler(h)
        # take default handlers from the root logger
        for h in roothandlers:
            adapter.add_handler(h)
        adapter.level = ifnone(level, defaultlevel)
        logger._initialized = True
    if level is not None:
        adapter.level = level
    return adapter


console = colored_console
loggers()


if __name__ == '__main__':

    # for i in range(10):

    expose('/vars/bufsize', 'hello')
    expose('/internal/state', 1)
    for ex in active_exposures():
        expose('/vars/bufsize', 'bla')
        # sleep(10)
        print(ex, inspect(ex))
        # portalocker.lock(f2, portalocker.LOCK_EX, timeout=0)

        # try:
        #     print(inspect('/vars/bufsize'))
        # except ExposureEmptyError:
        #     sys.exit(0)
        # sleep(5)

