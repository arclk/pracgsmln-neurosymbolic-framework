#!/usr/bin/env python
import os
import platform
import shlex
import struct
import subprocess
import colored
import sys
from colored.colored import stylize
from dnutils.tools import ifnot, ifnone
from threading import RLock
import re


def bf(s):
    return colored.stylize(s, colored.attr('bold'))


def ljust(t, l, f):
    s = cleanstr(t)
    n = l - len(s)
    if n <= 0: return t
    return t + f * n


# ------------------------------------------------------------------------------
# parts of this file are taken from https://gist.github.com/jtriley/1108174
# and adapted to Python 3
# ------------------------------------------------------------------------------

def get_terminal_size():
    """ getTerminalSize()
     - get width and height of console.rst
     - works on linux,os x,windows,cygwin(windows)
     originally retrieved from:
     http://stackoverflow.com/questions/566746/how-to-get-console.rst-window-width-in-python
    """
    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os in ['Linux', 'Darwin'] or current_os.startswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)  # default value
    return tuple_xy


def _get_terminal_size_windows():
    try:
        from ctypes import windll, create_string_buffer
        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12
        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom,
             maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
            return sizex, sizey
    except:
        pass


def _get_terminal_size_tput():
    # get terminal width
    # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width-height-of-a-terminal-window
    try:
        cols = int(subprocess.check_call(shlex.split('tput cols')))
        rows = int(subprocess.check_call(shlex.split('tput lines')))
        return (cols, rows)
    except:
        pass


def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios
            cr = struct.unpack('hh',
                               fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            return cr
        except:
            pass

    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (os.environ['LINES'], os.environ['COLUMNS'])
        except:
            return None
    return int(cr[1]), int(cr[0])


def tty(stream):
    isatty = getattr(stream, 'isatty', None)
    return isatty and isatty()


def barstr(width, percent, color=None, inf=False):
    '''
    Returns the string representation of an ASCII 'progress bar'.

    :param width:       the maximum space of the bar in number of of characters
    :param percent:     the percentage of ``width`` that the bar will consume.
    :param color:       string specifying the color of the bar
    :param inf:         boolean determining whether the bar is supposed to be "infinite".
    :return:            the string representation of the progress bar.
    '''
    width = width - 13  # constant number of characters for the numbers
    if not inf:
        barw = int(round(width * percent))
        bar = ''.ljust(barw, '=')
        bar = bar.ljust(width, ' ')
    else:
        bar = infbarstr(width, int(percent))
    if color is not None:
        filler = '\u25A0'
        bar = bar.replace('=', filler)
        bar = stylize('[', colored.attr('bold')) + stylize(bar, colored.fg(color)) + stylize(']', colored.attr('bold'))
    else:
        bar = '[%s]' % bar
    if inf:
        return bar
    return '{0} {1: >7.3f} %'.format(bar, percent * 100.)


def infbarstr(width, pos):
    '''
    Returns the string representation of an ASCII 'progress bar'.
    :param width:
    :param pos:
    :return:
    '''
    bw = int(round(width * .2))
    bar = ' ' * pos
    bar += '=' * bw
    bar = bar[0:width]
    front = int(max(0, (pos + bw) - width))
    bar = ('=' * front) + bar[front:]
    bar = bar.ljust(width, ' ')
    return bar


class ProgressBar():
    '''
    An ASCII progress bar to show progress in the console.rst.
    '''

    def __init__(self, layout='100%:0%', value=0, steps=None, label='', color=None, stream=sys.stdout, inf=False):
        self.layout = layout
        self.setlayout(layout)
        self.steps = steps
        self.inf = inf
        self.lock = RLock()
        if inf:
            self.steps = self.barwidth - 13
            self.step = self.value = 0
        elif steps is not None:
            self.step = value
            self.value = float(value) / steps
        else:
            self.value = value
            self.step = None
            self.steps = None
        self.color = color
        self._label = label
        if tty(sys.stdout):
            self.update(self.value)

    def setlayout(self, layout):
        '''Specifies the layout of the progress bar.

        ``layout`` must be a string of the form "X:Y" or "X", where
        `X` determines the width of the bar part of the progress bar and
        `Y` determines the width of the label part of the progress bar.
        Values can be absolute (in console.rst characters) or relative (in percentage values)
        to the console.rst width.

        :example:

            >>> bar = ProgressBar(value=.2, color='green', label='in progress...please wait...')
            [■■■■■■■■■■■■■■■■■■■                                                                           ]  20.000 %
            >>> bar.setlayout('70%:30%')
            >>> print(bar)
            [■■■■■■■■■■■■                                                  ]  20.000 % in progress...please wait...
            >>> bar.setlayout('100%:0%')
            >>> print(bar)
            [■■■■■■■■■■■■■■■■■■■                                                                           ]  20.000 %
            >>> bar.setlayout('60:40')
            >>> print(bar)
            [■■■■■■■■■                                      ]  20.000 % in progress...please wait...

        '''
        if ':' in layout:
            barw, lblw = layout.split(':')
        else:
            barw, lblw = layout, ''
        if '%' in barw:
            barw = float(barw.strip('% ')) / 100.
        elif barw:
            barw = int(barw)
        else:
            barw = -1
        if '%' in lblw:
            lblw = float(lblw.strip('% ')) / 100.
        elif lblw:
            lblw = int(lblw)
        else:
            lblw = -1
        if barw == -1 and lblw == -1:
            raise AttributeError('Illegal layout specification: "%s"' % layout)
        termw, _ = get_terminal_size()
        if barw != -1:
            self.barwidth = barw if type(barw) is int else int(round((termw * barw)))
        if lblw != -1:
            self.lblwidth = lblw if type(lblw) is int else int(round((termw * lblw)))
        else:
            self.lblwidth = termw - self.barwidth
        if barw == -1:
            self.barwidth = termw - self.lblwidth

    def label(self, label):
        '''Set the current label of the bar.'''
        self._label = label
        self.setlayout(self.layout)
        self.update(self.value)

    def update(self, value, label=None):
        '''Set the current value of the bar to ``value`` and update the label by ``label``.'''
        self.setlayout(self.layout)
        self.value = value
        if label is not None: self._label = label
        if value == 1: self._label = ''
        if tty(sys.stdout):
            sys.stdout.write(str(self))
            sys.stdout.flush()

    def finish(self, erase=True, msg='', end='\n'):
        '''Terminates the progress bar.

        :param erase:    If ``True``, the progress bar will be removed (overwritten) from the console.rst.
        :param msg:      Optional "goodbye"-message to be printed.
        :param end:      Final character to be printed (default is '\\n' to move to a new line)
        '''
        if erase: sys.stdout.write('\r' + msg.ljust(self.lblwidth + self.barwidth, ' '))
        sys.stdout.write(end)

    def inc(self, steps=1):
        '''Increment the current value of the progress bar by ``steps`` steps.'''
        self.setlayout(self.layout)
        with self.lock:
            if self.steps is None:
                raise Exception('Cannot call inc() on a real-valued progress bar.')
            self.step += steps
            if not self.inf:
                value = float(self.step) / self.steps
                self.update(value)
            else:
                self.update(self.step)
                self.step %= (self.barwidth - 13)

    def __str__(self):
        return '\r' + barstr(self.barwidth, self.value, color=self.color, inf=self.inf) + ' ' + self._label[
                                                                                                :self.lblwidth].ljust(
            self.lblwidth, ' ')



ansi_escape = re.compile(r'\x1b[^m]*m')


def cleanstr(s):
    return ansi_escape.sub('', s)


class StatusMsg(object):
    '''Print a Linux-style status message to the console.rst.'''
    ERROR = colored.stylize('ERROR', (colored.fg('red'), colored.attr('bold')))
    FAILED = colored.stylize('FAILED', (colored.fg('red'), colored.attr('bold')))
    OK = colored.stylize('OK', (colored.fg('green'), colored.attr('bold')))
    WARNING = colored.stylize('WARNING', (colored.fg('yellow'), colored.attr('bold')))
    PASSED = colored.stylize('PASSED', (colored.fg('green'), colored.attr('bold')))

    def __init__(self, message='', status=None, width='100%', stati=None):
        if stati is None:
            self.stati = {StatusMsg.ERROR, StatusMsg.OK, StatusMsg.WARNING, StatusMsg.FAILED, StatusMsg.PASSED}
        else:
            self.stati = stati
        self.widthstr = width
        self.setwidth(self.widthstr)
        self.msg = message
        self.status = status
        self.write()

    def setwidth(self, width):
        '''
        Sets the with in relative or absolute numbers of console.rst characters.
        :param width:
        :return:
        '''
        if '%' in width:
            consolewidth, _ = get_terminal_size()
            self.width = int(round(consolewidth * float(width.strip('%')) * .01))
        else:
            self.width = int(width)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, s):
        if s not in self.stati and s is not None:
            raise ValueError('Status "%s" cannot be set.' % str(s))
        self._status = s
        self.write()

    def message(self, msg):
        self.msg = msg
        self.write()

    def write(self):
        self.setwidth(self.widthstr)
        statuswidth = max(map(len, [cleanstr(s) for s in self.stati]))
        lblwidth = self.width - statuswidth - 4
        msg = self.msg
        if lblwidth < len(cleanstr(self.msg)):
            msg = self.msg[:lblwidth - 4] + '...'
        sts = ifnone(self._status, '')
        s = ljust(msg, lblwidth - 1, ' ') + ' [ %s ]' % sts.center(statuswidth + (len(sts) - len(cleanstr(sts))), ' ')
        sys.stdout.write('\r' + s)

    def finish(self, erase=False, end='\n'):
        if erase: sys.stdout.write('\r' + ' ' * self.width)
        sys.stdout.write(end)


if __name__ == "__main__":
    sizex, sizey = get_terminal_size()
    print('width =', sizex, 'height =', sizey)
