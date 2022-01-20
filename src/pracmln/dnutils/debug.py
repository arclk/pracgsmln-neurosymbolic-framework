'''
Created on May 22, 2017

@author: nyga
'''
import traceback
import sys
import os
from dnutils.tools import edict


def _currentframe():
    '''Return the frame object for the caller's stack frame.'''
    try:
        raise Exception()
    except:
        traceback.print_exc()
        return sys.exc_info()[2].tb_frame

if hasattr(sys, '_getframe'): 
    _currentframe = lambda: sys._getframe(2)


def _caller(tb=1):
    '''Find the stack frame of the caller so that we can note the source
    file name, line number and function name.
    
    :param tb:     The depth of the traceback. The default, `tb=1`, returns the
                   immediate caller, `tb=2` returns the caller of the caller, and
                   so on. This is useful for implementing more complex debug
                   output functions whose callers are deeper in the call stack.  
    '''
    f = _currentframe()
    #On some versions of IronPython, currentframe() returns None if
    #IronPython isn't run with -X:Frames.
    rv = "(unknown file)", 0, "(unknown function)"
    d = 0
    while hasattr(f, "f_code"):
        co = f.f_code
        rv = (co.co_filename, f.f_lineno, co.co_name)
        if d >= tb: break
        d += 1
        f = f.f_back
    return rv


def out(*args, file=sys.stdout, sep=' ', end='\n', flush=False, tb=1):
    '''Basic output function that prints a str-converted list of its arguments.
    
    `out` forwards all arguments to the ordinary `print` function, but appends 
    the file and line of its call, so it can be found easier from the console.rst output.
    
    :param file:    a file-like object (stream); defaults to the current sys.stdout.
    :param sep:     string inserted between values, default a space.
    :param end:     string appended after the last value, default a newline.
    :param flush:   whether to forcibly flush the stream.
    
    The keyword arguments are inherited from Python's :func:`print` function. 
    There is an additional keyword argument, ``tb``, which determines the depth of the
    calling frame. It is not passed to :func:`print`.'''
    rv = _caller(tb)
    print('{}: l.{}: {}'.format(os.path.basename(rv[0]), rv[1], args[0] if args else ''), 
          *(args[1:] if len(args) > 1 else []), file=file, sep=sep, end=end, flush=flush)


def stop(*args, file=sys.stdout, sep=' ', end='\n', flush=False, tb=1):
    '''Same as :func:`dnutils.debug.out`, but stops with a promt after having printed 
    the respective arguments until `<enter>` is pressed.'''
    out(*args, file=file, sep=sep, end=end, flush=flush, tb=tb+1)
    input('<press enter to continue>')
    

def trace(*args, **kwargs):
    '''Prints a stack trace of the current frame and terminates with
    a call of :func:`dnutils.debug.out` of the given arguments.'''
    print('=== STACK TRACE ===')
    sys.stdout.flush()
    traceback.print_stack(file=kwargs.get('file', sys.stdout))
    out(*args, **edict(kwargs) + {'tb': kwargs.get('tb', 1) + 1, 'flush': True})
    
    
def stoptrace(*args, **kwargs):
    '''Same as :func:`dnutils.trace`, but stops with a prompt after 
    having printed the stack trace.'''
    trace(**edict(kwargs) + {'tb': kwargs.get('tb', 1) + 1})
    stop(*args, **edict(kwargs) + {'tb': kwargs.get('tb', 1) + 1})