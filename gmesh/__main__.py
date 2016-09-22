from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import namedtuple, OrderedDict
import importlib
import sys
import textwrap


commands = OrderedDict()
Command = namedtuple('Command', ['func', 'parser'])
def command():
    def decorator(fn):
        def func(args):
            args = commands[fn.__name__].parser.parse_args(args)
            fn(**vars(args))
        commands[fn.__name__] = Command(
            func=func,
            parser=ArgumentParser(
                'rungmesh {}'.format(fn.__name__),
                description=fn.__doc__,
                add_help=False,
            ),
        )
        return fn
    return decorator


def argument(*args, **kwargs):
    def decorator(fn):
        parser = commands[fn.__name__].parser
        parser.add_argument(*args, **kwargs)
        return fn
    return decorator


@argument('cmd', nargs='?', help='the command for which to show help')
@command()
def help(cmd=None):
    """Show help for commands."""
    if cmd in commands:
        return commands[cmd].parser.print_help()
    if cmd:
        print("{}: no such command available; try 'rungmesh help'".format(command))
        sys.exit(2)

    # Create a list of commands in the same style as default ArgumentParser
    # help output
    max_cmd_len = max(len(cmd) for cmd in commands)
    indent = max_cmd_len + 4
    fill_width = 80 - indent
    cmdlist = []
    for name, cmd in commands.items():
        if cmd.parser.description:
            (first, *rest) = cmd.parser.description.split('\n')
            rest = textwrap.dedent('\n'.join(rest))
            (first, *rest) = textwrap.wrap('\n'.join([first, rest]), width=fill_width)
        else:
            first, rest = '', []
        cmdlist.append('  {name:{width}}  {first}'.format(
            name=name, width=max_cmd_len, first=first
        ))
        cmdlist.extend(' '*indent + s for s in rest)

    # Construct a dummpy parser just to print the help
    # We do this to make help strings uniform
    parser = ArgumentParser(
        'rungmesh',
        description='Miscellaneous. Very, very, miscellaneous.',
        formatter_class=RawDescriptionHelpFormatter,
        add_help=False,
        epilog="available commands:\n" + '\n'.join(cmdlist),
    )
    parser.add_argument('cmd', help='The command to execute.')
    parser.print_help()


@argument('--zval', type=float, required=False)
@argument('--yval', type=float, required=False)
@argument('--xval', type=float, required=False)
@argument('--nz', type=int, default=10)
@argument('--ny', type=int, default=10)
@argument('--nx', type=int, default=10)
@argument('--out', type=str, required=True)
@argument('filename', type=str)
@command()
def structure(filename, out, nx, ny, nz, xval, yval, zval):
    """Turn an unstructured VTK into a structured one."""
    tools = importlib.import_module('gmesh.tools')
    tools.structure(filename, out, nx, ny, nz, xval, yval, zval,)


@command()
def map():
    """Show the map."""
    gui = importlib.import_module('gmesh.gui')
    gui.run()


def main():
    args = sys.argv[1:]

    if not args:
        return commands['help'].func(args)

    cmd, args = args[0], args[1:]
    if cmd in commands:
        cmd = commands[cmd]
        return cmd.func(args)
    print("{}: no such command available, try 'rungmesh help'".format(cmd),
          file=sys.stderr)
    sys.exit(2)


main()
