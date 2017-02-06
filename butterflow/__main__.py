#!/usr/bin/env python2
# -*- coding: utf-8 -*-


if __name__ == '__main__':
    import sys
    from butterflow.core import CommandLineInterface

    # load the entry point and run it, will return non-zero on failure
    sys.exit(CommandLineInterface.main())
