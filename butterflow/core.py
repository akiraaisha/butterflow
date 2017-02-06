# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
from cv2 import calcOpticalFlowFarneback as sw_farneback_optical_flow
import avinfo
import motion
import ocl
import settings
import logging
import datetime
import argparse
import re
import collections
import inspect
import multiprocessing
from itertools import izip
import signal
import subprocess
import shutil
import timeit


# regex patterns
flt_pattern = r"(?P<flt>\d*\.\d+|\d+)"      # floats, and ints
wh_pattern = re.compile(r"""                # width & height
(?=(?P<semicolon>.+:.+)|.+)
(?(semicolon)
  (?P<width>-1|\d+):(?P<height>-1|\d+)|
  {}
)
(?<!^-1:-1$)  # ignore -1:-1
""".format(flt_pattern), re.X)
sl_pattern = r"(?=(?P<slash>.+/.+)|.+)"    # slashes (for fractional inputs)
# numerators & denominators
nd_pattern = r"(?P<numerator>\d*\.\d+|\d+)/(?P<denominator>\d*\.\d+|\d+)"
pr_pattern = re.compile(r"""               # playback rate
{}
(?(slash)
  {}|
  (?P<flt_or_x>\d*\.\d+x?|\d+x?)
)
""".format(sl_pattern, nd_pattern), re.X)
# time as hrs:mins:secs.xxx, mins:secs.xxx, secs.xxx
tm_pattern = r"""^
(?:
  (?:([01]?\d|2[0-3]):)?
  ([0-5]?\d):
)?
(\.\d{1,3}|[0-5]?\d(?:\.\d{1,3})?)$
"""
sr_tm_pattern = tm_pattern[1:-2]          # remove "^$", subregion time format
sr_end_pattern = r"end"                   # keyword: end
sr_ful_pattern = r"full"                  # keyword: full
# the complete subregion format, with times, targets, and keywords:
sr_pattern = re.compile(r"""^
a=(?P<tm_a>{tm}),
b=(?P<tm_b>{tm}),
(?P<target>fps|dur|spd)=
{}
(?P<val>
  (?(slash)
    {}|
    {}
  )
)$
""".format(sl_pattern, nd_pattern, flt_pattern, tm=sr_tm_pattern), re.X)


logging.basicConfig(level=settings.default['loglevel'],
                    format='[butterflow:%(levelname)s]: %(message)s')
log = logging.getLogger('butterflow')


class Display(object):  # TODO: support multiple displays
    window_title = 'Display'

    @staticmethod
    def show(A, wait=settings.default['imshow_ms']):
        if type(A) == np.ndarray:   # np.ndarray is a low-level array repr
            # "Arrays should be constructed using np.array, np.zeros or empty"
            A = np.asarray(A)
        cv2.imshow(Display.window_title, A)     # expects an np.array
        cv2.waitKey(wait)                       # ms-to wait, 0=until key pressed

    @staticmethod
    def destroy():
        cv2.destroyAllWindows()


class CommandLineInterface(object):
    """The main entry point, called in setup.py"""
    @staticmethod
    def main():
        par = argparse.ArgumentParser(usage='butterflow [options] [video]',
                                      add_help=False)
        req = par.add_argument_group('Required arguments')
        gen = par.add_argument_group('General options')
        dev = par.add_argument_group('Device options')
        dsp = par.add_argument_group('Display options')
        vid = par.add_argument_group('Video options')
        mix = par.add_argument_group('Mixing options')
        fgr = par.add_argument_group('Advanced options')

        req.add_argument('video', type=str, nargs='?', default=None,
                         help='Specify the input video')

        gen.add_argument('-h', '--help', action='help',
                         help='Show this help message and exit')
        gen.add_argument('--version', action='store_true',
                         help='Show program\'s version number and exit')
        gen.add_argument('-c', '--cache', action='store_true',
                         help='Show cache information and exit')
        gen.add_argument('--rm-cache', action='store_true',
                         help='Set to clear the cache and exit')
        gen.add_argument('-prb', '--probe', action='store_true',
                         help='Show media file information and exit')
        gen.add_argument('-v', '--verbosity', action='count',
                         help='Set to increase output verbosity')
        gen.add_argument('-q', '--quiet', action='store_true',
                         help='Set to suppress console output')

        dev.add_argument('-d', '--show-devices', action='store_true',
                         help='Show detected OpenCL devices and exit. The '
                         'currently selected device is marked with a star '
                         '`*`.')
        dev.add_argument('-device', type=int,
                         default=-1,
                         help='Specify the OpenCL device to use as an integer.'
                         ' Device numbers can be listed with the `-d` option. '
                         'The device will be chosen automatically if nothing is '
                         'specified.')
        dev.add_argument('-sw', action='store_true',
                         help='Set to force software rendering')

        dsp.add_argument('-p', '--show-preview', action='store_true',
                         help='Set to show video preview')
        dsp.add_argument('-e', '--embed-info', action='store_true',
                         help='Set to embed debugging info into the output '
                              'video')
        dsp.add_argument('-tt', '--text-type',
                         choices=['light', 'dark', 'stroke'],
                         default=settings.debugtext['text_type'],
                         help='Specify text type for embedded debugging info, '
                         '(default: %(default)s)')
        dsp.add_argument('-m', '--mark-frames', action='store_true',
                         help='Set to mark interpolated frames with a red frame '
                              'marker')

        vid.add_argument('-o', '--output-path', type=str,
                         default=settings.default['out_path'],
                         help='Specify path to the output video')
        vid.add_argument('-r', '--playback-rate', type=str,
                         help='Specify the playback rate as an integer or a float.'
                         ' Fractional forms are acceptable, e.g., 24/1.001 is the '
                         'same as 23.976. To use a multiple of the source '
                         'video\'s rate, follow a number with `x`, e.g., "2x" '
                         'will double the frame rate. The original rate will be '
                         'used by default if nothing is specified.')
        vid.add_argument('-s', '--subregions', type=str,
                         help='Specify rendering subregions in the form: '
                         '"a=TIME,b=TIME,TARGET=VALUE" where TARGET is either '
                         '`spd`, `dur`, `fps`. Valid TIME syntaxes are [hr:m:s], '
                         '[m:s], [s], [s.xxx], or `end`, which signifies to the '
                         'end the video. You can specify multiple subregions by '
                         'separating them with a colon `:`. A special subregion '
                         'format that conveniently describes the entire clip is '
                         'available in the form: "full,TARGET=VALUE".')
        vid.add_argument('-k', '--keep-subregions', action='store_true',
                         help='Set to render subregions that are not explicitly '
                              'specified')
        vid.add_argument('-vs', '--video-scale', type=str,
                         default=str(settings.default['video_scale']),
                         help='Specify output video size in the form: '
                         '"WIDTH:HEIGHT" or by using a factor. To keep the '
                         'aspect ratio only specify one component, either width '
                         'or height, and set the other component to -1, '
                         '(default: %(default)s)')
        vid.add_argument('-l', '--lossless', action='store_true',
                         help='Set to use lossless encoding settings')
        vid.add_argument('-sm', '--smooth-motion', action='store_true',
                         help='Set to tune for smooth motion. This mode yields '
                         'artifact-less frames by emphasizing blended frames over '
                         'warping pixels.')

        mix.add_argument('-a', '--audio', action='store_true',
                         help='Set to add the source audio to the output video')

        fgr.add_argument('--fast-pyr', action='store_true',
                         help='Set to use fast pyramids')
        fgr.add_argument('--pyr-scale', type=float,
                         default=settings.default['pyr_scale'],
                         help='Specify pyramid scale factor, '
                         '(default: %(default)s)')
        fgr.add_argument('--levels', type=int,
                         default=settings.default['levels'],
                         help='Specify number of pyramid layers, '
                         '(default: %(default)s)')
        fgr.add_argument('--winsize', type=int,
                         default=settings.default['winsize'],
                         help='Specify averaging window size, '
                         '(default: %(default)s)')
        fgr.add_argument('--iters', type=int,
                         default=settings.default['iters'],
                         help='Specify number of iterations at each pyramid '
                         'level, (default: %(default)s)')
        fgr.add_argument('--poly-n', type=int,
                         choices=settings.default['poly_n_choices'],
                         default=settings.default['poly_n'],
                         help='Specify size of pixel neighborhood, '
                         '(default: %(default)s)')
        fgr.add_argument('--poly-s', type=float,
                         default=settings.default['poly_s'],
                         help='Specify standard deviation to smooth derivatives, '
                         '(default: %(default)s)')
        fgr.add_argument('-ff', '--flow-filter', choices=['box', 'gaussian'],
                         default=settings.default['flow_filter'],
                         help='Specify which filter to use for optical flow '
                         'estimation, (default: %(default)s)')

        # preprocess args
        # fixup args that start with a dash, needed for the video scaling option
        for i, arg in enumerate(sys.argv):
            if arg[0] == '-' and arg[1].isdigit():
                sys.argv[i] = ' '+arg

        args = par.parse_args()

        if args.verbosity == 1:
            log.setLevel(settings.default['loglevel_1'])
        if args.verbosity >= 2:
            log.setLevel(settings.default['loglevel_2'])
        if args.quiet:
            log.setLevel(settings.default['loglevel_quiet'])
            settings.default['quiet'] = True

        if args.version:
            print(settings.default['version'])
            return 0

        cachedir = settings.default['tempdir']

        cachedirs = []
        tempfolder = os.path.dirname(cachedir)
        for dirpath, dirnames, filenames in os.walk(tempfolder):
            for d in dirnames:
                if 'butterflow' in d:
                    if 'butterflow-'+settings.default['version'] not in d:
                        cachedirs.append(os.path.join(dirpath, d))
            break

        if args.cache:
            nfiles = 0
            sz = 0
            for dirpath, dirnames, filenames in os.walk(cachedir):
                if dirpath == settings.default['clbdir']:
                    continue
                for filename in filenames:
                    nfiles += 1
                    fp = os.path.join(dirpath, filename)
                    sz += os.path.getsize(fp)
            sz = sz / 1024.0**2
            print('{} files, {:.2f} MB'.format(nfiles, sz))
            print('Cache: '+cachedir)
            return 0
        if args.rm_cache:
            cachedirs.append(cachedir)
            for i, x in enumerate(cachedirs):
                print('[{}] {}'.format(i, x))
            choice = raw_input('Remove these directories? [y/N] ')
            if choice != 'y':
                print('Leaving the cache alone, done.')
                return 0
            for x in cachedirs:
                if os.path.exists(x):
                    shutil.rmtree(x)
            print('Cache deleted, done.')
            return 0

        if args.show_devices:
            ocl.print_ocl_devices()
            return 0

        if not args.video:
            print('No file specified')
            return 1
        elif not os.path.exists(args.video):
            print('File doesn\'t exist')
            return 1

        if args.probe:
            avinfo.print_av_info(args.video)
            return 0

        av_info = avinfo.get_av_info(args.video)
        if av_info['frames'] == 0:
            print('Bad file with 0 frames')
            return 1

        extension = os.path.splitext(os.path.basename(
                                                  args.output_path))[1].lower()
        if extension[1:] != settings.default['v_container']:
            print('Bad output file extension. Must be {}.'.format(
                  settings.default['v_container'].upper()))
            return 0

        compat_dev_available = ocl.compat_ocl_device_available()

        if not compat_dev_available and not args.sw:
            print('No compatible OpenCL devices were detected.\n'
                  'Must force software rendering with the `-sw` flag to continue.')
            return 1

        log.info('Version '+settings.default['version'])

        for x in cachedirs:
            log.warn('Stale cache directory (delete with `--rm-cache`): %s' % x)

        if ocl.compat_ocl_device_available():
            log.info('At least one compatible OpenCL device was detected')
        else:
            log.warning('No compatible OpenCL devices were detected.')

        if args.device != -1:
            try:
                # TODO: save preferred device setting
                # TODO: set a different device with the same vendorid
                ocl.select_ocl_device(args.device)
            except (IndexError, RuntimeError) as error:
                print('Error: '+str(error))
                return 1
            except ValueError:
                if not args.sw:
                    print('An incompatible device was selected.\n'
                          'Must force software rendering with the `-sw` flag to '
                          'continue.')
                    return 1

        s = "Using device: %s"
        if args.device == -1:
            s += " (autoselected)"
        log.info(s % ocl.get_current_ocl_device_name())

        use_sw_interpolate = args.sw

        if args.flow_filter == 'gaussian':
            args.flow_filter = cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        else:
            args.flow_filter = 0
        if args.smooth_motion:
            args.poly_s = 0.01

        def optflow_fn(x, y,
                       pyr=args.pyr_scale, levels=args.levels,
                       winsize=args.winsize, iters=args.iters, polyn=args.poly_n,
                       polys=args.poly_s, fast=args.fast_pyr,
                       filt=args.flow_filter):
            fu = None
            fv = None
            if use_sw_interpolate:
                # this fn will actually use ocl automatically if the device is
                # compatible
                flow = sw_farneback_optical_flow(x, y,
                                pyr, levels, winsize, iters, polyn, polys, filt)
                # split the flows now, so we don't have to do it later
                fu = flow[:,:,0]
                fv = flow[:,:,1]
            else:
                fu, fv = motion.ocl_farneback_optical_flow(x, y,
                          pyr, levels, winsize, iters, polyn, polys, fast, filt)
            return fu, fv

        interpolate_fn = None
        if use_sw_interpolate:
            interpolate_fn = SwFrameInterpolater().sw_interpolate_flow
            log.warn('Hardware acceleration is disabled. Rendering will be slow. '
                     'Do Ctrl+c to quit or suspend the process with Ctrl+z and '
                     'then stop it with `kill %1`, etc. You can list suspended '
                     'processes with `jobs`.)')
        else:
            interpolate_fn = motion.ocl_interpolate_flow
            log.info('Hardware acceleration is enabled')

        # get dimensions
        w1, h1 = av_info['w'], av_info['h']
        w2, h2 = -1, -1
        s = args.video_scale.strip()
        if s == 1.0:
            w2 = w1
            h2 = h1
        else:
            m = re.match(wh_pattern, s)
            if m:
                if m.groupdict()['semicolon']:
                    w2 = int(m.groupdict()['width'])
                    h2 = int(m.groupdict()['height'])
                    if w2 == -1:
                        w2 = int(h2*w1/h1)
                    if h2 == -1:
                        h2 = int(w2*h1/w1)
                else:
                    flt = float(m.groupdict()['flt'])
                    w2 = int(w1*flt)
                    h2 = int(h1*flt)
            else:
                print('Error: Unknown W:H syntax: {}'.format(s))
                return 1

        def nearest_even_int(x, tag=""):
            new_x = x & ~1
            if x != new_x:
                log.warn("%s: %d is not divisible by 2, setting to %d",
                         tag, x, new_x)
            return new_x

        w2 = nearest_even_int(w2, "W")
        if w2 > 256:
            if w2 % 4 > 0:
                old_w2 = w2
                w2 -= 2
                w2 = max(w2, 0)
                log.warn('W: %d > 256 but is not divisible by 4, setting to %d',
                         old_w2, w2)
        h2 = nearest_even_int(h2, "H")

        if   w1*h1 > w2*h2:
            scaling_method = settings.default['scaler_dn']
        elif w1*h1 < w2*h2:
            scaling_method = settings.default['scaler_up']
        else:
            scaling_method = None

        # get playback rate
        r1 = av_info['rate']
        r2 = -1
        s = args.playback_rate
        if not s:
            r2 = r1
        else:
            m = re.match(pr_pattern, s)
            if m:
                if m.groupdict()['slash']:
                    r2 = (float(m.groupdict()['numerator']) /
                          float(m.groupdict()['denominator']))
                flt_or_x = m.groupdict()['flt_or_x']
                if 'x' in flt_or_x:
                    r2 = float(flt_or_x[:-1]) * r1
                else:
                    r2 = float(flt_or_x)
            else:
                print('Error: Unknown playback rate syntax: %s' % s)
                return 1

        try:
            sequence = VideoSequence.sequence_from_string(args.subregions,
                                                          av_info['duration'],
                                                          av_info['frames'])
        except (ValueError, AttributeError) as error:
            print('Error: '+str(error))
            return 1

        rnd = Renderer(args.video,
                       args.output_path,
                       sequence,
                       r2,
                       optflow_fn,
                       interpolate_fn,
                       w2,
                       h2,
                       scaling_method,
                       args.lossless,
                       args.keep_subregions,
                       args.show_preview,
                       args.embed_info,
                       args.text_type,
                       args.mark_frames,
                       args.audio)

        log.info('Rendering:')
        added_rate = False
        for x in str(rnd.sequence).split('\n'):
            x = x.strip()
            if not added_rate:
                x += ', Rate={}'.format(av_info['rate'])
                log.info(x)
                added_rate = True
                continue
            if not args.keep_subregions and 'autogenerated' in x:
                log.info(x[:-1]+', will skip when rendering)')
                continue
            log.info(x)

        temp_subs = rnd.sequence.subregions
        for x in rnd.sequence.subregions:
            overlaps = False
            for y in temp_subs:
                if x is y:
                    continue
                elif x.intersects(y):
                    overlaps = True
                    break
            if overlaps:
                log.warn('At least 1 subregion overlaps with another')
                break

        success = True
        total_time = 0
        try:
            total_time = timeit.timeit(rnd.render,
                                       setup='import gc;gc.enable()',
                                       number=1)
        except (KeyboardInterrupt, SystemExit):
            success = False
        if success:
            log_function = log.info
            if rnd.frs_written > rnd.frs_to_render:
                log_function = log.warn
                log.warn('Unexpected write ratio')
            log_function('Write ratio: {}/{}, ({:.2f}%)'.format(
                         rnd.frs_written,
                         rnd.frs_to_render,
                         rnd.frs_written*100.0/rnd.frs_to_render))
            txt = 'Final output frames: {} source, +{} interpolated, +{} duped, '\
                  '-{} dropped'
            log.info(txt.format(rnd.source_frs,
                                rnd.frs_interpolated,
                                rnd.frs_duped,
                                rnd.frs_dropped))
            old_sz = os.path.getsize(args.video) / 1024.0
            new_sz = os.path.getsize(args.output_path) / 1024.0
            log.info('Output file size:\t{:.2f} kB ({:.2f} kB)'.format(
                     new_sz, new_sz-old_sz))
            log.info('Rendering took {:.3g} mins, done.'.format(
                     total_time/60))
            return 0
        else:
            log.warn('Quit unexpectedly')
            log.warn('Files were left in the cache @ '+
                     settings.default['tempdir']+'.')
            return 1


class StringTools(object):
    @staticmethod
    def time_string_to_milliseconds(s):
        """Returns the time, in millisconds, from an input string with syntax:
        [hrs:mins:secs.xxx], [mins:secs.xxx], or [secs.xxx]"""
        hrs         = 0
        mins        = 0
        secs        = 0
        split = s.strip().split(':')
        n = len(split)
        if n >= 1 and split[-1] != '':
            secs = float(split[-1])
        if n >= 2 and split[-2] != '':
            mins = float(split[-2])
        if n == 3 and split[-3] != '':
            hrs = float(split[-3])
        return (hrs*3600 + mins*60 + secs) * 1000.0


class VideoWriter(object):
    def __init__(self):
        self.p = None

    def open(self, dest, w, h, rate, lossless=False):
        vf = []
        vf.append('format=yuv420p')

        call = [settings.default['avutil'],
                '-loglevel',    settings.default['av_loglevel'],
                '-y',
                '-threads',     '0',
                '-f',           'rawvideo',
                '-pix_fmt',     'bgr24',
                '-s',           '{}x{}'.format(w, h),
                '-r',           str(rate),
                '-i',           '-',
                '-map_metadata', '-1',
                '-map_chapters', '-1',
                '-vf',          ','.join(vf),
                '-r',           str(rate),
                '-an',
                '-sn',
                '-c:v',         settings.default['cv'],
                '-preset',      settings.default['preset']]
        if settings.default['cv'] == 'libx264':
            quality = ['-crf',  str(settings.default['crf'])]
            if lossless:
                quality = ['-qp', '0']
            call.extend(quality)
            call.extend(['-level', '4.2'])
        params = []
        call.extend(['-{}-params'.format(
                     settings.default['cv'].replace('lib', ''))])
        params.append('log-level={}'.format(settings.default['enc_loglevel']))
        if settings.default['cv'] == 'libx265':
            quality = 'crf={}'.format(settings.default['crf'])
            if lossless:
                quality = 'lossless=1'  # Bug: https://trac.ffmpeg.org/ticket/4284
            params.append(quality)
        if len(params) > 0:
            call.extend([':'.join(params)])
        call.extend([dest])

        log.info('[Subprocess] Opening a pipe to the video writer')
        log.debug('Call: {}'.format(' '.join(call)))
        self.p = subprocess.Popen(call, stdin=subprocess.PIPE)
        if self.p == 1:
            raise RuntimeError

    def write(self, fr):
        self.p.stdin.write(bytes(fr.data))

    def close(self):
        if self.p and not self.p.stdin.closed:
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait()
            log.info('[Subprocess] Closing the pipe to the video writer')

    def __del__(self):
        self.close()


class AudioVideoHelper(object):
    @staticmethod
    def combine_av(v, a, dest):
        tempfile = '{}+{}.{}.{}'.format(os.path.splitext(os.path.basename(v))[0],
                                        os.path.splitext(os.path.basename(a))[0],
                                        os.getpid(),
                                        settings.default['v_container'])
        tempfile = os.path.join(settings.default['tempdir'], tempfile)

        call = [settings.default['avutil'],
                '-loglevel',     settings.default['av_loglevel'],
                '-y',
                '-i',            v,
                '-i',            a,
                '-c',            'copy',
                tempfile]

        log.info('[Subprocess] Combinining audio & video')
        log.debug('Call: {}'.format(' '.join(call)))
        if subprocess.call(call) == 1:
            raise RuntimeError
        log.info("Moving:\t%s -> %s", os.path.basename(tempfile), dest)
        shutil.move(tempfile, dest)

    @staticmethod
    def concat_av_files(dest, files):
        tempfile = os.path.join(settings.default['tempdir'],
                                'list.{}.txt'.format(os.getpid()))
        log.info("Writing list file:\t{}".format(os.path.basename(tempfile)))
        with open(tempfile, 'w') as f:
            for file in files:
                if sys.platform.startswith('win'):
                    file = file.replace('/', '//')
                f.write('file \'{}\'\n'.format(file))

        call = [settings.default['avutil'],
                '-loglevel',     settings.default['av_loglevel'],
                '-y',
                '-f',            'concat',
                '-safe',         '0',
                '-i',            tempfile,
                '-c',            'copy',
                dest]

        log.info('[Subprocess] Concatenating audio files')
        log.debug('Call: {}'.format(' '.join(call)))
        if subprocess.call(call) == 1:
            raise RuntimeError
        log.info("Delete:\t%s", os.path.basename(tempfile))
        os.remove(tempfile)

    @staticmethod
    def get_atempo_chain(s):
        if s >= 0.5 and s <= 2.0:
            return [s]
        def solve(s, limit):
            vals = []
            x = int(np.log(s) / np.log(limit))
            for i in range(x):
                vals.append(limit)
            y = float(s) / np.power(limit, x)
            vals.append(y)
            return vals
        if s < 0.5:
            return solve(s, 0.5)
        else:
            return solve(s, 2.0)

    @staticmethod
    def extract_audio(v, dest, ss, to, speed=1.0):
        filename = os.path.splitext(os.path.basename(dest))[0]
        tempfile1 = os.path.join(settings.default['tempdir'],
              '{}.{}'.format(filename, settings.default['v_container']).lower())

        call = [settings.default['avutil'],
                '-loglevel',     settings.default['av_loglevel'],
                '-y',
                '-i',            v,
                '-ss',           str(ss/1000.0),
                '-to',           str(to/1000.0),
                '-map_metadata', '-1',
                '-map_chapters', '-1',
                '-vn',
                '-sn']
        if settings.default['ca'] == 'aac':
            call.extend(['-strict', '-2'])
        call.extend([
                '-c:a',          settings.default['ca'],
                '-b:a',          settings.default['ba'],
                tempfile1])

        log.info('[Subprocess] Audio chunk extraction')
        log.debug('Call: {}'.format(' '.join(call)))
        log.info("Extracting to:\t%s", os.path.basename(tempfile1))
        if subprocess.call(call) == 1:
            raise RuntimeError

        tempfile2 = os.path.join(settings.default['tempdir'],
                                 '{}.{}x.{}'.format(filename, speed,
                                               settings.default['a_container']))

        atempo_chain = AudioVideoHelper.get_atempo_chain(speed)
        chain_string = ""
        chain = []
        for i, tempo in enumerate(atempo_chain):
            chain.append('atempo={}'.format(tempo))
            chain_string += str(tempo)
            if i < len(atempo_chain)-1:
                chain_string += "*"
        log.info("Solved tempo chain for speed ({}x): {}".format(speed,
                                                                 chain_string))

        call = [settings.default['avutil'],
                '-loglevel',     settings.default['av_loglevel'],
                '-y',
                '-i',            tempfile1,
                '-filter:a',     ','.join(chain)]
        if settings.default['ca'] == 'aac':
            call.extend(['-strict', '-2'])
        call.extend(['-c:a',     settings.default['ca'],
                     '-b:a',     settings.default['ba'],
                     tempfile2])

        log.info('[Subprocess] Altering audio tempo')
        log.debug('Call: {}'.format(' '.join(call)))
        log.info("Writing to:\t%s", os.path.basename(tempfile2))
        if subprocess.call(call) == 1:
            raise RuntimeError

        log.info("Delete:\t%s", os.path.basename(tempfile1))
        os.remove(tempfile1)
        log.info("Moving:\t%s -> %s", os.path.basename(tempfile2),
                 os.path.basename(dest))
        shutil.move(tempfile2, dest)


class FrameSource(object): pass  # TODO: create an interface


class OpenCvFrameSource(FrameSource):
    """Uses the opencv video api as the default frame source. Try to ensure that
    the ffmpeg compiled, and which opencv is linked against, is the same ffmpeg
    being used to write video files."""

    def __init__(self, src):
        self.src = src
        self.capture = None
        self.frames = 0     # total frames in the video
        self._index = -1     # the next fr to be read (seek head position), zero-indexed

    @property
    def index(self):
        """The next fr to be read (seek head position), zero-indexed."""
        index = self.capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        if index != self._index:
            log.error('Frame source index mismatch:\t%d,%d' %
                      (index, self._index))
        return self._index

    def open(self):
        self.capture = cv2.VideoCapture(self.src)
        if not self.capture.isOpened():
            raise RuntimeError
        self.frames = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self._index = 0

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def seek_to_fr(self, index):
        # log.debug('Seek: %d', index)
        if index < 0 or index > self.frames-1:
            raise IndexError
        if self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, index) is not True:
            raise RuntimeError
        else:
            self._index = index

    def read(self):
        """Read a fr at self.index and return it. Return None if there are no
        frames available. Seek position will +1 automatically if successful."""
        # log.debug('Read: %d', self.index)
        if self.index < 0 or self.index > self.frames-1:
            return None
        success, fr = self.capture.read()
        if success is not True:     # this can can be `False` or `None`
            raise RuntimeError
        self._index += 1
        return fr

    def __del__(self):
        self.close()


class Frame(object):
    def __init__(self, d, i):
        self.data = d           # the fr as returned by a `FrameSource`
        self.index = i          # for internal tracking


class SourceFrame(Frame):       pass
class InterpolatedFrame(Frame): pass


class Renderer(object):
    def __init__(self, src, dest, sequence, rate, optflow_fn, interpolate_fn, w,
                 h, scaling_method, lossless, keep_subregions, show_preview,
                 embed_info, text_type, mark_frames, audio):
        self.src                = src
        self.dest               = dest
        self.sequence           = sequence
        self.rate               = rate
        self.optflow_fn         = optflow_fn
        self.interpolate_fn     = interpolate_fn
        self.w                  = w
        self.h                  = h
        self.scaling_method     = scaling_method
        self.lossless           = lossless
        self.keep_subregions    = keep_subregions
        self.show_preview       = show_preview
        self.show_progbar       = True
        self.embed_info         = embed_info
        self.text_type          = text_type
        self.mark_frames        = mark_frames
        self.audio              = audio
        self.fr_source          = None
        self.av_info            = avinfo.get_av_info(src)
        self.source_frs         = 0
        self.frs_interpolated   = 0
        self.frs_duped          = 0
        self.frs_dropped        = 0
        self.frs_written        = 0
        self.subs_to_render     = 0
        self.frs_to_render      = 0
        self.curr_sub_idx       = 0
        Display.window_title = os.path.basename(self.src) + ' - Butterflow'
        self.progress           = 0
        self.video_writer       = VideoWriter()

    def calc_frs_to_render(self, sub):
        """Returns the number of frames to be rendered in the subregion, based
        on its target duration, rate, speed, and its relation to the target
        global playback rate."""
        reg_duration = sub.duration / 1000.0
        frs_to_render = 0.0

        if sub.target_dur:
            frs_to_render = self.rate * (sub.target_dur / 1000.0)
        elif sub.target_fps:
            frs_to_render = sub.target_fps * reg_duration
        elif sub.target_spd:
            frs_to_render = self.rate * reg_duration * (1 / sub.target_spd)

        return int(frs_to_render+0.5)

    def render_subregion(self, sub):
        frs_to_render         = self.calc_frs_to_render(sub)
        if sub.fa == sub.fb:
            frs_to_render = 1
        interpolate_each_go   = int((float(frs_to_render)/sub.len)+0.5)
        if frs_to_render <= sub.len:
            interpolate_each_go = 0

        log.info("Frames to render:\t%d", frs_to_render)
        log.info("Frames in region:\t%d-%d", sub.fa, sub.fb)
        log.info("Region length:\t%d", sub.len)
        log.info("Region duration:\t%fs", sub.duration / 1000.0)
        log.info("Number of frame pairs:\t%d", sub.pairs)

        if interpolate_each_go == 0:
            log.warn("Interpolation rate:\t0 (only render S-frames)")
        else:
            log.info("Interpolation rate:\t%d", interpolate_each_go)

        steps_string = ""
        steps = SwFrameInterpolater.time_steps_for_nfrs(interpolate_each_go)
        for i, x in enumerate(steps):
            steps_string += "{:.3f}".format(x)
            if i < len(steps)-1:
                steps_string += ","
        if steps_string == "":
            steps_string = "N/A"
        log.info("Time stepping:\t%s", steps_string)

        log.info("Frames to write:\t%d", frs_to_render)

        if sub.pairs >= 1:
            will_have = (interpolate_each_go * sub.pairs) + sub.pairs
        else:
            will_have = 1

        extra_frs = will_have - frs_to_render

        log.info("Will have S and I-Frames:\t%d", will_have)
        log.info("Extra frames (to discard):\t%d", extra_frs)

        drop_every          = 0
        dupe_every          = 0

        if extra_frs > 0:
            drop_every = will_have / np.fabs(extra_frs)
        if extra_frs < 0:
            dupe_every = will_have / np.fabs(extra_frs)

        log.info("Drop every:\t%d", drop_every)
        log.info("Dupe every:\t%d", dupe_every)

        src_seen            = 0
        frs_interpolated    = 0
        work_idx            = 0
        frs_written         = 0
        frs_duped           = 0
        frs_dropped         = 0
        runs                = 0
        final_run           = False

        fr_1 = None

        if self.fr_source.index != sub.fa:
            self.fr_source.seek_to_fr(sub.fa)

        fr_2 = self.fr_source.read()

        if fr_2 is None:
            log.warn("First frame in the region is None (B is None)")
        else:
            src_seen += 1

        if frs_to_render == 1:
            runs = 1
            final_run = True
            log.info("Ready to run:\t1 time (only writing a S-frame)")
        else:
            # self.fr_source.seek_to_fr(sub.fa + 1)
            runs = sub.len
            log.info("Ready to run:\t%d times", runs)

        if self.scaling_method == settings.default['scaler_dn']:
            fr_2 = ImageProcess.scale(fr_2, self.w, self.h, self.scaling_method)

        show_n = settings.default['show_n_runs']
        show_period = settings.default['show_progress_period']
        showed_snipped_message = False

        def in_show_debug_range(x):
            if show_n == -1:
                return True
            return x <= show_n or x >= runs - show_n + 1

        if show_n == -1 or show_n*2 >= runs:
            log.info("Showing all runs:")
        else:
            log.info("Showing a sample of the first and last %d runs:", show_n)

        log.info('S-Frame\tI-Frames\tCompensation\tTotal\tProgress')

        for run in range(0, runs):
            # TODO: what to do when the period=1?
            fr_period = max(1, int(show_period * (runs - show_n*2)))

            if not in_show_debug_range(run) and not showed_snipped_message:
                log.info('<Snipping %d runs from the console, but will update '
                         'progress periodically every %d frames rendered>',
                         runs - show_n*2, fr_period)
                showed_snipped_message = True

            if not in_show_debug_range(run) and showed_snipped_message:
                if run % fr_period == 0:
                    log.info("<Rendering progress: {:.2f}%>".format(
                             self.progress*100))

            if run >= runs - 1:
                final_run = True

            pair_a = sub.fa + run
            pair_b = pair_a + 1 if run + 1 < runs else pair_a

            if final_run:
                log.info("Run %d (this is the final run):", run)
            # else:
            #     log.debug("Run %d:", run)
            # log.debug("Pair A: %d, B: %d", pair_a, pair_b)

            frs_to_write = []

            fr_1 = fr_2
            if fr_1 is None:
                log.error("A is None")

            if final_run:
                frs_to_write.append(SourceFrame(fr_1, 1))
                log.info("To write: S{}".format(pair_a))
            else:
                try:
                    fr_2 = self.fr_source.read()
                except RuntimeError:
                    log.error("Couldn't read %d (will abort runs)",
                              self.fr_source.index)
                    log.warn("Setting B to None")
                    fr_2 = None

                if fr_2 is None:
                    log.warn("B is None")
                    frs_to_write.append(SourceFrame(fr_1, 1))
                    final_run = True
                    log.info("To write: S{}".format(pair_a))

                if not final_run:
                    src_seen += 1

                    if self.scaling_method == settings.default['scaler_dn']:
                        fr_2 = ImageProcess.scale(fr_2, self.w, self.h,
                                                  self.scaling_method)

                    fr_1_gr = cv2.cvtColor(fr_1, cv2.COLOR_BGR2GRAY)
                    fr_2_gr = cv2.cvtColor(fr_2, cv2.COLOR_BGR2GRAY)

                    fu, fv = self.optflow_fn(fr_1_gr, fr_2_gr)
                    bu, bv = self.optflow_fn(fr_2_gr, fr_1_gr)

                    fr_1_32 = np.float32(fr_1) * 1/255.0
                    fr_2_32 = np.float32(fr_2) * 1/255.0

                    will_write = True

                    would_drp = []
                    c_interpolate_each_go = interpolate_each_go
                    c_work_idx = work_idx - 1

                    for x in range(1 + interpolate_each_go):
                        c_work_idx += 1
                        if drop_every > 0 and np.fmod(c_work_idx, drop_every) < 1.0:
                            would_drp.append(x + 1)

                    if len(would_drp) > 0:
                        txt = ""
                        for i, x in enumerate(would_drp):
                            txt += "{}".format(str(x))
                            if i < len(would_drp)-1:
                                txt += ","
                        # log.debug("Would drop indices:\t" + txt)

                        if len(would_drp) <= interpolate_each_go:
                            c_interpolate_each_go -= len(would_drp)
                            # log.debug("Compensating interpolation rate:\t%d (%+d)",
                            #           c_interpolate_each_go, -len(would_drp))
                        else:
                            will_write = False
                        if not will_write:
                            work_idx += 1
                            self.frs_dropped += 1
                            if in_show_debug_range(run):
                                log.info("Compensating, dropping S-frame")

                    if will_write:
                        interpolated_frs = self.interpolate_fn(fr_1_32, fr_2_32,
                                          fu, fv, bu, bv, c_interpolate_each_go)
                        frs_interpolated += len(interpolated_frs)

                        frs_to_write.append(SourceFrame(fr_1, 0))
                        for i, fr in enumerate(interpolated_frs):
                            frs_to_write.append(InterpolatedFrame(fr, i+1))

                        if in_show_debug_range(run):
                            temp_progress = (float(self.frs_written) +
                                         len(frs_to_write)) / self.frs_to_render
                            temp_progress *= 100.0
                            log.info("To write: S{}\tI{}\t{:+d},{}\t{:.2f}%".format(
                                    pair_a, len(interpolated_frs),
                                    -len(would_drp), len(frs_to_write),
                                    temp_progress))

            for i, to_write in enumerate(frs_to_write):
                fr               = to_write.data
                fr_type          = type(to_write)
                idx_between_pair = to_write.index

                work_idx += 1
                writes_needed = 1

                if dupe_every > 0 and np.fmod(work_idx, dupe_every) < 1.0:
                    frs_duped += 1
                    writes_needed = 2
                if final_run:
                    writes_needed = (frs_to_render - frs_written)
                    if drop_every > 0 and np.fmod(work_idx, drop_every) < 1.0:
                        self.frs_dropped += 1
                        if i == 0:
                            log.warn("Dropping S{}".format(pair_a))
                        else:
                            log.warn("Dropping I{}".format(idx_between_pair))
                        continue

                for write_idx in range(writes_needed):
                    fr_to_write = fr
                    frs_written += 1
                    self.frs_written += 1
                    self.progress = float(self.frs_written)/self.frs_to_render
                    is_dupe = False
                    if write_idx == 0:
                        if fr_type == SourceFrame:
                            self.source_frs += 1
                        else:
                            self.frs_interpolated += 1
                    else:
                        is_dupe = True
                        self.frs_duped += 1
                        if fr_type == SourceFrame:
                            log.warn("Duping S%d", pair_a)
                        else:
                            log.warn("Duping I%d", idx_between_pair)

                    if self.scaling_method == settings.default['scaler_up']:
                        fr = ImageProcess.scale(fr, self.w, self.h,
                                                self.scaling_method)

                    if self.mark_frames:
                        FrameMarker.draw(fr, fill=(fr_type == InterpolatedFrame))
                    if self.embed_info:
                        if writes_needed > 1:
                            fr_to_write = fr.copy()
                        DebugText.draw(fr_to_write, self.text_type, self.rate,
                                       self.optflow_fn, self.frs_written,
                                       pair_a, pair_b, idx_between_pair, fr_type,
                                       is_dupe, frs_to_render, frs_written, sub,
                                       self.curr_sub_idx, self.subs_to_render,
                                       drop_every, dupe_every, src_seen,
                                       frs_interpolated, frs_dropped, frs_duped)

                    if self.show_preview:
                        fr_to_show = fr.copy()
                        if self.show_progbar:
                            ProgressBar.draw(fr_to_show, progress=self.progress)
                        Display.show(fr_to_show)

                    self.video_writer.write(fr_to_write)

    def render(self):
        filename = os.path.splitext(os.path.basename(self.src))[0]
        tempfile1 = os.path.join(settings.default['tempdir'],
                                 '{}.{}.{}'.format(filename, os.getpid(),
                                       settings.default['v_container']).lower())
        log.info("Rendering to:\t%s", os.path.basename(tempfile1))
        log.info("Final destination:\t%s", self.dest)

        self.fr_source = OpenCvFrameSource(self.src)
        self.fr_source.open()
        self.video_writer.open(tempfile1, self.w, self.h, self.rate,
                               lossless=self.lossless)

        for sub in self.sequence.subregions:
            if not self.keep_subregions and sub.skip:
                continue
            else:
                self.subs_to_render += 1
                self.frs_to_render += self.calc_frs_to_render(sub)

        self.progress = 0
        log.info("Rendering progress:\t{:.2f}%".format(0))

        s = lambda w, h: min(float(self.w) / float(w), float(self.h) / float(h))

        dds = s(settings.debugtext['w_fit'], settings.debugtext['h_fit'])
        pbs = s(  settings.progbar['w_fit'],   settings.progbar['h_fit'])
        fms = s( settings.frmarker['w_fit'],  settings.frmarker['h_fit'])

        txt = 'Frame, Shape={}x{}, is too small to draw on: '\
              'Type={{}}\tScale={{}} < Min={{}}'.format(self.w, self.h)

        if self.embed_info and dds < settings.debugtext['min_scale']:
            log.warning(txt.format('info', dds, settings.debugtext['min_scale']))
            self.embed_info = False
        if self.show_preview and pbs < 1.0:
            log.warning(txt.format('progress', pbs, 1.0))
            self.show_prog_bar = False
        if self.mark_frames and fms < 1.0:
            log.warning(txt.format('marker', fms, 1.0))
            self.mark_frames = False

        for i, sub in enumerate(self.sequence.subregions):
            if not self.keep_subregions and sub.skip:
                log.info("Skipping Subregion (%d): %s", i, str(sub))
                continue
            else:
                log.info("Start working on Subregion (%d): %s", i, str(sub))
                self.render_subregion(sub)
                self.curr_sub_idx += 1
                log.info("Done rendering Subregion (%d)", i)

        if self.show_preview:
            Display.destroy()
        self.fr_source.close()
        self.video_writer.close()

        log.info("Rendering is finished")

        if self.audio:
            if self.av_info['a_stream_exists']:
                self.mix_orig_audio_with_rendered_video(tempfile1)
                return
            else:
                log.warn('Not mixing because no audio stream exists in the '
                         'input file')
        log.info("Moving: %s -> %s", os.path.basename(tempfile1), self.dest)
        shutil.move(tempfile1, self.dest)

    def mix_orig_audio_with_rendered_video(self, vid):
        progress = 0

        def update_progress():
            log.info("Mixing progress:\t{:.2f}%".format(progress*100))

        filename = os.path.splitext(os.path.basename(self.src))[0]
        audio_files = []
        to_extract = 0
        for sub in self.sequence.subregions:
            if not self.keep_subregions and sub.skip:
                continue
            else:
                to_extract += 1
        if to_extract == 0:
            progress += 1.0/3
            update_progress()
        progress_chunk = 1.0/to_extract/3

        for i, sub in enumerate(self.sequence.subregions):
            if not self.keep_subregions and sub.skip:
                continue
            tempfile1 = os.path.join(settings.default['tempdir'],
                                     '{}.{}.{}.{}'.format(filename, i,
                          os.getpid(), settings.default['a_container']).lower())
            log.info("Start working on audio from subregion (%d):", i)
            log.info("Extracting to:\t%s", os.path.basename(tempfile1))

            speed = sub.target_spd
            if speed is None:
                reg_duration = (sub.tb - sub.ta) / 1000.0
                frs = self.calc_frs_to_render(sub)
                speed = (self.rate * reg_duration) / frs
                log.info("Speed not set for mix, calculated as: %fx", speed)

            AudioVideoHelper.extract_audio(self.src, tempfile1, sub.ta, sub.tb,
                                           speed)
            audio_files.append(tempfile1)

            progress += progress_chunk
            update_progress()

        tempfile2 = os.path.join(settings.default['tempdir'],
                                 '{}.merged.{}.{}'.format(filename, os.getpid(),
                                       settings.default['a_container']).lower())
        log.info("Merging to:\t%s", os.path.basename(tempfile2))

        AudioVideoHelper.concat_av_files(tempfile2, audio_files)
        progress += 1.0/3
        update_progress()
        AudioVideoHelper.combine_av(vid, tempfile2, self.dest)
        progress += 1.0/3
        update_progress()

        for file in audio_files:
            log.info("Delete:\t%s", os.path.basename(file))
            os.remove(file)

        log.info("Delete:\t%s", os.path.basename(tempfile2))
        os.remove(tempfile2)
        log.info("Delete:\t%s", os.path.basename(vid))
        os.remove(vid)


def fr_at_time_step_wrp(args):  # to pass multiple args for Pool.map
    return SwFrameInterpolater.fr_at_time_step(*args)


def init_worker():  # captures ctrl+c
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class SwFrameInterpolater(object):
    """A parallel, naive implementation of software frame interpolation using
    provided optical flows (displacement fields)."""

    def __init__(self):
        pass

    @staticmethod
    def time_steps_for_nfrs(n):
        sub_divisions = n + 1
        time_steps = []
        for i in range(n):
            time_steps.append(max(0.0,
                              min(1.0, (1.0 / sub_divisions) * (i+1))))
        return time_steps

    @staticmethod
    def fr_at_time_step(target_fr, u, v, ts):
        shape = target_fr.shape
        fr = np.zeros(shape, dtype=np.float32)
        for idx in np.ndindex(shape):
            py = np.rint(idx[0] + v[idx[0], idx[1]] * ts)
            px = np.rint(idx[1] + u[idx[0], idx[1]] * ts)
            ch = idx[2]
            fr[idx] = target_fr[np.asscalar(np.int32(np.clip(py, 0, shape[0]-1))),
                                np.asscalar(np.int32(np.clip(px, 0, shape[1]-1))),
                                ch]
        return ts, fr

    def sw_interpolate_flow(self, prev_fr, next_fr, fu, fv, bu, bv, int_each_go):
        frames = []
        time_steps = SwFrameInterpolater.time_steps_for_nfrs(int_each_go)
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus, init_worker)
        work_steps = cpus/2
        try:
            for i in range(0, len(time_steps), work_steps):
                def blend_results(*args):
                    res = args[0]
                    def pairwise(iterable):
                        a = iter(iterable)
                        return izip(a, a)
                    for n, p in pairwise(res):
                        def alpha_blend(a, b, alpha):
                            return (1-alpha)*a + alpha*b
                        prv = p[1]
                        nxt = n[1]
                        bfr = alpha_blend(prv, nxt, n[0])  # n[0] is the step
                        bfr = (bfr*255.0).astype(np.uint8)
                        frames.append(bfr)
                task_list = []
                for j in range(work_steps):
                    if i+j > len(time_steps)-1:
                        continue
                    ts = time_steps[i+j]
                    task_list.extend([(next_fr, fu, fv, ts),
                                      (prev_fr, bu, bv, ts)])
                r = pool.map_async(fr_at_time_step_wrp, task_list,
                                   callback=blend_results)
                r.wait()  # block on results, will return them in order
        except KeyboardInterrupt:
            pool.terminate()
            class KeyboardInterruptError(Exception): pass
            raise KeyboardInterruptError  # re-raise
        pool.close()
        return frames


class VideoSequence(object):
    def __init__(self, duration, frames):
        self.duration       = duration      # in milliseconds
        self.frames         = frames        # total frames in the video
        self.subregions     = []
        self.add_subregion(Subregion(0, duration, skip=True))   # initial subregion is the whole video

    @classmethod
    def sequence_from_string(cls, s, duration, frames):    # duration=in milliseconds, src_frs=total frames
        seq = cls(duration, frames)
        if not s:
            seq.subregions[0].skip = False
            return seq
        duration = str(datetime.timedelta(seconds=duration / 1000.0))
        partition = list(duration.partition('.'))
        partition[-1] = partition[-1][:3]  # keep secs.xxx...
        duration = ''.join(partition)
        s = re.sub(sr_ful_pattern,
                   'a=0,b={}'.format(duration), re.sub(sr_end_pattern,
                                                       duration, s))
        subs = re.split(':a', s)
        new_subs = []
        if len(subs) > 1:
            for sub in subs:
                if not sub.startswith('a'):
                    new_subs.append('a'+sub)
                else:
                    new_subs.append(sub)
            subs = new_subs
        for sub in subs:
            match = re.match(sr_pattern, sub)
            if match:
                substr = str(sub)
                try:
                    to_ms = StringTools.time_string_to_milliseconds
                    sub = Subregion(to_ms(match.groupdict()['tm_a']),
                                    to_ms(match.groupdict()['tm_b']))
                except AttributeError as e:
                    raise AttributeError("Bad subregion: {} ({})".format(substr, e))
                target = match.groupdict()['target']
                val = match.groupdict()['val']
                if target == 'fps':
                    val = rate_from_input_str(val, -1)
                elif target == 'dur':
                    val = float(val)*1000.0
                elif target == 'spd':
                    val = float(val)
                setattr(sub, 'target_'+target, val)
                try:
                    seq.add_subregion(sub)
                except ValueError as e:
                    raise ValueError("Bad subregion: {} ({})".format(substr, e))
            else:
                raise ValueError('Unknown subregion syntax: {}'.format(sub))
        return seq

    def relative_pos(self, time):   # relative position in the video [0,1]
        return max(0.0, min(float(time) / self.duration, 1.0))

    def nearest_fr(self, time):
        return max(0, min(int(self.relative_pos(time) * self.frames),
                          self.frames-1))

    def add_subregion(self, sub):
        duration_string = str(datetime.timedelta(seconds=self.duration/1000.0))
        s = "{} > duration={}"
        if sub.ta > self.duration:
            raise ValueError(s.format(sub.ta, duration_string))
        if sub.tb > self.duration:
            raise ValueError(s.format(sub.tb, duration_string))

        sub.fa = self.nearest_fr(sub.ta)
        sub.fb = self.nearest_fr(sub.tb)

        if len(self.subregions) == 0 and sub.skip:
            self.subregions.append(sub)
            return

        temp_subs = []
        for s in self.subregions:
            if not s.skip:
                temp_subs.append(s)
        temp_subs.append(sub)
        temp_subs.sort(key=lambda x: (x.fb, x.fa), reverse=False)
        self.subregions = temp_subs

        temp_subs = []
        seq_len = len(self.subregions)
        i = 0
        while i < seq_len:
            curr = self.subregions[i]
            if i == 0 and curr.ta != 0:  # beginning to first subregion
                new_sub = Subregion(0, curr.ta, skip=True)
                new_sub.fa = 0
                new_sub.fb = curr.fa
                temp_subs.append(new_sub)
            temp_subs.append(curr)
            if i+1 == seq_len:          # last subregion to end
                if curr.tb != self.duration:
                    new_sub = Subregion(curr.tb, self.duration, skip=True)
                    new_sub.fa = curr.fb
                    new_sub.fb = self.frames-1
                    temp_subs.append(new_sub)
                break
            next = self.subregions[i+1]
            if curr.tb != next.ta:      # between subregion
                new_sub = Subregion(curr.tb, next.ta, skip=True)
                new_sub.fa = curr.fb
                new_sub.fb = next.fa
                temp_subs.append(new_sub)
            i += 1
        self.subregions = temp_subs

    def __str__(self):
        s = 'Sequence: Duration={} ({:.2f}s), Frames={}\n'.format(
            str(datetime.timedelta(seconds=self.duration/1000.0)),
            self.duration/1000,
            self.frames)
        for i, sub in enumerate(self.subregions):
            s += 'Subregion ({}): {}'.format('{}'.format(i), sub)
            if i < len(self.subregions)-1:
                s += '\n'
        return s


class Subregion(object):
    def __init__(self, ta, tb, skip=False):     # `skip`=render or not
        if ta > tb:
            raise AttributeError("a>b")
        if ta < 0:
            raise AttributeError("a<0")
        if tb < 0:
            raise AttributeError("a<0")
        self.ta           = ta      # starting time, in milliseconds
        self.tb           = tb      # end time, in milliseconds
        self.fa           = 0       # start frame, calculated when added to a VideoSequence
        self.fb           = 0       # end frame
        self.target_spd   = None
        self.target_dur   = None    # in milliseconds
        self.target_fps   = None
        self.skip         = skip
        if skip:
            self.target_spd = 1.0   # render at 1x speed if keeping regions

    @property
    def duration(self):             # in milliseconds
        return (self.tb - self.ta)

    @property
    def len(self):                  # frame length
        return (self.fb - self.fa) + 1

    @property                       # number of frame pairs in the region
    def pairs(self):
        return self.len - 1

    def intersects(self, o):
        """A `Subregion` intersects with another if either end, in terms of time
        and frame, falls within each others ranges or when one `Subregion` covers,
        or is enveloped by another."""
        return self.time_intersects(o) or self.fr_intersects(o)

    def time_intersects(self, o):
        if self is o or \
           self.ta == o.ta and self.tb == o.tb or \
           self.ta >  o.ta and self.ta <  o.tb or \
           self.tb >  o.ta and self.tb <  o.tb or \
           self.ta <  o.ta and self.tb >  o.tb:
            return True
        else:
            return False

    def fr_intersects(self, o):
        if self is o or \
           self.fa == o.fa and self.fb == o.fb or \
           self.fa >  o.fa and self.fa <  o.fb or \
           self.fb >  o.fa and self.fb <  o.fb or \
           self.fa <  o.fa and self.fb >  o.fb:
            return True
        else:
            return False

    def __str__(self):
        vs = lambda x: x if x is not None else '?'
        ts = lambda x: str(datetime.timedelta(seconds=x/1000.0))

        s = 'Time={}-{} Frames={}-{} Speed={},Duration={},Fps={}'.format(
            ts(self.ta),
            ts(self.tb),
            self.fa,
            self.fb,
            vs(self.target_spd),
            vs(self.target_dur),
            vs(self.target_fps))
        if self.skip:
            s += ' (autogenerated subregion)'
        return s


class ImageProcess(object):
    @staticmethod
    def scale(fr, w, h, method):    # TODO: breaking when there's more than 1 subregion?
        return cv2.resize(fr, (w, h), interpolation=method)

    @staticmethod
    def alpha_blend(A, B, alpha, gamma=0):
        """Similar to cv2.addWeighted, which calculates the weighted sum of two
        arrays, but where beta=(1-alpha)."""
        return A*alpha + B*(1-alpha) + gamma    # gamma is a scalar added to each sum

    @staticmethod
    def apply_mask(A, B, mask):
        """Apply `mask` to `A`. A=top layer, B=bottom layer. 1=fully opaque A,
        0=transparent A (show B underneath)"""
        M = np.zeros(A.shape, dtype=np.uint8)
        for idx in np.ndindex(A.shape):
            alpha = mask[idx[0]][idx[1]]        # mask should only be 1-ch, apply to all channels in A
            M[idx] = ImageProcess.alpha_blend(A[idx], B[idx],
                                              alpha)  # 1=fully opaque A, 0=show B underneath
        return M


class SamplingRect(object):
    def __init__(self, center, v1, v2):
        self.center = center
        self.v1 = v1            # vertex,          top-left of rect
        self.v2 = v2            # opposite vertex, bot-right of rect

    @property
    def c(self):
        """Number of cols (width of rect)"""
        return self.v2[1] - self.v1[1]

    @property
    def r(self):
        """Number of rows (height of rect)"""
        return self.v2[0] - self.v1[0]


class MatrixSampler(object):
    """Divides a nxm matrix into a grid of `n`, even-sized, although may not be
    similar in length and width, SamplingRects. Granularity, the area of the
    rects, is controlled by how big or small `n` is."""
    def __init__(self, r, c, n):    # n=number of samples / controls granularity, must be a perfect square
        if n < 1 or not int(np.sqrt(n)+0.5) ** 2 == n:  # is perfect square?
            raise ValueError('n (%d) is an imperfect square' % n)
        self.r = r             # rows, height of M
        self.c = c             # cols, width of M
        self.sampling_rects = self.get_sampling_rects(n)  # array of samples

    def get_sampling_rects(self, n):
        """Returns array of SamplingRects. `n`=number of samples. Splits matrix
        into rects, return center of rects, lengths and widths may be unequal.
        `n` must be a perfect square number, e.g. 1,4,9,16,25,36,49,...
        See: http://oeis.org/search?q=1%2C+4%2C9%2C16%2C25"""
        # print("Rows: %d, Cols: %d" % (r, c))
        rects = []
        dividers = int(np.sqrt(n)-1)  # total horizontal and vertical dividers=dividers*2
        if dividers == 0:
            rects.append((int(self.r/2), int(self.c/2)))
        else:
            spx = dividers + 1          # SamplingRects per row and col (regions to be sampled)
            rs = self.r / spx             # step for row
            cs = self.c / spx             # step for col
            for r in xrange(0, spx):
                for c in xrange(0, spx+1):  # NOTE: +1 to spx to cover more of the image
                    rects.append(SamplingRect(
                                               ( int(rs*r+rs/2), int(cs*c+cs/2) ),  # center
                                               ( int(rs*r), int(cs*c) ),            # v1, top-left
                                               ( int(rs*r+rs), int(cs*c+cs) ) ))      # v2, bottom-right
                    # last_r = rects[-1]
                    # print('Center: {}, v1: {}, v2: {}'.format(last_r.center, last_r.v1, last_r.v2))
        # print(rects)
        return rects

    def draw(self, fr, center_points=True, rect_outlines=False, point_text=False):  # visualize the SamplingRects
        for r in self.sampling_rects:
            center = (r.center[1], r.center[0])     # flip coordinates when drawing, origin at top left
            v1     = (r.v1[1], r.v1[0])     # top-left vertex
            v2     = (r.v2[1], r.v2[0])     # bot-right
            if center_points:
                cv2.line(fr, center, center,  # center dot
                              cv2.cv.RGB(255, 255, 255),
                              1,             # thickness
                              4)              # line type
            if rect_outlines:
                cv2.rectangle(fr, v1, v2,
                              cv2.cv.RGB(0, 0, 0),
                              1,          # thickness
                              4)          # line type
            if point_text:
                cv2.putText(fr,
                            "(%d,%d)" % (r.center[0], r.center[1]),
                            (center[0]-25, center[1]+20),  # origin: bottom left of text
                            cv2.cv.CV_FONT_HERSHEY_PLAIN,
                            0.7,          # scale
                            cv2.cv.RGB(255, 255, 255),
                            1,            # thickness
                            8)            # line type
        return fr

    def get_sample(self, rect, M, ch=0):  # `M`=the matric to splice, `ch`=channel of the matrix
        """Splices `M`, returns a sample from a sampling rect"""
        shape = M.shape
        if shape[0] != self.r or shape[1] != self.c:
            raise ValueError('M rows=%d,cols=%d must equal rows=%d,cols=%d' % (
                             shape[0], shape[1], self.r, self.c))
        v1 = rect.v1                # top-left vertex
        v2 = rect.v2                # bot-right
        if len(shape) == 3:  # >1 channel
            return M[v1[0]:v2[0],           # top-left row -> bot-right row (h)
                     v1[1]:v2[1],           # top-left col -> bot-right col (w)
                     ch]
        else:
            return M[v1[0]:v2[0],
                     v1[1]:v2[1]]


class FlowMagnitudeDirectionInfo(object):
    masks_written = 0       # for internal tracking

    def __init__(self, u, v, maglimit=-1):
        self.u = u                  # u flow
        self.v = v                  # v flow
        self.max_magnitude      = -1
        self.M_raw_mag    = np.zeros(u.shape, dtype=np.float32)   # raw mag data
        self.M_scaled_mag = np.zeros(u.shape, dtype=np.float32)   # scaled, relative to the max mag, use mostly for masking: values between [0,1]: 1=white (fully opaque), 0=black (fully transparent)
        self.M_angle      = np.zeros(u.shape, dtype=np.float32)   # angle/direction/bearing in radians from [-pi,pi], or [-180,180]
        # self.M_quadrant   = np.zeros(u.shape, dtype=np.uint8)     # pointing to quadrant
        self.get_magnitude_direction_info(limit=maglimit)

    def get_magnitude_direction_info(self, limit=-1):
        for idx in np.ndindex(self.u.shape):  # O(n^2) time
            x = self.u[idx]  # c
            y = self.v[idx]  # r
            m = np.sqrt(x*x + y*y)  # get pixel magnitude at each location given u and v flows
            # TODO: need to profile these functions?
            # m = self.magnitude(np.array([self.v[idx],    # y
            #                              self.u[idx]]))  # x, same as np.linalg.norm(np.array((self.v[idx],0))-np.array((0,self.u[idx])))?
            if m > self.max_magnitude:
                self.max_magnitude = m
            # if m < limit:    # if it doesn't move more than X, make 0=transparent
                # m = 0
            self.M_raw_mag[idx] = m        # set raw magnitude vals
            # np.arctan2(x1,x2) ~ inverse tangent(x1/x2) "the "y-coordinate" is the first function parameter (x1), the "x-coordinate" is the second" (x2)
            # returns: "Array of angles in radians, in the range [-pi, pi]." [-180,180]
            # >>> x = np.array([+1, -1, -1, +1,  0,  0, +1, -1, 0])
            # >>> y = np.array([+1, +1, -1, -1, +1, -1,  0,  0, 0])
            # >>> np.rad2deg(np.arctan2(y, x))
            # array([  45.,  135., -135.,  -45.,   90.,  -90.,    0.,  180.,    0.])
            # if np.isclose(x, 0) and np.isclose(y, 0):  # indeterminate case?
            #     self.M_angle[idx] = np.nan
            # else:
            #     self.M_angle[idx] = np.arctan2(y, x)  # set direction
            self.M_angle[idx] = np.arctan2(y, x)  # set direction

    def rescale_magnitude_data(self, max_magnitude):
        # to make portion of mask transparent so layer underneath shows through:
        self.M_scaled_mag = self.M_raw_mag / max_magnitude    # best to use max of forward and backward flows


class MotionBlur(object):
    @staticmethod
    def rotateX(M, theta): pass  # TODO: implement this (so we can rotate the kernel to an arbitrary angle)

    @staticmethod
    def visualize_kernel(K):   # diplays a MotionBlur.kernel, `K`, as a black & white image
        K = (K / np.max(K)) * 255   # np.uint8, [0,255]
        Display.show(K)

    # TODO: memoize kernel?
    @staticmethod
    def kernel(n, a, half=True):  # `n`=size of the kernel, should be odd, `a`=angle in degrees divisble by 45
        """Returns the motion blur kernel `K`, an n-by-n matrix with values of
        either 0 or 1/n or 1/(n*0.5) if half=True (the elements in `K` will sum
        to 1). `n` must be odd. `a`, the angle of the blur, must be divisible
        by 45. Possible `a` values: 0, 45, 90, 135, 180, 225, 270, 315, 360..."""
        if n % 2 == 0:
            raise ValueError('`n` %d must be odd' % n)
        if a != 0:
            if a % 45 != 0:
                raise ValueError('`a` %f must be 0 or divisible by 45' % a)
        K = np.zeros((n, n), dtype=np.float32)
        h = int((n-1)/2)  # halfway index
        if a == 0 or (a/45) % 2 == 0:     # step by 90
            if half:
                K[h,h:] = np.ones(h+1)        # set only half
            else:
                K[h,:] = np.ones(n)         # fill rows/2 with ones
            if a != 0:
                rotate_n = int(a/90.0)
                K = np.rot90(K, k=rotate_n)  # times to rotate counterclockwise
        else:                            # step by 45
            np.fill_diagonal(K, 1.0)     # fill 135-315 diagonal with ones
            if half:
                K[h+1:,h+1:] = 0             # clear out 315 opposite half
            K = np.rot90(K, k=3)         # rotate to 45, the starting position
            if a != 45:
                rotate_n = int((a/45-1)/2)
                K = np.rot90(K, k=rotate_n)
        if half:
            K = K*(1.0/(h+1))  # normalize (could divide by np.count_nonzero())
        else:
            K = K*(1.0/n)
        return K


class FlowVisualizer(object):
    def __init__(self, sampler, md):  # `sampler`: MatrixSampler, `md`: FlowMagnitudeDirectionInfo
        self.sampler = sampler      # MatrixSampler that can sample FlowMagnitudeDirectionInfo (same r,c)
        self.md = md           # FlowMagnitudeDirectionInfo

    def draw(self, fr):  # `fr`: to draw on
        for rect in self.sampler.sampling_rects:
            avg_scaled_mag = np.average(self.sampler.get_sample(rect,
                                        self.md.M_scaled_mag))   # scaled magnitude data

            draw_arrow = True
            c = None
            if avg_scaled_mag > 0.5:  # limit
                c = cv2.cv.RGB(128,   0, 128)    # purple
            elif avg_scaled_mag > 0.25 and avg_scaled_mag > 0.1:
                c = cv2.cv.RGB(128, 128,   0)    # yellow
            elif avg_scaled_mag > 0.05:
                c = cv2.cv.RGB(0,   128,   0)    # green
                draw_arrow = True
            else:
                continue    # don't draw if there's little to no movement

            center = (rect.center[1], rect.center[0])  # flip for drawing: (x,y), origin top-left, going right->down
            v1 = (rect.v1[1], rect.v1[0])          # flipped
            v2 = (rect.v2[1], rect.v2[0])          # flipped

            cv2.rectangle(fr, v1, v2, c, -1)  # color rect based on magnitude

            # "A simple way to calculate the mean of a series of angles (in the interval [0°, 360°)) is to
            # calculate the mean of the cosines and sines of each angle, and obtain the angle by calculating the
            # inverse tangent."
            # https://en.wikipedia.org/wiki/Mean_of_circular_quantities
            # angles = self.sampler.get_sample(rect, self.md.M_angle)     # get sample, direction data (or flow data?), in rect
            #
            # avg_y = np.average(np.sin(angles))    # avg y-coord of a unit vector on a unit circle (col/width)
            # avg_x = np.average(np.cos(angles))    # avg x-coord (row/height)
            #
            # avg_direction = np.arctan2(avg_y, avg_x)    # convert to polar-coords [-pi,pi]
            #
            # if np.isclose(np.rad2deg(avg_direction), 0):
            #     pass
            #
            # y = np.sin(avg_direction)   # convert back to cartesian, avg y-coord of a unit vector on a unit circle (col/width)
            # x = np.cos(avg_direction)   # avg x-coord (row/height)

            y_displacement = self.sampler.get_sample(rect, self.md.v)     # get flow data, in rect
            x_displacement = self.sampler.get_sample(rect, self.md.u)     # get flow data, in rect

            y = np.average(y_displacement)
            x = np.average(x_displacement)

            # direction = np.arctan2(y, x)    # general direction
            # y = np.sin(direction)           # y-coord on a unit circle (col/width)
            # x = np.cos(direction)           # x-coord (row/height)

            # setattr(rect, 'average_direction', avg_dir)  # TODO: save?
            # setattr(rect, 'x_coord', x)
            # setattr(rect, 'y_coord', y)

            # scale = min(rect.r, rect.c) * avg_scaled_mag  # draw arrow, scale it to keep inside rect
            scale = min(rect.r, rect.c)  # draw arrow, scale it to keep inside rect
            y = int(y * scale)  # TODO: don't have to cast down just yet, save precision
            x = int(x * scale)

            # h = int(np.sqrt(x*x + y*y)/2)  # half of arrow length, use to adjust arrow center to use full rect
            arrow_from = center                     # (x,y)
            arrow_to = (center[0]+x, center[1]+y)   # already flipped # TODO: fix this
            # arrow_to = (center[0]+x-h, center[1]+y-h)   # already flipped # TODO: fix this

            # if x > 0 and y > 0:  # pointing to quadrant=1
            #     arrow_from = (center[0]-h, center[1]-h)  # extend back to opposite quadrant=3
            # elif x < 0 and y > 0:  # q=2
            #     arrow_from = (center[0]-h, center[1]+h)  # q=4
            # elif x < 0 and y < 0:  # q=3
            #     arrow_from = (center[0]+h, center[1]+h)  # q=1
            # elif x > 0 and y < 0:  # q=4
            #     arrow_from = (center[0]+h, center[1]-h)  # q=2
            # if x == 0 and y == 0:
            #     arrow_to = center[0]+x, center[1]+y  # from center of rect

            if draw_arrow:
                ArrowShape.draw(fr, arrow_from, arrow_to,  # from -> to, using flipped coords
                                0.25,      # tip length as a percentage of total length
                                draw_tip=False, closed_tip=True)


class Drawable(object): pass   # TODO: implement interface, raise NotImplementedError if draw() isn't defined


class DebugText(Drawable):  # fit: 768x216
    @staticmethod
    def draw(fr, text_type, rate, optflow_fn, tot_frs_written, pair_a, pair_b,
             idx_between_pair, fr_type, is_dupe, frs_to_render, frs_written,
             sub, curr_sub_idx, subs_to_render, drop_every, dupe_every, src_seen,
             frs_interpolated, frs_dropped, frs_duped):
        # drawing with the origin at the top left
        # drawing text from the top left and right of the fr
        w = fr.shape[1]         # cols
        h = fr.shape[0]         # rows

        # scale the text up and down to fit inside the fr
        scale = min(min( float(w)  / settings.debugtext['w_fit'],   # cols/smallest fitting width  768
                         settings.debugtext['max_scale']),
                     min(float(h)  / settings.debugtext['h_fit'],  # rows/smallest fitting height 216
                         settings.debugtext['max_scale']))

        def draw_stroke(x, y):  # draws the dark colored outline around light colored text
            cv2.putText(fr, x, y,
                        settings.debugtext['font_face'],
                        scale,
                        settings.debugtext['dark_color'],
                        settings.debugtext['stroke_thick'],
                        settings.debugtext['font_type'])

        color = settings.debugtext['light_color']
        if text_type == 'dark':
            color = settings.debugtext['dark_color']

        def draw_text(x, y):
            cv2.putText(fr, x, y,
                        settings.debugtext['font_face'],
                        scale,
                        color,
                        settings.debugtext['thick'],
                        settings.debugtext['font_type'])

        txt = "butterflow {} ({})\n"\
              "Res: {}x{}\n"\
              "Playback Rate: {:.2f} fps\n"
        txt = txt.format(settings.default['version'], sys.platform, w, h, rate)

        argspec = inspect.getargspec(optflow_fn)
        defaults = list(argspec.defaults)           # arg values
        args = argspec.args[-len(defaults):]        # args
        flow_kwargs = collections.OrderedDict(zip(args, defaults))  # dict of (args, arg values)

        if flow_kwargs is not None:
            flow_format = ''
            i = 0
            for k, v in flow_kwargs.items():
                value_format = "{}"
                if isinstance(v, bool):
                    value_format = "{:1}"
                flow_format += ("{}: "+value_format).format(k.capitalize()[:1],v)
                if i == len(flow_kwargs)-1:
                    flow_format += '\n\n'
                else:
                    flow_format += ', '
                i += 1
            txt += flow_format

        yes_or_no_string = lambda x: 'Y' if x else 'N'

        txt += "Frame: {}\n"\
               "Pair Index: {}, {}, {}\n"\
               "Type Src: {}, Int: {}, Dup: {}\n"\
               "Mem: {}\n"
        txt = txt.format(tot_frs_written,
                         pair_a, pair_b, idx_between_pair,
                         yes_or_no_string(fr_type is SourceFrame),
                         yes_or_no_string(fr_type is InterpolatedFrame),
                         yes_or_no_string(is_dupe > 0),
                         hex(id(fr)))               # object's memory address: 0x...

        for i, line in enumerate(txt.split('\n')):
            line_sz, _ = cv2.getTextSize(line,
                                         settings.debugtext['font_face'],
                                         scale,
                                         settings.debugtext['thick'])
            _, line_h = line_sz
            origin = (int(settings.debugtext['l_pad']),
                      int(settings.debugtext['t_pad'] +
                      (i * (line_h + settings.debugtext['ln_b_pad']))))
            if text_type == 'stroke':
                draw_stroke(line, origin)
            draw_text(line, origin)

        txt = "Region {}/{}, F: [{}, {}], T: [{:.2f}, {:.2f}s]\n"\
              "Len F: {}, T: {:.2f}s\n"
        txt = txt.format(curr_sub_idx,
                         subs_to_render-1,
                         sub.fa,
                         sub.fb,
                         sub.ta / 1000.0,
                         sub.tb / 1000.0,
                         sub.len,
                         sub.duration / 1000.0)

        def string_or_placeholder(str_fmt, str):
            if str is None:
                return settings.debugtext['placeh']
            return str_fmt.format(str)

        txt += "Target Spd: {} Dur: {} Fps: {}\n"
        txt = txt.format(string_or_placeholder('{:.2f}',  sub.target_spd),
                         string_or_placeholder('{:.2f}s',
                                               sub.target_dur / 1000.0)
                         if sub.target_dur else settings.debugtext['placeh'],
                         string_or_placeholder('{:.2f}',  sub.target_fps))

        txt += "Out Len F: {}, T: {:.2f}s\n"\
               "Drp every {:.1f}, Dup every {:.1f}\n"\
               "Src seen: {}, Int: {}, Drp: {}, Dup: {}\n"\
               "Write Ratio: {}/{} ({:.2f}%)\n"
        txt = txt.format(frs_to_render, frs_to_render / float(rate),
                         drop_every, dupe_every,
                         src_seen, frs_interpolated, frs_dropped, frs_duped,
                         frs_written, frs_to_render,
                         frs_written * 100.0 / frs_to_render)

        for i, line in enumerate(txt.split('\n')):
            line_sz, _ = cv2.getTextSize(line,
                                         settings.debugtext['font_face'],
                                         scale,
                                         settings.debugtext['thick'])
            line_w, line_h = line_sz
            origin = (int(w - settings.debugtext['r_pad'] - line_w),
                      int(settings.debugtext['t_pad'] +
                      (i * (line_h + settings.debugtext['ln_b_pad']))))
            if text_type == 'stroke':
                draw_stroke(line, origin)
            draw_text(line, origin)


class ProgressBar(Drawable):  # fit: 420x142
    @staticmethod
    def draw(fr, progress=0.0):  # `progress`: from [0,1]
        # drawing with the origin at the top left
        w = fr.shape[1]
        h = fr.shape[0]

        def draw_stroke(v1, v2, shift_x, shift_y, extend_x, extend_y):
            cv2.rectangle(fr,
                          (v1[0]+shift_x-extend_x, v1[1]+shift_y-extend_y),  # top-left vertex
                          (v2[0]+shift_x+extend_x, v2[1]+shift_y+extend_y),  # bot-right opposite vertex
                          settings.progbar['stroke_color'],
                          settings.progbar['ln_type'])

        def draw_rectangle(v1, v2):
            cv2.rectangle(fr, v1, v2,
                          settings.progbar['color'],
                          settings.progbar['ln_type'])

        t_v1 = (int(w * settings.progbar['side_pad']),
                int(h * settings.progbar['t_pad']))                 # top-top-left
        t_v2 = (int(w * (1 - settings.progbar['side_pad'])), t_v1[1] +
                         settings.progbar['out_thick'])             # top-bot-right

        draw_stroke(t_v1, t_v2, 0, -settings.progbar['stroke_thick'], 1, 0)
        draw_rectangle(t_v1, t_v2)

        b_v1 = (t_v1[0], t_v2[1] + 2 * settings.progbar['in_pad'] +
                settings.progbar['in_thick'])                       # bot-top-left
        b_v2 = (t_v2[0], b_v1[1] + settings.progbar['out_thick'])   # bot-bot-right

        draw_stroke(b_v1, b_v2, 0, settings.progbar['stroke_thick'], 1, 0)
        draw_rectangle(b_v1, b_v2)

        l_v1 = t_v1                                                 # left-top-left
        l_v2 = (b_v1[0] + settings.progbar['out_thick'],
                b_v1[1] + settings.progbar['out_thick'])            # left-bot-right

        draw_stroke(l_v1, l_v2, -settings.progbar['stroke_thick'], 0, 0, 0)
        draw_rectangle(l_v1, l_v2)

        r_v1 = (t_v2[0] - settings.progbar['out_thick'], t_v1[1])   # right-top-left
        r_v2 = b_v2                                                 # right-bot-right

        draw_stroke(r_v1, r_v2, settings.progbar['stroke_thick'], 0, 0, 0)
        draw_rectangle(r_v1, r_v2)

        if progress <= 0:
            return

        padding = settings.progbar['out_thick'] + settings.progbar['in_pad']
        max_w = int(r_v2[0] - padding)
        min_w = int(l_v1[0] + padding)

        bar_v1 = (t_v1[0] + padding, l_v1[1] + padding)
        bar_v2 = (max(min_w, min(max_w,
                                 int(max_w * progress))), b_v2[1] - padding)

        draw_stroke(bar_v1, bar_v2, 0, 0, 1, 1)
        draw_rectangle(bar_v1, bar_v2)                              # actual progress bar


class FrameMarker(Drawable):  # fit: 572x142
    """Draws a frame marker, an outer white circle, with a red inner circle,
    only when `fill=True`, on the bottom right corner of a `fr`."""
    @staticmethod
    def draw(fr, fill=True):  # `fill`: the circle or not, will show the fill color
        # drawing with the origin at the top left
        x = int(fr.shape[1] - (settings.frmarker['r_pad'] +
                               settings.frmarker['outer_radius']))
        y = int(fr.shape[0] -  settings.frmarker['d_pad'] -
                               settings.frmarker['outer_radius'])
        cv2.circle(fr, (x,y),                  # the white outer circle
                   settings.frmarker['outer_radius'],
                   cv2.cv.RGB(255, 255, 255),
                   -1,                         # filled circle
                   cv2.cv.CV_AA)
        color = cv2.cv.RGB(255, 255, 255)
        if fill:
            color = cv2.cv.RGB(255, 0, 0)
        cv2.circle(fr, (x,y),                 # the inner circle, red or white
                   settings.frmarker['inner_radius'],
                   color,
                   -1,                        # filled circle
                   cv2.cv.CV_AA)


class ArrowShape(Drawable):
    @staticmethod
    def draw(fr, p1, p2, tip_length, draw_tip=True, closed_tip=False):  # from p1 -> p2, pass flipped coords (c,r) or (x,y)
        # drawing with the origin at top left
        tip_size = np.linalg.norm(np.array(p1)-np.array(p2)) * tip_length  # scale the tip
        # angle of the vector
        a = np.arctan2(p1[1]-p2[1],    # y1-y2
                       p1[0]-p2[0])    # x1-x2
        # find end point of vector, given starting point, magnitude R, and angle
        # (x2-x1)=R cos(theta), solve for x2
        # (y2-y1)=R sin(theta), solve for y2
        aa = np.deg2rad(45)            # angle of the arrow tip
        t1 = (p2[0] + tip_size * np.cos(a + aa),    # vert1 of tip
              p2[1] + tip_size * np.sin(a + aa))
        t2 = (p2[0] + tip_size * np.cos(a - aa),    # vert2 of tip
              p2[1] + tip_size * np.sin(a - aa))
        cv2.line(fr, p1, p2,                        # line from -> to
                 cv2.cv.RGB(255, 255, 255),
                 1)
        if draw_tip:
            if closed_tip:
                triangle = np.array([list(p2),              # tip of arrow, tri vert 0
                                     list(t1),              # tri vert 1
                                     list(t2)], np.int32)   # tri vert 2
                cv2.fillConvexPoly(fr, triangle,
                                   cv2.cv.RGB(0, 0, 255),
                                   cv2.cv.CV_AA)
            else:
                cv2.line(fr, p2, (int(t1[0]), int(t1[1])),  # tri vert 1 line
                         cv2.cv.RGB(255, 255, 255), 1)
                cv2.line(fr, p2, (int(t2[0]), int(t2[1])),  # tri vert 2 line
                         cv2.cv.RGB(255, 255, 255), 1)
