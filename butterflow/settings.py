# -*- coding: utf-8 -*-

import os
import logging
import cv2


default = {
    'debug_opts':     False,    # display debugging opts?
    'show_n_runs':            15,   # show first and last n runs, -1 shows all
    'show_progress_period':   0.1,  # show progress every % if snipped
    # default logging level
    # levels in order of urgency: critical, error, warning, info, debug
    'loglevel':     logging.WARNING,    # starting loglevel
    # loglevel will be set to INFO if verbose count is n
    'loglevel_1':     logging.INFO,
    'loglevel_2':     logging.DEBUG,
    'verbose':        False,
    'quiet':          False,
    # quiet loglevel
    'loglevel_quiet': logging.ERROR,
    # only support ffmpeg for now, but we can change to avutil if needed
    # Documentation: https://ffmpeg.org/ffmpeg.html
    'avutil':         'ffmpeg',
    # avutil and encoder loglevel
    # options: panic, fatal, error, warning, info, verbose, debug, trace
    'av_loglevel':    'error',  # `info` is default
    # default loglevel is `info` for x264 and x265
    # x265 opts: `none`, `error`, `warning`, `info`, `debug`, plus `full`
    'enc_loglevel':   'info',
    # x265 is considered a work in progress and is under heavy development
    # + 50-75% more compression efficiency than x264
    # + retains same visual quality
    # - veryslow preset encoding speed is noticeably slower than x264
    # for x265 options, See: http://x265.readthedocs.org/en/default/cli.html
    #
    # x264 is stable and used in many popular video conversion tools
    # + uses gpu for some lookahead ops but doesn't mean algos are optimized
    # + well tuned for high quality encodings
    'cv':             'libx264',
    # See: https://trac.ffmpeg.org/wiki/Encode/H.264
    # See: https://trac.ffmpeg.org/wiki/Encode/H.264#a2.Chooseapreset
    # presets: ultrafast, superfast, veryfast, faster, fast, medium, slow,
    # slower, veryslow
    'preset':         'veryslow',
    'crf':            18,       # visually lossless
    # scaling opts
    'video_scale':    1.0,
    'scaler_up':      cv2.cv.CV_INTER_AREA,
    # CV_INTER_CUBIC looks best but is slower, CV_INTER_LINEAR is faster but
    # still looks okay
    'scaler_dn':      cv2.cv.CV_INTER_CUBIC,
    # mixing opts
    'v_container':    'mp4',
    # See: https://trac.ffmpeg.org/wiki/Encode/HighQualityAudio
    'a_container':    'm4a',   # will keep some useful metadata
    # audio codec and quality
    # See: https://trac.ffmpeg.org/wiki/Encode/AAC
    'ca':             'aac',   # built in encoder, doesnt require an ext lib
    'ba':             '192k',  # bitrate, usable >= 192k
    'qa':             4,       # quality scale of audio from 0.1 to 10
    # location of files and directories
    'out_path':       os.path.join(os.getcwd(), 'out.mp4'),
    # farneback optical flow options
    'pyr_scale':      0.5,
    'levels':         3,
    'winsize':        25,
    'iters':          3,
    'poly_n_choices': [5, 7],
    'poly_n':         5,
    'poly_s':         1.1,
    'fast_pyr':       False,
    'flow_filter':    'box',
    # -1 is max threads and it's the opencv default
    'ocv_threads':    -1,   # 0 will disable threading optimizations
    # milliseconds to display image in preview window
    'imshow_ms':      1,    # 0 will display until key is pressed
}

# drawables settings

debugtext = {
    'text_type':     'light',      # other options: `dark`, `stroke`
    'light_color':   cv2.cv.RGB(255, 255, 255),
    'dark_color':    cv2.cv.RGB(0, 0, 0),
    # w_fit and h_fit is the minimium size in which the unscaled
    # CV_FONT_HERSHEY_PLAIN font text fits in the rendered video. The font is
    # scaled up and down based on these reference values
    'w_fit':         768,
    'h_fit':         216,
    'font_face':     cv2.cv.CV_FONT_HERSHEY_PLAIN,
    'font_type':     cv2.cv.CV_AA,
    'max_scale':     1.0,
    'thick':         1,
    'stroke_thick':  2,
    't_pad':         30,
    'l_pad':         20,
    'r_pad':         20,
    'ln_b_pad':      10,    # spacing between lines
    'min_scale':     0.55,  # don't draw if the font is scaled below this
    'placeh':     '?',      # placeholder if value in fmt text is None
}

progbar = {
    'w_fit':        420,
    'h_fit':        142,
    't_pad':        0.7,    # relative padding from the top of the fr
    'side_pad':     0.12,   # padding on each side
    'out_thick':    3,      # px of lines that make up the outer rectangle
    'stroke_thick': 1,      # size of the stroke in px
    'ln_type':      -1,     # -1=a filled line
    'in_pad':       3,      # pad from the inner bar
    'in_thick':     15,     # thickness of the inner bar
    'color':        cv2.cv.RGB(255,255,255),
    'stroke_color': cv2.cv.RGB(192,192,192),
}

frmarker = {
    'w_fit':        572,
    'h_fit':        142,
    'outer_radius': 4,
    'r_pad':        20,     # padding from bot-right-right corner of fr
    'd_pad':        20,     # padding from bot-right-down
    'inner_radius': 3,
}

# override default settings with development settings
# ignore errors when dev_settings.py does not exist
# ignore errors when `default` variable is not defined in the file
try:
    from butterflow import dev_settings
    for k, v in dev_settings.default.items():
        default[k] = v
except ImportError:
    pass
except AttributeError:
    pass
