from tempfile import mktemp
import subprocess
import os
import logging
import kaldi_model

from .util.paths import get_binary

EXECUTABLE_PATH = get_binary("ext/k3")
logger = logging.getLogger(__name__)

STDERR = subprocess.DEVNULL


class Kaldi:
    def __init__(self, nnet_dir=None, hclg_path=None, proto_langdir=None):

        args = []

        if nnet_dir is not None:
            args.extend([nnet_dir, hclg_path])

        if not os.path.exists(hclg_path):
            logger.error('hclg_path does not exist: %s', hclg_path)

        self.model = kaldi_model.kaldi_model(*args)
        self.finished = False

    def process_chunk(self, buf):
        # Wait until we're ready

        cnt = int(len(buf)/2)
        buf_file = mktemp("_gentle_buf")
        file = open(buf_file, "wb")
        file.write(buf)
        file.close()
        ret = self.model.process_chunk(buf_file, cnt)
        if os.path.exists(buf_file):
            os.remove(buf_file)
        print(ret)
        return ret

    def get_final(self):
        self._cmd("get-final")
        words = []
        while True:
            line = self._p.stdout.readline().decode()
            if line.startswith("done"):
                break
            parts = line.split(' / ')
            if line.startswith('word'):
                wd = {}
                wd['word'] = parts[0].split(': ')[1]
                wd['start'] = float(parts[1].split(': ')[1])
                wd['duration'] = float(parts[2].split(': ')[1])
                wd['phones'] = []
                words.append(wd)
            elif line.startswith('phone'):
                ph = {}
                ph['phone'] = parts[0].split(': ')[1]
                ph['duration'] = float(parts[1].split(': ')[1])
                words[-1]['phones'].append(ph)

        return words

    def stop(self):
        if not self.finished:
            self.finished = True


if __name__ == '__main__':
    import numm3
    import sys

    infile = sys.argv[1]

    k = Kaldi()

    buf = numm3.sound2np(infile, nchannels=1, R=8000)
    print('loaded_buf', len(buf))

    idx = 0
    while idx < len(buf):
        k.push_chunk(buf[idx:idx+160000].tostring())
        print(k.get_final())
        idx += 160000
