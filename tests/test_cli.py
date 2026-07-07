import json

import numpy as np
from astropy.io import fits

from astro_denoise_passthrough.__main__ import main


def test_cli_protocol(tmp_path, capsys):
    image = np.random.default_rng(0).normal(100.0, 2.0, (32, 32)).astype(np.float32)
    fits.PrimaryHDU(data=image).writeto(tmp_path / "in.fits")

    rc = main(["--input", str(tmp_path / "in.fits"), "--output", str(tmp_path / "out.fits")])
    assert rc == 0
    assert (tmp_path / "out.fits").exists()
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["algorithm"] == "passthrough"
    assert status["status"] == "ok"
