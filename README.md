# AoII over a grant-free IRSA uplink, replayed on real IoT sensor traces

Thirty devices share five grant-free uplink channels through one base station.
Collisions are destructive, there is no per-packet acknowledgement, and every
device picks a randomized transmission probability to keep the base station's
estimate of it both fresh and correct -- the age of incorrect information. The
ground truth each device reports is a real sensor trace (Intel Berkeley, BTS,
SINDIT) replayed as a zero-order hold, not a synthetic Markov source.

Four policies are compared: `reactive` (send whenever the local estimate is
stale), `random` (a fixed rate that maximizes IRSA throughput), `hybrid`
(reactive plus a fixed supplement sized to the leftover load budget) and
`vhct`, which learns that supplement online with one X-armed bandit per device.

The uplink itself has two backends. `ideal` counts slot occupancy. `software`
and `ota` build a real OFDM waveform, decode a CRC, and cancel what they
decoded -- in a software channel or on ADALM-Pluto radios.

## Layout

- `main.py`, `util.py` -- CLI entry point; every knob and its per-dataset default.
- `simulator/env.py` -- the slot loop: sample the sources, contend, age the estimates.
- `simulator/irsa.py` -- replication patterns and the ideal collision channel.
- `simulator/solver/` -- the four policies; `base/` holds the shared test loop.
- `simulator/normalizer/` -- raw logs to one `timestamp,value` CSV per sensor.
- `scenario/` -- what `--scenario` dispatches to: `solve`, `normalize`, `ota_calib`,
  `stat/` (trace statistics and result tables), `plot/` (figures).
- `sdr/` -- IRSA frames on a waveform: `irsa.py` (encode and cancel), `link.py`
  (radios or software), `server.py` (the JSON-lines slot server).
- `pysdr/` -- OFDM modem and Pluto drivers, from the wireless-pgd testbed.
- `ref/irsa/` -- a third-party IRSA implementation, kept for reference only.
- `data/<dataset>/` -- normalized traces (gitignored); `result/` -- csv and figures.

## Install

```bash
python3 -m pip install -r requirements.txt      # workstation
python3 -m pip install -r requirements-pi.txt   # the pi driving the radios
```

`data/` holds the normalized traces. To rebuild them from raw logs, put
`data/raw/berkeley/raw.csv` (`moteid,timestamp,temperature,humidity,light,voltage`),
`data/raw/bts/raw.csv` (`station_id,parameter_id,reading_time,value`) and
`data/raw/sindit/*.csv` (`time,value`, one file per sensor) in place and run
`make prepare`. That wipes `data/<dataset>/` first.

## Reproduce

```bash
make stat_sensor stat_dataset   # trace statistics (Table I)
make run summary                # 3 datasets x 4 policies x 5 seeds (Table II)
make rq4                        # sensitivity to the feedback round length
make plots autocorr             # figures, straight into ../paper/figure/
```

Over the air, three Plutos on a Raspberry Pi reachable as `pi5`: the device
under test (`ota_tx_uri`), the mass of the other devices (`ota_crowd_uri`) and
the base station (`ota_rx_uri`). Pointing the first two at one radio puts every
device on a single transmitter, which is what a bench with one good path
supports.

```bash
make ota-sw ota-sw-calib        # software loopback -- run this first, no radios
make ota-sync                   # push pysdr/ and sdr/ to the pi
make ota-calib                  # delivery rate against transmit gain
make ota                        # the policies, over the air
```

Transmit power means a dB SNR on the software channel and a Pluto transmit
gain (an attenuation, never positive) on the radios. `make ota-calib` is what
maps one onto the other; read the operating point off its plateau.
