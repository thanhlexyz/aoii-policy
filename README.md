# AoII over a grant-free IRSA uplink, replayed on real IoT sensor traces

Thirty devices share five grant-free uplink channels through one base station.
Collisions are destructive, there is no per-packet acknowledgement, and every
device picks a randomized transmission probability to keep the base station's
estimate of it both fresh and correct -- the age of incorrect information. The
ground truth each device reports is a real sensor trace (Intel Berkeley, BTS,
SINDIT) replayed as a zero-order hold, or the generated `synthetic` trace whose
streams encode one property each (long lockups, load, drift, correlated
bursts) with a regime change halfway.

Seven policies are compared: `reactive` (send whenever the local estimate is
stale), `random` (a fixed rate that maximizes IRSA throughput), `hybrid`
(reactive plus a fixed supplement sized to the leftover load budget),
`threshold` (send once a local AoII proxy crosses a threshold), `grid` and
`vhct` (reactive plus a uniform supplement learned per device by UCB1 or an
X-armed bandit), and `repair`, the proposed policy: reactive plus blind repair
(spread repeats of the record after every trigger) plus a background rate
the base station learns via multiple-play Thompson sampling and broadcasts each round.

The uplink itself has two backends. `ideal` counts slot occupancy. `software`
and `ota` build a real OFDM waveform, decode a CRC, and cancel what they
decoded -- in a software channel or on ADALM-Pluto radios.

## Layout

- `main.py`, `util.py` -- CLI entry point; every knob and its per-dataset default.
- `simulator/env.py` -- the slot loop: sample the sources, contend, age the estimates.
- `simulator/irsa.py` -- replication patterns and the ideal collision channel.
- `simulator/solver/` -- the seven policies; `base/` holds the shared test loop.
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
`make prepare`. That wipes `data/<dataset>/` first. The synthetic trace needs
no raw input: `python3 main.py --scenario=normalize --dataset=synthetic`
regenerates it from `simulator/normalizer/synthetic/` (seeded).

## Reproduce

```bash
make stat_sensor stat_dataset   # trace statistics
make run                        # 3 datasets x 7 policies x 5 seeds (RQ1)
make ablation                   # ingredients of the proposed policy (RQ2)
make causal frozen robustness subsets round_interval   # protocol variations (RQ3)
make scaling                    # 30 to 240 sensors on the synthetic trace
make ota-sw                     # the software waveform channel (RQ4)
make results                    # every csv summary, ../paper/figure/tab-main.tex
                                # and the RQ2-RQ4 figures under ../paper/figure/simulation/
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
