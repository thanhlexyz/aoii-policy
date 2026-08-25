# aoii-dataset: AoII-aware randomized sampling over an IRSA uplink, replayed
# on the Berkeley, BTS and SINDIT sensor traces.
PY       ?= python3
DATASETS ?= berkeley bts sindit
SOLVERS  ?= reactive random hybrid threshold grid vhct
RUN       = $(PY) main.py --no_progress_bar --device=cpu

all: run summary plots

# ---------------- data ----------------
# raw -> data/<dataset>/{<sensor_id>.csv,index.csv,min_timestamp.txt}
# destructive: each normalizer wipes data/<dataset>/ before writing.
prepare:
	$(foreach d,$(DATASETS),$(PY) main.py --scenario=normalize --dataset=$(d);)

# ---------------- trace statistics ----------------
stat_sensor:
	mkdir -p result/csv/stat
	$(foreach d,$(DATASETS),\
	  $(PY) main.py --scenario=stat_sampling_interval_per_sensor --dataset=$(d) > result/csv/stat/diff_$(d).csv & \
	  $(PY) main.py --scenario=stat_autocorr_per_sensor --dataset=$(d) > result/csv/stat/autocorr_$(d).csv & ) \
	wait

stat_dataset:
	$(foreach d,$(DATASETS),\
	  $(PY) main.py --scenario=stat_sampling_interval_per_dataset --dataset=$(d); \
	  $(PY) main.py --scenario=stat_autocorr_per_dataset --dataset=$(d);)

# ---------------- simulation ----------------
SEEDS ?= 0 1 2 3 4
run:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) & ))) \
	wait

summary:
	$(PY) main.py --scenario=stat_final

# ---------------- ablation of the proposed policy ----------------
ablation:
	$(foreach d,$(DATASETS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=vhct --seed=$(k) --vhct_descriptor=generic --tag=_generic & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=vhct --seed=$(k) --vhct_reactive=0 --tag=_norx & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=vhct --seed=$(k) --vhct_descriptor=generic --vhct_reactive=0 --tag=_generic_norx & )) \
	wait
	$(PY) main.py --scenario=stat_ablation

# ---------------- threshold baseline: offline sweep of (tau, rho) ----------------
TAUS ?= 0.02 0.1 0.5 2
RHOS ?= 0.5 1
threshold_sweep:
	$(foreach d,$(DATASETS),$(foreach t,$(TAUS),$(foreach r,$(RHOS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=threshold --seed=$(k) --threshold_tau=$(t) --threshold_rho=$(r) --tag=_t$(t)_r$(r) & )))) \
	wait
	$(PY) main.py --scenario=stat_threshold

# ---------------- figures ----------------
# writes straight into ../paper/figure/simulation/
plots:
	$(PY) main.py --scenario=make_figures

autocorr:
	$(PY) main.py --scenario=plot_autocorr

example_sensor:
	$(PY) main.py --scenario=plot_example_sensor --dataset=bts

# ---------------- over the air ----------------
PI      ?= pi5
PI_DIR  ?= ~/aoii-dataset
PI_PY   ?= ~/venv/bin/python

# software loopback of the whole radio path (no radios) -- run this first
ota-sw:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --channel=software --dataset=$(d) --solver=$(s) --seed=$(k) --tag=_sw & ))) \
	wait
	$(PY) main.py --scenario=stat_channel

ota-sw-calib:
	$(PY) main.py --scenario=ota_calib --channel=software

ota-sync:
	tar czf - --exclude='__pycache__' pysdr sdr | ssh $(PI) 'mkdir -p $(PI_DIR) && cd $(PI_DIR) && tar xzf -'

ota-calib:
	$(PY) main.py --scenario=ota_calib --channel=ota --ota_host=$(PI) --ota_dir=$(PI_DIR) --ota_python=$(PI_PY)

# pluto transmit gain (an attenuation), read off the calibration plateau
OTA_STEP     ?= 2000
OTA_POWER    ?= -25
OTA_DATASETS ?= sindit berkeley
OTA_SEEDS    ?= 0 1 2
ota:
	$(foreach d,$(OTA_DATASETS),$(foreach k,$(OTA_SEEDS),$(foreach s,$(SOLVERS),\
	  $(RUN) --scenario=solve --channel=ota --dataset=$(d) --solver=$(s) --seed=$(k) --n_step=$(OTA_STEP) --tag=_ota \
	         --ota_tx_power=$(OTA_POWER) --ota_crowd_power=$(OTA_POWER) \
	         --ota_host=$(PI) --ota_dir=$(PI_DIR) --ota_python=$(PI_PY);)))
	$(PY) main.py --scenario=stat_channel

# ---------------- housekeeping ----------------
kill:
	pkill -u $(USER) -f 'main.py' || true

clean:
	rm -rf result __pycache__ */__pycache__ */*/__pycache__

# ---------------- RQ4: sensitivity to the feedback round length ----------------
ROUNDS ?= 25 50 100 200
rq4:
	$(foreach d,$(DATASETS),$(foreach n,$(ROUNDS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=vhct --round_interval=$(n) --tag=_N$(n) --seed=$(k) & ))) \
	wait
	$(PY) main.py --scenario=stat_round_interval
