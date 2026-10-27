# aoii-dataset: AoII-aware randomized sampling over an IRSA uplink, replayed
# on the Berkeley, BTS and SINDIT sensor traces.
PY       ?= python3
DATASETS ?= berkeley bts sindit synthetic
SOLVERS  ?= reactive random hybrid threshold grid vhct repair
# the proposed policy closes a feedback round every 25 slots (the per-device
# learners keep the 100-slot round of their original protocol)
EXTRA_repair = --round_interval=25
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
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) $(EXTRA_$(s)) & ))) \
	wait

summary:
	$(PY) main.py --scenario=stat_final

# every csv summary and ../paper/figure table the paper quotes, then the figures
results: summary
	$(PY) main.py --scenario=stat_ablation
	$(PY) main.py --scenario=stat_robustness
	$(PY) main.py --scenario=stat_subsets
	$(PY) main.py --scenario=stat_round_interval
	$(PY) main.py --scenario=stat_channel
	$(PY) main.py --scenario=stat_scaling
	$(PY) main.py --scenario=stat_ota_table
	$(PY) main.py --scenario=make_figures

# ---------------- ablation of the proposed policy (RQ2) ----------------
# what each ingredient buys: which parameters of the schedule are learned, the
# repeats, the background, the learner and its scale, and the round length
ABLATION = _bgonly:--repair_q_grid=0.05,--repair_n_grid=3 _qbg:--repair_n_grid=3 \
           _fixed:--repair_learn=none,--repair_q_grid=0.05,--repair_n_grid=3,--repair_bg_grid=0.01 \
           _nobg:--repair_bg_grid=0 _norepeat:--repair_n_grid=0 \
           _std:--repair_scale=std _ucb:--repair_learn=cgrid _cvhct:--repair_learn=cvhct,--repair_log=1 _device:--repair_learn=grid \
           _N100:--round_interval=100
ablation:
	$(foreach d,$(DATASETS),$(foreach k,$(SEEDS),$(foreach a,$(ABLATION),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=repair --seed=$(k) $(EXTRA_repair) \
	         $(subst $(comma), ,$(word 2,$(subst :, ,$(a)))) --tag=$(word 1,$(subst :, ,$(a))) & ))) \
	wait
	$(PY) main.py --scenario=stat_ablation

comma := ,

# ---------------- round-3 protocol checks ----------------
# causal normalization: scale from the first 10% of the run only
causal:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --causal_norm=0.1 --tag=_causal $(THRESH_$(d)) $(EXTRA_$(s)) & ))) wait

# learners frozen after 40% of the run: time-separated evaluation
frozen:
	$(foreach d,$(DATASETS),$(foreach s,grid vhct repair,$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --freeze_after=0.4 --tag=_frozen $(EXTRA_$(s)) & ))) wait

# sensitivity over epsilon, alpha, K, L, and a later time block (two seeds)
ROB_SEEDS ?= 0 1
robustness:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(ROB_SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --error_threshold=0.005 --tag=_eps0.005 $(THRESH_$(d)) $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --error_threshold=0.02 --tag=_eps0.02 $(THRESH_$(d)) $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --alpha=0.5 --tag=_alpha0.5 $(THRESH_$(d)) $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --alpha=2 --tag=_alpha2 $(THRESH_$(d)) $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --n_device=20 --tag=_K20 $(THRESH_$(d)) $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --n_device=40 --tag=_K40 $(THRESH_$(d)) $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --avg_n_irsa_replica=3 --tag=_L3 $(THRESH_$(d)) $(EXTRA_$(s)) & ))) wait
	$(foreach s,$(SOLVERS),$(foreach k,$(ROB_SEEDS),\
	  $(RUN) --scenario=solve --dataset=berkeley --solver=$(s) --seed=$(k) --start_slot=1500 --tag=_t1500 $(THRESH_berkeley) $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=sindit --solver=$(s) --seed=$(k) --start_slot=100000 --tag=_t100000 $(THRESH_sindit) $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --start_slot=20000 --tag=_t20000 $(THRESH_synthetic) $(EXTRA_$(s)) & )) wait
	$(PY) main.py --scenario=stat_robustness

# ---------------- scaling in the number of sensors ----------------
# K devices over M = K / 6 slots, so the per-slot budget is held fixed; the
# synthetic trace cycles through its 90 streams
SCALE_K     ?= 30 60 120 240
SCALE_SEEDS ?= 0 1 2
scaling:
	$(foreach K,$(SCALE_K),$(foreach s,$(SOLVERS),$(foreach k,$(SCALE_SEEDS),\
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --n_device=$(K) --n_irsa_slot=$$(( $(K) / 6 )) --tag=_scaleK$(K) $(THRESH_synthetic) $(EXTRA_$(s)) & ))) wait
	$(PY) main.py --scenario=stat_scaling

# ---------------- robustness across sensor populations ----------------
# three disjoint 30-stream subsets per trace (every stream live through the run)
SUBSET_SEEDS ?= 0 1
subsets:
	$(foreach o,0 30 60,$(foreach s,$(SOLVERS),$(foreach k,$(SUBSET_SEEDS),\
	  $(RUN) --scenario=solve --dataset=berkeley --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(THRESH_berkeley) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 150 180,$(foreach s,$(SOLVERS),$(foreach k,$(SUBSET_SEEDS),\
	  $(RUN) --scenario=solve --dataset=bts --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(THRESH_bts) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 10 20,$(foreach s,$(SOLVERS),$(foreach k,$(SUBSET_SEEDS),\
	  $(RUN) --scenario=solve --dataset=sindit --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(THRESH_sindit) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 30 60,$(foreach s,$(SOLVERS),$(foreach k,$(SUBSET_SEEDS),\
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(THRESH_synthetic) $(EXTRA_$(s)) & ))) wait
	$(PY) main.py --scenario=stat_subsets

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
THRESH_berkeley ?= --threshold_tau=0.5 --threshold_rho=1
THRESH_bts      ?= --threshold_tau=2 --threshold_rho=0.5
THRESH_sindit   ?= --threshold_tau=2 --threshold_rho=0.5
THRESH_synthetic ?= --threshold_tau=2 --threshold_rho=1
PI      ?= pi5
PI_DIR  ?= ~/aoii-dataset
PI_PY   ?= ~/venv/bin/python

# software loopback of the whole radio path (no radios) -- run this first
ota-sw:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --channel=software --dataset=$(d) --solver=$(s) --seed=$(k) --tag=_sw $(EXTRA_$(s)) & ))) \
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
	  $(RUN) --scenario=solve --channel=ota --dataset=$(d) --solver=$(s) --seed=$(k) --n_step=$(OTA_STEP) --tag=_ota $(EXTRA_$(s)) \
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
round_interval:
	$(foreach d,$(DATASETS),$(foreach n,$(ROUNDS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=repair --round_interval=$(n) --tag=_N$(n) --seed=$(k) & ))) \
	wait
	$(PY) main.py --scenario=stat_round_interval
