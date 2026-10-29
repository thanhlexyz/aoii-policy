# AoII-aware status updating over a feedback-free IRSA uplink, replayed on the
# Berkeley, BTS, SINDIT and Synthetic traces. Run every target from core/;
# inputs are read from ../data/data, results go to ../data/csv and figures to
# ../data/figure (`make paper` copies the generated ones into the paper).
PY       ?= python3
DATASETS ?= synthetic berkeley bts sindit
SOLVERS  ?= reactive random hybrid threshold ucb vhct mpgts
SEEDS    ?= 0 1 2 3 4 5 6 7 8 9
# every learner closes a feedback round every 100 slots (the default); the
# 25-slot round of the proposed policy is the `_N25` ablation
EXTRA_mpgts =
RUN       = $(PY) main.py --no_progress_bar --device=cpu
PAPER    ?= ../../paper

all: run results

# ---------------- data ----------------
# raw -> ../data/data/<dataset>/{<sensor_id>.csv,index.csv,min_timestamp.txt}
# destructive: each normalizer wipes ../data/data/<dataset>/ before writing.
prepare:
	$(foreach d,$(DATASETS),$(PY) main.py --scenario=normalize --dataset=$(d);)

# median inter-event time per trace, which sets the slot interval (Sec. IV-A):
# the per-sensor intervals into ../data/csv/stat/, then their spread per trace
stat_dataset:
	mkdir -p ../data/csv/stat
	$(foreach d,$(DATASETS),$(PY) main.py --scenario=stat_sampling_interval_per_sensor --dataset=$(d) > ../data/csv/stat/diff_$(d).csv;)
	$(foreach d,$(DATASETS),$(PY) main.py --scenario=stat_sampling_interval_per_dataset --dataset=$(d);)

# ---------------- RQ1: Table I ----------------
run:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) $(EXTRA_$(s)) & ))) \
	wait

summary:
	$(PY) main.py --scenario=stat_final

# every csv summary and table the paper quotes, then the figures
results: summary
	$(PY) main.py --scenario=stat_ablation
	$(PY) main.py --scenario=stat_fixed
	$(PY) main.py --scenario=stat_robustness
	$(PY) main.py --scenario=stat_subsets
	$(PY) main.py --scenario=stat_channel
	$(PY) main.py --scenario=stat_scaling
	$(PY) main.py --scenario=make_figures

# ---------------- Threshold baseline: one (tau, rho) for every trace ----------------
# each pair of the grid on every trace and seed; stat_threshold ranks the pairs
# on the first 40% of the run and picks the one with the best mean rank over
# the traces, which is the default of --threshold_tau and --threshold_rho
TAUS ?= 0.02 0.1 0.5 2
RHOS ?= 0.5 1
threshold:
	$(foreach d,$(DATASETS),$(foreach t,$(TAUS),$(foreach r,$(RHOS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=threshold --seed=$(k) --threshold_tau=$(t) --threshold_rho=$(r) --tag=_tune_t$(t)_r$(r) & )))) \
	wait
	$(PY) main.py --scenario=stat_threshold

# ---------------- RQ2: Fig. 4 and the fixed-schedule reference ----------------
# what each ingredient buys: which parameters of the schedule are learned (the
# `_q` variant is the 36-schedule grid that learns q as well), the repeats,
# the background, the learner, and the round length; flags of
# one variant are joined by '+'
ABLATION = _bgonly:--mpgts_n_grid=5 _q:--mpgts_q_grid=0.02,0.05,0.1+--mpgts_n_grid=1,3,5 \
           _fixed:--mpgts_learn=none+--mpgts_n_grid=5+--mpgts_bg_grid=0.01 \
           _nobg:--mpgts_bg_grid=0 _norepeat:--mpgts_n_grid=0 \
           _ucb:--mpgts_learn=cucb _device:--mpgts_learn=ucb \
           _N25:--round_interval=25
ablation:
	$(foreach d,$(DATASETS),$(foreach k,$(SEEDS),$(foreach a,$(ABLATION),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=mpgts --seed=$(k) $(EXTRA_mpgts) \
	         $(subst +, ,$(word 2,$(subst :, ,$(a)))) --tag=$(word 1,$(subst :, ,$(a))) & ))) \
	wait
	$(PY) main.py --scenario=stat_ablation

# hindsight reference: each candidate schedule of the proposed policy fixed for
# the whole run (the learner off); stat_fixed ranks them per trace
FIXED_N  ?= 3 5
FIXED_BG ?= 0 0.002 0.01 0.03
fixed:
	$(foreach d,$(DATASETS),$(foreach n,$(FIXED_N),$(foreach b,$(FIXED_BG),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=mpgts --seed=$(k) --mpgts_learn=none \
	         --mpgts_n_grid=$(n) --mpgts_bg_grid=$(b) --tag=_fix_n$(n)_bg$(b) & )))) \
	wait
	$(PY) main.py --scenario=stat_fixed

# ---------------- RQ3: robustness ----------------
# causal normalization: scale from the first 10% of the run only
causal:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --causal_norm=0.1 --tag=_causal $(EXTRA_$(s)) & ))) wait

# learners frozen after 40% of the run: time-separated evaluation
frozen:
	$(foreach d,$(DATASETS),$(foreach s,ucb vhct mpgts,$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --freeze_after=0.4 --tag=_frozen $(EXTRA_$(s)) & ))) wait

# sensitivity over epsilon, alpha, K, L, and a later time block
robustness:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --error_threshold=0.005 --tag=_eps0.005 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --error_threshold=0.02 --tag=_eps0.02 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --alpha=0.5 --tag=_alpha0.5 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --alpha=2 --tag=_alpha2 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --n_device=20 --tag=_K20 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --n_device=40 --tag=_K40 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --avg_n_irsa_replica=3 --tag=_L3 $(EXTRA_$(s)) & ))) wait
	$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=berkeley --solver=$(s) --seed=$(k) --start_slot=1500 --tag=_t1500 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=sindit --solver=$(s) --seed=$(k) --start_slot=100000 --tag=_t100000 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --start_slot=20000 --tag=_t20000 $(EXTRA_$(s)) & )) wait
	$(PY) main.py --scenario=stat_robustness

# three disjoint 30-stream subsets per trace (every stream live through the run)
subsets:
	$(foreach o,0 30 60,$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=berkeley --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 150 180,$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=bts --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 10 20,$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=sindit --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 30 60,$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(EXTRA_$(s)) & ))) wait
	$(PY) main.py --scenario=stat_subsets

# ---------------- RQ3: Fig. 5, scaling in the number of sensors ----------------
# K devices over M = K / 6 slots, so the per-slot budget is held fixed; the
# synthetic trace cycles through its 90 streams
SCALE_K ?= 30 60 120 240
scaling:
	$(foreach K,$(SCALE_K),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --n_device=$(K) --n_irsa_slot=$$(( $(K) / 6 )) --tag=_scaleK$(K) $(EXTRA_$(s)) & ))) wait
	$(PY) main.py --scenario=stat_scaling

# ---------------- RQ4: Fig. 6, over the air ----------------
PI      ?= pi5
PI_DIR  ?= ~/aoii-dataset
PI_PY   ?= ~/venv/bin/python

# no-radio check of the radio path: the same frames through a software channel
ota-sw-calib:
	$(PY) main.py --scenario=ota_calib --channel=software

# copy the radio-side code to the Raspberry Pi that drives the radios
ota-sync:
	tar czf - --exclude='__pycache__' pysdr sdr | ssh $(PI) 'mkdir -p $(PI_DIR) && cd $(PI_DIR) && tar xzf -'

ota-calib:
	$(PY) main.py --scenario=ota_calib --channel=ota --ota_host=$(PI) --ota_dir=$(PI_DIR) --ota_python=$(PI_PY)

# pluto transmit gain (an attenuation), read off the calibration plateau
OTA_STEP     ?= 2000
OTA_POWER    ?= -25
OTA_DATASETS ?= berkeley bts sindit
OTA_SEEDS    ?= 0 1 2 3 4
ota:
	$(foreach d,$(OTA_DATASETS),$(foreach k,$(OTA_SEEDS),$(foreach s,$(SOLVERS),\
	  $(RUN) --scenario=solve --channel=ota --dataset=$(d) --solver=$(s) --seed=$(k) --n_step=$(OTA_STEP) --tag=_ota $(EXTRA_$(s)) \
	         --ota_tx_power=$(OTA_POWER) --ota_crowd_power=$(OTA_POWER) \
	         --ota_host=$(PI) --ota_dir=$(PI_DIR) --ota_python=$(PI_PY);)))
	$(PY) main.py --scenario=stat_channel

# ---------------- figures and the paper ----------------
# Fig. 4-6 and Table I into ../data/figure
figures:
	$(PY) main.py --scenario=make_figures

# copy the generated table and figures into the paper (the hand-drawn TikZ
# figures of the paper are left alone)
paper:
	mkdir -p $(PAPER)/figure/simulation
	cp ../data/figure/tab-main.tex $(PAPER)/figure/
	cp ../data/figure/simulation/*.pdf $(PAPER)/figure/simulation/

# ---------------- Theorem 1 ----------------
# the Lean development behind Theorem 1: builds and prints the axioms of every
# theorem (only propext, Classical.choice and Quot.sound)
proof:
	cd ../lean && lake build && lake env lean Check.lean

# ---------------- housekeeping ----------------
kill:
	pkill -u $(USER) -f 'main.py' || true

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
