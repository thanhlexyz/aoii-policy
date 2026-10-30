PY       ?= python3
DATASETS ?= synthetic berkeley bts sindit
SOLVERS  ?= reactive random hybrid threshold ucb vhct mpgts
SEEDS    ?= 0 1 2 3 4 5 6 7 8 9
EXTRA_mpgts =
RUN       = $(PY) main.py --no_progress_bar --device=cpu
PAPER    ?= ../../paper

all: run results

prepare:
	$(foreach d,$(DATASETS),$(PY) main.py --scenario=normalize --dataset=$(d);)

stat_dataset:
	mkdir -p ../data/csv/stat
	$(foreach d,$(DATASETS),$(PY) main.py --scenario=stat_sampling_interval_per_sensor --dataset=$(d) > ../data/csv/stat/diff_$(d).csv;)
	$(foreach d,$(DATASETS),$(PY) main.py --scenario=stat_sampling_interval_per_dataset --dataset=$(d);)

run:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) $(EXTRA_$(s)) & ))) \
	wait

summary:
	$(PY) main.py --scenario=stat_final

results: summary
	$(PY) main.py --scenario=stat_ablation
	$(PY) main.py --scenario=stat_fixed
	$(PY) main.py --scenario=stat_robustness
	$(PY) main.py --scenario=stat_subset
	$(PY) main.py --scenario=stat_scaling
	$(PY) main.py --scenario=make_figure

TAUS ?= 0.02 0.1 0.5 2
RHOS ?= 0.5 1
threshold:
	$(foreach d,$(DATASETS),$(foreach t,$(TAUS),$(foreach r,$(RHOS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=threshold --seed=$(k) --threshold_tau=$(t) --threshold_rho=$(r) --tag=_tune_t$(t)_r$(r) & )))) \
	wait
	$(PY) main.py --scenario=stat_threshold

ABLATION = _bgonly:--repeat_budget_grid=5 _q:--repeat_probability_grid=0.02,0.05,0.1+--repeat_budget_grid=1,3,5 \
           _fixed:--learner=fixed+--repeat_budget_grid=5+--background_rate_grid=0.01 \
           _nobg:--background_rate_grid=0 _norepeat:--repeat_budget_grid=0 \
           _ucb:--learner=ucb_server _device:--learner=ucb_per_sensor \
           _N25:--round_length=25
ablation:
	$(foreach d,$(DATASETS),$(foreach k,$(SEEDS),$(foreach a,$(ABLATION),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=mpgts --seed=$(k) $(EXTRA_mpgts) \
	         $(subst +, ,$(word 2,$(subst :, ,$(a)))) --tag=$(word 1,$(subst :, ,$(a))) & ))) \
	wait
	$(PY) main.py --scenario=stat_ablation

FIXED_N  ?= 3 5
FIXED_BG ?= 0 0.002 0.01 0.03
fixed:
	$(foreach d,$(DATASETS),$(foreach n,$(FIXED_N),$(foreach b,$(FIXED_BG),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=mpgts --seed=$(k) --learner=fixed \
	         --repeat_budget_grid=$(n) --background_rate_grid=$(b) --tag=_fix_n$(n)_bg$(b) & )))) \
	wait
	$(PY) main.py --scenario=stat_fixed

causal:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --causal_norm=0.1 --tag=_causal $(EXTRA_$(s)) & ))) wait

frozen:
	$(foreach d,$(DATASETS),$(foreach s,ucb vhct mpgts,$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --freeze_after=0.4 --tag=_frozen $(EXTRA_$(s)) & ))) wait

robustness:
	$(foreach d,$(DATASETS),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --error_tolerance=0.005 --tag=_eps0.005 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --error_tolerance=0.02 --tag=_eps0.02 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --cost_weight=0.5 --tag=_alpha0.5 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --cost_weight=2 --tag=_alpha2 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --n_sensor=20 --tag=_K20 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --n_sensor=40 --tag=_K40 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=$(d) --solver=$(s) --seed=$(k) --avg_n_replica=3 --tag=_L3 $(EXTRA_$(s)) & ))) wait
	$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=berkeley --solver=$(s) --seed=$(k) --start_slot=1500 --tag=_t1500 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=sindit --solver=$(s) --seed=$(k) --start_slot=100000 --tag=_t100000 $(EXTRA_$(s)) & \
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --start_slot=20000 --tag=_t20000 $(EXTRA_$(s)) & )) wait
	$(PY) main.py --scenario=stat_robustness

subsets:
	$(foreach o,0 30 60,$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=berkeley --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 150 180,$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=bts --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 10 20,$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=sindit --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(EXTRA_$(s)) & ))) wait
	$(foreach o,0 30 60,$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --stream_offset=$(o) --tag=_o$(o) $(EXTRA_$(s)) & ))) wait
	$(PY) main.py --scenario=stat_subset

SCALE_K ?= 30 60 120 240
scaling:
	$(foreach K,$(SCALE_K),$(foreach s,$(SOLVERS),$(foreach k,$(SEEDS),\
	  $(RUN) --scenario=solve --dataset=synthetic --solver=$(s) --seed=$(k) --n_sensor=$(K) --n_irsa_slot=$$(( $(K) / 6 )) --tag=_scaleK$(K) $(EXTRA_$(s)) & ))) wait
	$(PY) main.py --scenario=stat_scaling

figures:
	$(PY) main.py --scenario=make_figure

paper:
	mkdir -p $(PAPER)/figure/simulation
	cp ../data/figure/tab-main.tex $(PAPER)/figure/
	cp ../data/figure/simulation/*.pdf $(PAPER)/figure/simulation/

proof:
	cd ../lean && lake build && lake env lean Check.lean

test:
	cd .. && python3 -m pytest -q test

kill:
	pkill -u $(USER) -f 'main.py' || true

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
