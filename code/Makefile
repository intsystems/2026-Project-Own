.PHONY: quick test clean

quick:
	PYTHONPATH=src python run_experiment.py --quick --methods adamw,muon,monarch_muon,monarch_dion --output-dir outputs/quick

test:
	PYTHONPATH=src pytest -q

clean:
	rm -rf outputs .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
