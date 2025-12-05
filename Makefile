# Prism Engine Makefile

PYTHON=python

# Rebuild the database from scratch
db:
	$(PYTHON) -m data.sql.rebuild

# Run DB health check
health:
	$(PYTHON) -m data.sql.health

# Run DB sanity check
sanity:
	$(PYTHON) -m data.sql.sanity

# Run full diagnostic doctor
doctor:
	$(PYTHON) -m data.sql.doctor

# Full system physical
physical:
	bash doctor.sh

