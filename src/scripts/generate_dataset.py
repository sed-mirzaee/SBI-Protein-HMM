from simulator import ProteinHMMSimulator

sim = ProteinHMMSimulator(random_seed=1)
protein = sim.simulate(30)
print(protein)

# Expected output columns:
#
# - `Position`
# - `State`
# - `AminoAcid`
