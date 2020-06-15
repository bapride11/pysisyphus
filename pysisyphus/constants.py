BOHR2M = 5.291_772_109e-11
BOHR2ANG = BOHR2M * 1e10
ANG2BOHR = 1.889726125457828
AU2J = 4.359_744_722_207_1e-18
AU2KJPERMOL = 2625.499
AU2EV = 27.211386
# eV/Å -> Hartree/Bohr
EVANG2AUBOHR = 1/AU2EV/ANG2BOHR
# fs -> Bohr * sqrt(amu/Hartree)
FS2AU = 0.9682885864793366
# Boltzman constant
KB = 1.38064852E-23  # (m² kg s⁻² K⁻¹) or just (J / K)
# Atomic mass unit to kg
AMU2KG = 1.660_539_066_60e-27
SEC2AU = 2.4188843265857e-17
# Velocity from m/s to au
AU2MPERSEC = BOHR2M / SEC2AU
MPERSEC2AU = 1 / AU2MPERSEC
# Velocity from m/s to Bohr/fs
MPERSEC2AUPERFS = 1/BOHR2M * 1e-15
# Hartree to kcal mol⁻¹
AU2KCALMOL = 627.509474
# Force to acceleration from Hartree/(Bohr*amu) -> Bohr / fs²
FORCE2ACC = AU2J / AMU2KG * 1 / BOHR2M**2 / 1e30
