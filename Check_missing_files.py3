import os

# Parameters
directory         = "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_pp08_large_14rl_NewCooling_ANALYSIS/Pmag_over_rho_jets"
filename_template = "rho_b_xy_rho_b_xz_{:04d}.png"

start = 0
end   = 2038


# Check files
missing_files = []
for i in range(start, end + 1):
    filename = filename_template.format(i)
    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath):
        missing_files.append(filename)

# Report
if missing_files:
    print(f"Missing files ({len(missing_files)}):")
    for f in missing_files:
        print(f)
else:
    print("All files are present.")

