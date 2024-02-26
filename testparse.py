import argparse as ap

p = ap.ArgumentParser(description="Run model comparison")
p.add_argument('-p', action='store')
args = p.parse_args()
print(args.p)
