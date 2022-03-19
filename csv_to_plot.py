import argparse
import pandas as pd
import plotly.express as px

def get_options(args=None):
	parser = argparse.ArgumentParser(description="")
	# Data
	parser.add_argument('--path', type=str, help="The path to the csv file")
	parser.add_argument('--xlabel', type=str, default='Step', help="The column used as x variable")
	parser.add_argument('--ylabel', type=str, default='Value', help="The column used as y variable")
	parser.add_argument('--width', type=int, default=1600, help='The width of the plot')
	parser.add_argument('--height', type=int, default=900, help='The height of the plot')
	parser.add_argument('--figname', type=str, default=None, help='The name of the output figure')
	opts = parser.parse_args(args)
	return opts

if __name__ == "__main__":
	opts = get_options()
	print(opts)
	df = pd.read_csv(opts.path)
	df[opts.ylabel] = -df[opts.ylabel]

	fig = px.line(df, x = opts.xlabel, y = opts.ylabel, title=opts.figname, width=opts.width, height=opts.height)
	fig.update_yaxes(rangemode="tozero")
	fig.write_image(f"{opts.figname}.pdf")

# Example usage: python csv_to_plot.py --path example.csv --figname example