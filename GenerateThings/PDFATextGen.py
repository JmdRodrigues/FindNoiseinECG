from matplotlib.backends.backend_pdf import PdfPages
import os

def get_file_name(folder):

	file_name = folder.split('/')[-2]

	return file_name

def pdf_report_creator(file_path, filename):

	# Open files to save imgs and txt
	print(file_path + "/" + filename)
	PDFName = filename + '.pdf'
	pp = PdfPages(file_path + "/" + PDFName[:-4] + "/" + PDFName)
	TXTName = PDFName[:-4] + '.txt'
	REPORTfile = open(file_path + "/" + TXTName[:-4] + "/" + TXTName, "w")  # REPORT FILE

	return pp, REPORTfile

def pdf_text_closer(REPORTfile, pp):

	# Close report and graphics PDF
	REPORTfile.close()
	pp.close()