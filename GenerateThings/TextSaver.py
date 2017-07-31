from GenerateThings.PDFATextGen import pdf_report_creator, get_file_name, pdf_text_closer

def SaveReport(filename, sizes, Sens, Spec, CorrectPeriod, ReportTxt):

	Name = filename

	TP = sizes[0]
	TN = sizes[1]
	FP = sizes[2]
	FN = sizes[3]

	print("\nCreating Report file...")
	#Save Report
	newLine = '\n----------------------------------------------------------------------\n'
	ReportTxt.write(newLine)
	ReportTxt.write(str(Name) + "\n")
	ReportTxt.write("Clustering Method: Agglomerative Clustering" + " g\n")
	ReportTxt.write("True Positives:    " + str(TP) + " cases....................................." + str(round(100*TP/sum(sizes),2)) + '%' + ' \n')
	ReportTxt.write("True Negatives:  " + str(TN) + " cases....................................." + str(round(100*TN/sum(sizes), 2)) + '%' + ' \n')
	ReportTxt.write("False Positives:   " + str(FP) + " cases....................................." + str(round(100*FP/sum(sizes), 2)) + '%' + ' \n')
	ReportTxt.write("False Negatives: " + str(FN) + " cases....................................." + str(round(100*FN/sum(sizes), 2)) + '%' + ' \n')
	ReportTxt.write("Sensitivity (TP/(TP+FN): " + str(round(100*Sens, 2)) + '%' + ' \n')
	ReportTxt.write("Specificity (TN/(TN+FP): " + str(round(100*Spec, 2)) + '%' + ' \n')
	ReportTxt.write("Accuracy ((TP + TN)/(Sum(ALL): " + str(round(100*CorrectPeriod, 2)) + '%')
	ReportTxt.write(newLine)
