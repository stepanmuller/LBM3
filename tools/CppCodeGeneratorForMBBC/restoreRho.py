import re
import numpy as np

def restoreRho(index, normal, fk, SCQrho, SCMfk, SCQu):
	lines = []
	normalCode = (normal[0] + 5) * 100 + (normal[1] + 5) * 10 + (normal[2] + 5)
	if index == 0:
		lines.append("__host__ __device__ void restoreRho(")
		lines.append("	const int &outerNormalX, const int &outerNormalY, const int &outerNormalZ,")
		lines.append("	BCStruct &BC,")
		lines.append("	const float (&f)[27]")
		lines.append(")")
		lines.append("{")
		lines.append("	float rho = BC.dRho + 1.f;")
		lines.append("	const float &ux = BC.ux;")
		lines.append("	const float &uy = BC.uy;")
		lines.append("	const float &uz = BC.uz;")
		lines.append("\t")
		lines.append("	const int normalCode = (outerNormalX + 5) * 100 + (outerNormalY + 5) * 10 + (outerNormalZ + 5);")
		lines.append("	if ( normalCode == " + str(normalCode) + " )")
	else:
		lines.append("	else if ( normalCode == " + str(normalCode) + " )")
	lines.append("	{")
	
	directions = ["x", "y", "z"]
	line = "		const float scqu ="
	for j, multiplier in enumerate(SCQu[0]):
		if multiplier == "0":
			continue
		elif multiplier == "1":
			line += " + u" + directions[j]
			continue
		elif multiplier == "-1":
			line += " - u" + directions[j]
			continue
		else:
			line += " + "
			line += multiplier
			line += " * u"
			line += directions[j]
	if line[-1] == "=":
		line += " 0.f"
	line += ";"
	lines.append(line)
		
	line = "		const float s = " + SCQrho[0][0] + " + scqu;"
	lines.append(line)
		
	line = "		const float scmf ="
	for j, multiplier in enumerate(SCMfk[0]):
		if multiplier == "0":
			continue
		elif multiplier == "1":
			line += " + " + fk[j]
			continue
		elif multiplier == "-1":
			line += " - " + fk[j]
			continue
		else:
			line += " + "
			line += multiplier
			line += " * "
			line += fk[j]
	if line[-1] == "=":
		line += " 0.f"
	line += ";"
	lines.append(line)
		
	line = "		rho = scmf / s;"
	lines.append(line)
	
	lines.append("	}")
	
	return lines
