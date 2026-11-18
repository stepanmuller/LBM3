def restoreFaceUxUyUz(normal, cx, cy, cz):
	"""Emit face velocity restoration; return no replacement for edges/corners.

	Uses equations (2.43), (2.44), and (2.52) in LBM (20), section 2.2.3.
	The caller supplies rho and writable ux, uy, uz. Populations must be the
	full (weight-compensated) f, as in the existing MBBC restoration routines.
	Only c.n = 0 and c.n = +1 populations are read. The formula assumes a
	positive density and a nonzero tangential denominator; no clipping or
	additional closure is introduced.
	"""
	if sum(abs(component) for component in normal) != 1:
		return []

	velocities = (cx, cy, cz)
	normalAxis = next(axis for axis in range(3) if normal[axis] != 0)
	cn = [sum(normal[axis] * velocities[axis][i] for axis in range(3))
		  for i in range(len(cx))]
	zero = [i for i, value in enumerate(cn) if value == 0]
	plus = [i for i, value in enumerate(cn) if value == 1]

	def populationSum(indices, coefficients=None):
		terms = []
		for i in indices:
			coefficient = 1 if coefficients is None else coefficients[i]
			if coefficient == 1:
				terms.append("+ f[" + str(i) + "]")
			elif coefficient == -1:
				terms.append("- f[" + str(i) + "]")
		return " ".join(terms) if terms else "0.f"

	lines = [
		"\t\t// Face restoration: rhoZ = sum(c.n=0) f + 2 sum(c.n=+1) f.",
		"\t\tconst float fZero = " + populationSum(zero) + ";",
		"\t\tconst float fPlus = " + populationSum(plus) + ";",
		"\t\tconst float uNormal = (fZero + 2.f * fPlus) / rho - 1.f;",
		"\t\tu" + "xyz"[normalAxis] + " = "
			+ ("" if normal[normalAxis] > 0 else "-") + "uNormal;",
		"\t\t// Set k_nnt = 0 for each tangent (LBM (20), Eq. 2.52).",
		"\t\tconst float tangentDenominator = fZero + 2.f * (uNormal + 1.f) * rho * uNormal;",
	]
	for axis in range(3):
		if axis == normalAxis:
			continue
		tangent = "xyz"[axis]
		lines.append("\t\tconst float jZero" + tangent + " = "
			+ populationSum(zero, velocities[axis]) + ";")
		lines.append("\t\tconst float jPlus" + tangent + " = "
			+ populationSum(plus, velocities[axis]) + ";")
		lines.append("\t\tu" + tangent + " = ((2.f * uNormal + 1.f) * jZero"
			+ tangent + " + 4.f * uNormal * jPlus" + tangent
			+ ") / tangentDenominator;")
	return lines
