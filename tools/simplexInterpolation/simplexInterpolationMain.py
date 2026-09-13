"""Keep the working one-sided linear velocity; search quadratic corrections.

SymPy (exact rational algebra), not numerical SciPy fitting.

The target child is at (0,0,0); real source positions are O=(-1,-1,-1),
I+=(3,-1,-1), I-=(-5,-1,-1), and the corresponding J/K neighbors.
Thus one coarse spacing is H=4 in these coordinates. Derivative sources are
with respect to these coordinates: divide coarse-lattice gradients by 4.

The fixed L is exactly the pasted four-point one-sided reconstruction.
The correction Q vanishes at O,I+,J+,K+, so these source values are retained.
Only 18 correction coefficients are unknown.

Default observations: 9 centered second velocity differences, 9 centered
normal-derivative differences, 9 centered shear-derivative differences.
They are linear combinations of the 63 REAL sources, not new independent data.
The selector search chooses 18 independent observations, respecting all axis
permutations. This is a search within this curvature-observation family.
"""
from itertools import permutations
from collections import Counter
import sympy as sp

x, y, z = sp.symbols('x y z')
AXES = (x, y, z)
H = sp.Integer(4)
POINTS = {
    'O': (-1,-1,-1),
    'Ip': (3,-1,-1), 'Im': (-5,-1,-1),
    'Jp': (-1,3,-1), 'Jm': (-1,-5,-1),
    'Kp': (-1,-1,3), 'Km': (-1,-1,-5),
}
PLUS = ('Ip','Jp','Kp')
MINUS = ('Im','Jm','Km')

# Anchored curvature basis. Pure-square terms must include a linear subtraction:
# L already contains the curvature contribution to its one-sided slopes.
# Each basis function is zero at O and at the three positive face neighbors.
CORRECTION_TERMS = [
    ('xx', (x+1)*(x-3)),
    ('yy', (y+1)*(y-3)),
    ('zz', (z+1)*(z-3)),
    ('xy', (x+1)*(y+1)),
    ('xz', (x+1)*(z+1)),
    ('yz', (y+1)*(z+1)),
]
PRINT_COEFFICIENTS = True
PRINT_SHEAR_EXPRESSIONS = False


def build_system():
    quantities = ['ux','uy','uz','Nx','Ny','Nz','Dxy','Dxz','Dyz']
    # Keep the familiar quantity-group / point / component ordering.
    names = [f'{q}_{point}' for group in [quantities[:3],quantities[3:6],quantities[6:]]
             for point in POINTS for q in group]
    S = sp.Matrix(sp.symbols(names))
    source = dict(zip(names,S))
    L, Q, coefficients = [], [], []
    for component,prefix in zip('xyz','abc'):
        U0 = source[f'u{component}_O']
        linear = U0 + sum((source[f'u{component}_{point}']-U0)*(axis+1)/H
                          for axis,point in zip(AXES,PLUS))
        c = sp.symbols([prefix+suffix for suffix,term in CORRECTION_TERMS])
        correction = sum(ci*term for ci,(suffix,term) in zip(c,CORRECTION_TERMS))
        coefficients.extend(c)
        L.append(linear); Q.append(correction)
    C = sp.Matrix(coefficients)
    total = [L[i]+Q[i] for i in range(3)]
    derivatives = {
        'Nx': sp.diff(total[0],x), 'Ny': sp.diff(total[1],y), 'Nz': sp.diff(total[2],z),
        'Dxy': sp.diff(total[0],y)+sp.diff(total[1],x),
        'Dxz': sp.diff(total[0],z)+sp.diff(total[2],x),
        'Dyz': sp.diff(total[1],z)+sp.diff(total[2],y),
    }
    forms = dict(zip(['ux','uy','uz'],total)) | derivatives
    predicted = {f'{q}_{p}':forms[q].subs(dict(zip(AXES,coords)))
                 for p,coords in POINTS.items() for q in quantities}

    # Metadata also defines how each observation transforms under axis swaps.
    rows, rhs, metadata, labels = [], [], [], []
    for component in range(3):
        q = 'u'+'xyz'[component]
        for direction in range(3):
            p,m = PLUS[direction],MINUS[direction]
            rows.append((predicted[f'{q}_{p}']-2*predicted[f'{q}_O']+predicted[f'{q}_{m}'])/H**2)
            rhs.append((source[f'{q}_{p}']-2*source[f'{q}_O']+source[f'{q}_{m}'])/H**2)
            metadata.append(('velocity',(component,),direction))
            labels.append(f'{q}_'+'xyz'[direction]*2+' [velocity second difference]')
    for q,pair,kind in [('N'+a,(i,),'normal') for i,a in enumerate('xyz')] + [
            ('Dxy',(0,1),'shear'),('Dxz',(0,2),'shear'),('Dyz',(1,2),'shear')]:
        for direction in range(3):
            p,m = PLUS[direction],MINUS[direction]
            rows.append((predicted[f'{q}_{p}']-predicted[f'{q}_{m}'])/(2*H))
            rhs.append((source[f'{q}_{p}']-source[f'{q}_{m}'])/(2*H))
            metadata.append((kind,pair,direction))
            labels.append(q+'_'+'xyz'[direction]+' [derivative difference]')
    R = sp.Matrix(rows).applyfunc(sp.expand)
    B = sp.Matrix(rhs).jacobian(S)
    M = R.jacobian(C)
    assert R==M*C, 'The fixed linear part must cancel from curvature observations.'
    assert sp.Matrix(rhs)==B*S
    return S,C,L,Q,M,B,metadata,labels


def symmetric_selectors(metadata,n):
    transformations=[]
    for p in permutations(range(3)):
        transformations.append([metadata.index((kind,tuple(sorted(p[a] for a in pair)),p[axis]))
                                for kind,pair,axis in metadata])
    remaining=set(range(len(metadata)));orbits=[]
    while remaining:
        i=min(remaining);orbit=sorted({t[i] for t in transformations})
        orbits.append(orbit);remaining.difference_update(orbit)
    def visit(i,chosen):
        if len(chosen)>n:return
        if i==len(orbits):
            if len(chosen)==n:yield sorted(chosen)
            return
        yield from visit(i+1,chosen)
        yield from visit(i+1,chosen+orbits[i])
    return orbits,list(visit(0,[]))


def shear_tests(W,S):
    tests=[]
    for flow in range(3):
        for variation in range(3):
            if flow==variation:continue
            levels=sorted({p[variation] for p in POINTS.values()})
            U=dict(zip(levels,sp.symbols('U0:3')))
            G=dict(zip(levels,sp.symbols('G0:3')))
            shear='D'+''.join(sorted(['xyz'[flow],'xyz'[variation]]))
            samples=[]
            for name in map(str,S):
                q,p=name.split('_');level=POINTS[p][variation]
                samples.append(U[level] if q=='u'+'xyz'[flow] else G[level] if q==shear else 0)
            value=(W*sp.Matrix(samples)).applyfunc(sp.factor)
            passed=all(value[i]==0 for i in range(3) if i!=flow)
            tests.append((f'u{"xyz"[flow]}=U({"xyz"[variation]})',passed,value))
    return tests


def quadratic_sample_matrix(S):
    monomials=[1,x,y,z,x*x,y*y,z*z,x*y,x*z,y*z]
    d=sp.symbols('d0:30')
    u=[sum(d[10*i+j]*term for j,term in enumerate(monomials)) for i in range(3)]
    f=dict(zip(['ux','uy','uz'],u))
    f.update(Nx=sp.diff(u[0],x),Ny=sp.diff(u[1],y),Nz=sp.diff(u[2],z),
             Dxy=sp.diff(u[0],y)+sp.diff(u[1],x),
             Dxz=sp.diff(u[0],z)+sp.diff(u[2],x),
             Dyz=sp.diff(u[1],z)+sp.diff(u[2],y))
    samples=[]
    for source in S:
        q,p=str(source).split('_');samples.append(f[q].subs(dict(zip(AXES,POINTS[p]))))
    exact_target=sp.Matrix([u_i.subs({x:0,y:0,z:0}) for u_i in u]).jacobian(d)
    return sp.Matrix(samples).jacobian(d),exact_target


def main():
    S,C,L,Q,M,B,metadata,labels=build_system()
    orbits,selectors=symmetric_selectors(metadata,len(C))
    target={x:0,y:0,z:0}
    linear_map=sp.Matrix([v.subs(target) for v in L]).jacobian(S)
    correction_evaluation=sp.Matrix([v.subs(target) for v in Q]).jacobian(C)
    polynomial_samples,exact_target=quadratic_sample_matrix(S)
    print('Independent real sources:',len(S))
    print('Correction coefficients:',list(C))
    print('Correction basis:',CORRECTION_TERMS)
    print('Fixed linear target:',sp.Matrix([v.subs(target) for v in L]))
    print('Observation matrix shape/rank:',M.shape,M.rank())
    print('\nAvailable curvature observations:')
    for i,label in enumerate(labels):print(f'  {i:2}: {label}')
    print('Row orbits:',orbits)
    print('Symmetric candidate selectors:',len(selectors))
    ranks=Counter();results=[]
    for sel in selectors:
        A=M.extract(sel,range(len(C)));rank=A.rank();ranks[rank]+=1
        if rank!=len(C):continue
        coefficient_map=A.inv()*B.extract(sel,range(len(S)))
        assert A*coefficient_map==B.extract(sel,range(len(S)))
        W=linear_map+correction_evaluation*coefficient_map
        assert W*polynomial_samples==exact_target
        tests=shear_tests(W,S)
        passed=all(t[1] for t in tests)
        matches=[i for i in range(M.rows) if (M*coefficient_map)[i,:]==B[i,:]]
        print('\nINVERTIBLE selector:',sel)
        print('Selected observations:',[labels[i] for i in sel])
        print('Matched curvature observations:',matches)
        print('Quadratic reproduction: PASS')
        for name,ok,value in tests:
            print(name,':','PASS' if ok else 'FAIL')
            if PRINT_SHEAR_EXPRESSIONS:print('  target velocity:',list(value))
        if PRINT_COEFFICIENTS:
            print('Correction coefficients:')
            for c,value in zip(C,coefficient_map*S):print(' ',c,'=',sp.factor(value))
        results.append({'selector':sel,'passes_shear':passed,'W':W,'coefficient_map':coefficient_map})
    print('\nSUMMARY')
    print('Candidates:',len(selectors))
    print('Rank counts:',dict(sorted(ranks.items())))
    print('Invertible:',len(results))
    print('Pass all six general shear tests:',sum(r['passes_shear'] for r in results))
    return results

if __name__=='__main__':main()
