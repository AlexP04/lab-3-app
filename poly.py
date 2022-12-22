#Python import
import numpy as np
import re

#Other packages
from solve import Solve
from basis import *
 
#Class (private) to store polynom in and 
class _Polynom(object):
    ###Constructor
    def __init__(self, coeficients, symbol='x', subscribe=None, accuracy=0.000001):
        self.coeficients = coeficients
        self.symbol = symbol
        self.subscribe = subscribe
        self.accuracy = accuracy
        
    #Show polynom as string :: private
    def __print__(self):
        result = []
        for degree, c in reversed(list(enumerate(self.coeficients))):
            
            if len(result) == 0:
                if c < 0:
                    sign = '-'
                else:
                    sign = ''
            else:
                if c < 0:
                    sign = ' - '
                else:
                    sign =  ' + '
          
            c  = abs(c)
            
            if c < self.accuracy:
                continue
                
            if c == 1 and degree != 0:
                c = ''

            f = {0: '{}{:f}', 1: '{}{:f}'+self.symbol}.get(degree, '{}{:f}' + self.symbol + '^{{{}}}')
            res = f.format(sign, c, degree)
            res = res.replace(self.symbol, r' x_{{{}}}'.format(self.subscribe))
            result.append(res)
            
        return ''.join(result)
    
    
#Class, that builds output for user of polynoms created
class Builder(object):
    ###Constructor
    def __init__(self, solution):   
        
        self._solution = solution
        degree = max(solution.degree) - 1
        
        if solution.polynomial_type == 'Chebyshev':
            self.symbol = 'T'
            self.basis = basis(degree,mode = 'chebyshev')
        elif solution.polynomial_type == 'Chebyshev shifted':
            self.symbol = 'U^*'
            self.basis =  basis(degree,mode = 'chebyshev shifted')
        elif solution.polynomial_type == 'Sinus based':
            self.symbol = 'sin'
        elif solution.polynomial_type == 'Cosinus based':
            self.symbol = 'cos'
        
        self.fmode = self._solution.fmode
#         try:
#             self.basis = basis(degree,mode = solution.polynomial_type)
#         except:
#             self.basis = []
        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).ravel() for X in solution.X_]
        self.maxX = [X.max(axis=0).ravel() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).ravel()
        self.maxY = solution.Y_.max(axis=0).ravel()
        
        #specifying function to use
        if self._solution.fmode == 1:
            self.func = self._solution.func
            self.func_inv = self._solution.func_inv
    
    #Standartize coeficients for polynom :: private
    def __standardtize__(self, c):
        std_coeffs = np.zeros(c.shape)
        for index in range(c.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(c.shape)
            if type(c) is np.matrix:
                std_coeffs += c[index].getA1() * cp[0]
            else:
                std_coeffs += c[index] * cp
        return std_coeffs.squeeze()
    
    #Find lamdas (lowest level aggregation) for each and every X_i to summarize functions further:: private
    def __compose_lambdas__(self):
        self.lvl1 = list()
        for i in range(self._solution.Y.shape[1]):
            current_1 = list()
            shift = 0
            for j in range(3): 
                current_2 = list()
                for k in range(self._solution.dim[j]):
                    current_3 = self._solution.L[shift:shift + self._solution.degree[j], i].ravel()
                    shift += self._solution.degree[j]
                    current_2.append(current_3)
                current_1.append(current_2)
            self.lvl1.append(current_1)
    
    #Print first-level aggregation results as a string o use further with different depths :: private
    def __print_1__(self, mode = 1, i=0, j=0, k=0):
        texts = list()
        
        if self.fmode==1:
            # for trygonometry functions
            if self.symbol == 'cos' or self.symbol == 'sin':
                if mode == 1:
                    for n in range(len(self.lvl1[i][j][k])):
                        texts.append(r'(1 + \mathrm{{{func}}}(2 \pi \cdot \{symbol}({deg} \cdot x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                            self.lvl1[i][j][k][n], 
                            j+1, k+1, deg=n, symbol = self.symbol, func=self.func
                        ))

                elif mode == 2:
                    for k in range(len(self.lvl1[i][j])):
                        shift = sum(self._solution.dim[:j]) + k
                        for n in range(len(self.lvl1[i][j][k])):
                            texts.append(r'(1 + \mathrm{{{func}}}(2 \pi \cdot \{symbol}({deg} \cdot x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                                self.a[i][shift] * self.lvl1[i][j][k][n],
                                j+1, k+1, deg=n, symbol = self.symbol, func=self.func
                            ))

                else:
                    for j in range(3):
                        for k in range(len(self.lvl1[i][j])):
                            shift = sum(self._solution.dim[:j]) + k
                            for n in range(len(self.lvl1[i][j][k])):
                                texts.append(r'(1 + \mathrm{{{func}}}(2 \pi \cdot \{symbol}({deg} \cdot x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                                    self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k][n],
                                    j + 1, k + 1, deg=n, symbol = self.symbol, func=self.func
                                ))
            else:
                if mode == 1:
                    for n in range(len(self.lvl1[i][j][k])):
                        texts.append(r'(1 + \mathrm{{{func}}}({symbol}_{{{deg}}}(x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                            self.lvl1[i][j][k][n], 
                            j+1, k+1, deg=n,symbol = self.symbol, func=self.func
                        ))

                elif mode == 2:
                    for k in range(len(self.lvl1[i][j])):
                        shift = sum(self._solution.dim[:j]) + k
                        for n in range(len(self.lvl1[i][j][k])):
                            texts.append(r'(1 + \mathrm{{{func}}}({symbol}_{{{deg}}}(x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                                self.a[i][shift] * self.lvl1[i][j][k][n],
                                j+1, k+1, deg=n, symbol = self.symbol, func=self.func
                            ))

                else:
                    for j in range(3):
                        for k in range(len(self.lvl1[i][j])):
                            shift = sum(self._solution.dim[:j]) + k
                            for n in range(len(self.lvl1[i][j][k])):
                                texts.append(r'(1 + \mathrm{{{func}}}({symbol}_{{{deg}}}(x_{{{1}{2}}})))^{{{0:.6f}}}'.format(
                                    self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k][n],
                                    j + 1, k + 1, deg=n, symbol = self.symbol, func=self.func
                                ))
                
        else:
            if self.symbol == 'cos' or self.symbol == 'sin':
                if mode == 1:
                    for n in range(len(self.lvl1[i][j][k])):
                        texts.append(r'(1 + \{symbol}(2 \pi \cdot {deg}\cdot x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                            self.lvl1[i][j][k][n], 
                            j+1, k+1,symbol = self.symbol, deg=n
                        ))

                elif mode == 2:
                    for k in range(len(self.lvl1[i][j])):
                        shift = sum(self._solution.dim[:j]) + k
                        for n in range(len(self.lvl1[i][j][k])):
                            texts.append(r'(1 + \{symbol}(2 \pi \cdot {deg}\cdot x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                                self.a[i][shift] * self.lvl1[i][j][k][n],
                                j+1, k+1, deg=n, symbol = self.symbol
                            ))

                else:
                    for j in range(3):
                        for k in range(len(self.lvl1[i][j])):
                            shift = sum(self._solution.dim[:j]) + k
                            for n in range(len(self.lvl1[i][j][k])):
                                texts.append(r'(1 + \{symbol}(2 \pi \cdot {deg}\cdot x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                                    self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k][n],
                                    j + 1, k + 1, deg=n, symbol = self.symbol
                                ))
            else:
                if mode == 1:
                    for n in range(len(self.lvl1[i][j][k])):
                        texts.append(r'(1 + {symbol}_{{{deg}}}(x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                            self.lvl1[i][j][k][n], 
                            j+1, k+1, deg=n, symbol = self.symbol
                        ))

                elif mode == 2:
                    for k in range(len(self.lvl1[i][j])):
                        shift = sum(self._solution.dim[:j]) + k
                        for n in range(len(self.lvl1[i][j][k])):
                            texts.append(r'(1 + {symbol}_{{{deg}}}(x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                                self.a[i][shift] * self.lvl1[i][j][k][n],
                                j+1, k+1, deg=n, symbol = self.symbol
                            ))

                else:
                    for j in range(3):
                        for k in range(len(self.lvl1[i][j])):
                            shift = sum(self._solution.dim[:j]) + k
                            for n in range(len(self.lvl1[i][j][k])):
                                texts.append(r'(1 + {symbol}_{{{deg}}}(x_{{{1}{2}}}))^{{{0:.6f}}}'.format(
                                    self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k][n],
                                    j + 1, k + 1, deg=n, symbol = self.symbol
                                ))
            
                        
        res = ' + '.join(texts).replace('+ -', ' -')
        return res

    #Prints F-function in special form :: private
    def __print_final_1__(self, i):
        texts = list()
        for j in range(3):
            for k in range(len(self.lvl1[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                raw_coeffs = self.__standardtize__(self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k])
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = np.poly1d([1 / diff, - self.minX[j][k]] / diff)
                add_poly = np.poly1d([1])
                current_poly = np.poly1d([0])
                for n in range(len(raw_coeffs)):
                    current_poly += add_poly * raw_coeffs[n]
                    add_poly *= mult_poly
                 
                
                current_poly = current_poly * (self.maxY[i] - self.minY[i]) + self.minY[i]
                current_poly = np.poly1d(current_poly.coeffs, variable='(x_{0}{1})'.format(j+1, k+1))
                
                texts.append(str(_Polynom(
                    current_poly, 
                    symbol='(x_{0}{1})'.format(j+1, k+1),
                    subscribe='{0}{1}'.format(j+1, k+1)).__print__()))
                
        res = ' + '.join(texts).replace('+ -', '- ')
        return res
        

    #Prints F-function in special form (just another form) :: private
    def __print_2__(self, i):
        
        texts = list()
        for j in range(3):
            for k in range(len(self.lvl1[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                current_polynom = np.poly1d(self.__standardtize__(self.c[i][j] * self.a[i][shift] *
                                                                     self.lvl1[i][j][k])[::-1],
                                         variable='(x_{0}{1})'.format(j+1, k+1))
                texts.append(str(_Polynom(
                    current_polynom, 
                    symbol="(x_"+str(j+1)+str(k+1)+")".format(j+1, k+1),
                    subscribe='{0}{1}'.format(j+1, k+1)).__print__()) )
        res = ' + '.join(texts).replace('+ -', '- ')
        return res
    
    # Prints F-function in special form (just another form) :: private
    def __print_final_2__(self, i):
        if self.fmode == 1:
            res = []
            for j in range(3):
                coef = self.c[i][j]
                res.append(f'(1 + \\mathrm{{{self.func}}}(\\Phi_{{{i+1}{j+1}}} (x_{j+1})))^{{{coef:.6f}}}')
            
        else:
            res = ''
            for j in range(3):
                coef = self.c[i][j]
                if coef >= 0:
                    res += f'(1 + \\Phi_{{{i+1}{j+1}}} (x_{j+1}))^{{{coef:.6f}}}'
                else:
                    res += f'(1 + \\Phi_{{{i+1}{j+1}}} (x_{j+1}))^{{{coef:.6f}}}'
            if self.c[i][0] >= 0:
                res = res[2:-1]
            else:
                res = res[:-1]
        return '\cdot'.join(res) + ' - 1'
    
    # Method to get refined result, generates final string of result :: public
    def get_results(self):
        self.__compose_lambdas__()
        if self.fmode == 1:
            lvl1_texts = [r'$\Psi^{{{0}}}_{{[{1},{2}]}}+1 = \mathrm{{{inv_func}}}[{result}]$'.format(i + 1, j + 1, k + 1, inv_func=self.func_inv,result=self.__print_1__(1, i, j, k)) + '\n' for i in range(self._solution.Y.shape[1]) for j in range(3) for k in range(self._solution.dim[j])]
            lvl2_texts = [r'$\Phi_{{{0}{1}}}+1 = \mathrm{{{inv_func}}}[{result} ]$'.format(i + 1, j + 1, inv_func=self.func_inv, result=self.__print_1__(2, i, j)) + '\n' for i in range(self._solution.Y.shape[1]) for j in range(3)]
            f_texts = [r'$\Phi_{{{0}}} + 1 = \mathrm{{{inv_func}}}[{result}]$'.format(i + 1, inv_func=self.func_inv, result=self.__print_1__(3, i)) + '\n' for i in range(self._solution.Y.shape[1])]
            f_texts_l = [r'$\Phi_{i}(x_1, x_2, x_3) = \mathrm{{{inv_func}}}[{result}]$'.format(i=i+1, inv_func=self.func_inv, result=self.__print_final_2__(i)) + '\n'  for i in range(self._solution.Y.shape[1])]

            res = [r'$\Phi_i$ derived from $\Phi_{i1}(x_1)$, $\Phi_{i2}(x_2)$, $\Phi_{i3}(x_3)$:' + '\n'] + f_texts_l + [r'$\Phi_i$:' + '\n'] + f_texts + [r'$\Phi_{ik}$:' + '\n'] + lvl2_texts + [r'$\Psi$:' + '\n'] + lvl1_texts 
        else:
            lvl1_texts = [r'$\Psi_{{{1}{2}}}^{{[{0}]}}(x_{{{1}{2}}}) = {result}$'.format(i+1, j+1, k+1, result=self.__print_1__(1, i, j, k)) + '\n' for i in range(self._solution.Y.shape[1]) for j in range(3) for k in range(self._solution.dim[j])]

            lvl2_texts = [r'$\Phi_{{{0}{1}}}(x_{{{1}}}) = {result}$'.format(i+1, j+1, result=self.__print_1__(2, i, j)) + '\n'
                           for i in range(self._solution.Y.shape[1])
                           for j in range(3)]
            f_texts = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1, result=self.__print_1__(3, i)) + '\n'
                         for i in range(self._solution.Y.shape[1])]
#             f_texts_t = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1,result=self.__print_final_1__(i)) + '\n' for i in range(self._solution.Y.shape[1])]
            
#             f_texts_td = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(
#                                                 i+1, result=self.__print_2__(i)) + '\n'
#                                                 for i in range(self._solution.Y.shape[1])]
            f_texts_l = [r'$\Phi_{i}(x_1, x_2, x_3) = {result}$'.format(i=i+1, result=self.__print_final_2__(i)) + '\n' 
                                    for i in range(self._solution.Y.shape[1])]
            res = [r'$\Phi_i$ derived from $\Phi_{i1}(x_1)$, $\Phi_{i2}(x_2)$, $\Phi_{i3}(x_3)$:' + '\n'] + f_texts_l + [r'$\Phi_i$:' + '\n'] + f_texts + [r'$\Phi_{ik}$:' + '\n'] + lvl2_texts + [r'$\Psi$:' + '\n'] + lvl1_texts 
        
        return '\n'.join(
           res
        )
