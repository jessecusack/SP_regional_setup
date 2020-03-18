function mm=mmin(a)
% function MMAX(A)
% Return maximum value of matrix A
%
% See also MINMAX MMIN MMEAN

% D. Menemenlis (dimitri@ocean.mit.edu), 21 aug 94

ix=~isnan(a);
mm=min(a(ix));
