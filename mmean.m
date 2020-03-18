function mm=mmean(a)
% function MMEAN(A)
% Return mean value of matrix A
%
% See also MINMAX MMAX MMIN

% D. Menemenlis (dimitri@ocean.mit.edu), 21 aug 94

ix = find(~isnan(a));
if isempty(ix)
  mm = nan;
else
  mm = mean(a(ix));
end
