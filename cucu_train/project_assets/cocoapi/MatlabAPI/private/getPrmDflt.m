function varargout = getPrmDflt( prm, dfs, checkExtra )
% Helper to set default values (if not already set) of parameter struct.
%
% Takes input parameters and a list of 'name'/default pairs, and for each
% 'name' for which prm has no value (prm.(name) is not a field or 'name'
% does not appear in prm list), getPrmDflt assigns the given default
% value. If default value for variable 'name' is 'REQ', and value for
% 'name' is not given, an error is thrown. See below for usage details.
%
% USAGE (nargout==1)
%  prm = getPrmDflt( prm, dfs, [checkExtra] )
%
% USAGE (nargout>1)
%  [ param1 ... paramN ] = getPrmDflt( prm, dfs, [checkExtra] )
%
% INPUTS
%  prm          - param struct or cell of form {'name1' v1 'name2' v2 ...}
%  dfs          - cell of form {'name1' def1 'name2' def2 ...}
%  checkExtra   - [0] if 1 throw error if prm contains params not in dfs
%                 if -1 if prm contains params not in dfs adds them
%
% OUTPUTS (nargout==1)
%  prm    - parameter struct with fields 'name1' through 'nameN' assigned
%
% OUTPUTS (nargout>1)
%  param1 - value assigned to parameter with 'name1'
%   ...
%  paramN - value assigned to parameter with 'nameN'
%
% EXAMPLE
%  dfs = { 'x','REQ', 'y',0, 'z',[], 'eps',1e-3 };
%  prm = getPrmDflt( struct('x',1,'y',1), dfs )
%  [ x y z eps ] = getPrmDflt( {'x',2,'y',1}, dfs )
%
% See also INPUTPARSER
%
% Piotr's Computer Vision Matlab Toolbox      Version 2.60
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

if( mod(length(dfs),2) ), error('odd number of default parameters'); end
if nargin<=2, checkExtra = 0; end

% get the input parameters as two cell arrays: prmVal and prmField
if iscell(prm) && length(prm)==1, prm=prm{1}; end
if iscell(prm)
  if(mod(length(prm),2)), error('odd number of parameters in prm'); end
  prmField = prm(1:2:end); prmVal = prm(2:2:end);
else
  if(~isstruct(prm)), error('prm must be a struct or a cell'); end
  prmVal = struct2cell(prm); prmField = fieldnames(prm);
end

% get and update default values using quick for loop
dfsField = dfs(1:2:end); dfsVal = dfs(2:2:end);
if checkExtra>0
  for i=1:length(prmField)
    j = find(strcmp(prmField{i},dfsField));
    if isempty(j), error('parameter %s is not valid', prmField{i}); end
    dfsVal(j) = prmVal(i);
  end
elseif checkExtra<0
  for i=1:length(prmField)
    j = find(strcmp(prmField{i},dfsField));
    if isempty(j), j=length(dfsVal)+1; dfsField{j}=prmField{i}; end
    dfsVal(j) = prmVal(i);
  end
else
  for i=1:length(prmField)
    dfsVal(strcmp(prmField{i},dfsField)) = prmVal(i);
  end
end

% check for missing values
if any(strcmp('REQ',dfsVal))
  cmpArray = find(strcmp('REQ',dfsVal));
  error(['Required field ''' dfsField{cmpArray(1)} ''' not specified.'] );
end

% set output
if nargout==1
  varargout{1} = cell2struct( dfsVal, dfsField, 2 );
else
  varargout = dfsVal;
end
