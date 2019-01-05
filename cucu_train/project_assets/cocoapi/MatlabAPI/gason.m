function out = gason( in )
% Convert between JSON strings and corresponding JSON objects.
%
% This parser is based on Gason written and maintained by Ivan Vashchaev:
%                 https://github.com/vivkin/gason
% Gason is a "lightweight and fast JSON parser for C++". Please see the
% above link for license information and additional details about Gason.
%
% Given a JSON string, gason calls the C++ parser and converts the output
% into an appropriate Matlab structure. As the parsing is performed in mex
% the resulting parser is blazingly fast. Large JSON structs (100MB+) take
% only a few seconds to parse (compared to hours for pure Matlab parsers).
%
% Given a JSON object, gason calls the C++ encoder to convert the object
% back into a JSON string representation. Nearly any Matlab struct, cell
% array, or numeric array represent a valid JSON object. Note that gason()
% can be used to go both from JSON string to JSON object and back.
%
% Gason requires C++11 to compile (for GCC this requires version 4.7 or
% later). The following command compiles the parser (may require tweaking):
%   mex('CXXFLAGS=\$CXXFLAGS -std=c++11 -Wall','-largeArrayDims',...
%     'private/gasonMex.cpp','../common/gason.cpp',...
%     '-I../common/','-outdir','private');
% Note the use of the "-std=c++11" flag. A number of precompiled binaries
% are included, please do not contact us for help with compiling. If needed
% you can specify a compiler by adding the option 'CXX="/usr/bin/g++"'.
%
% Note that by default JSON arrays that contain only numbers are stored as
% regular Matlab arrays. Likewise, JSON arrays that contain only objects of
% the same type are stored as Matlab struct arrays. This is much faster and
% can use considerably less memory than always using Matlab cell arrays.
%
% USAGE
%  object = gason( string )
%  string = gason( object )
%
% INPUTS/OUTPUTS
%  string     - JSON string
%  object     - JSON object
%
% EXAMPLE
%  o = struct('first',{'piotr','ty'},'last',{'dollar','lin'})
%  s = gason( o ) % convert JSON object -> JSON string
%  p = gason( s ) % convert JSON string -> JSON object
%
% See also
%
% Microsoft COCO Toolbox.      version 2.0
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
% Licensed under the Simplified BSD License [see coco/license.txt]

out = gasonMex( 'convert', in );
