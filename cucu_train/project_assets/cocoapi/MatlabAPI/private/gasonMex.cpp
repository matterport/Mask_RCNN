/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
#include "gason.h"
#include "mex.h"
#include "string.h"
#include "math.h"
#include <cstdint>
#include <iomanip>
#include <sstream>
typedef std::ostringstream ostrm;
typedef unsigned long siz;
typedef unsigned short ushort;

siz length( const JsonValue &a ) {
  // get number of elements in JSON_ARRAY or JSON_OBJECT
  siz k=0; auto n=a.toNode(); while(n) { k++; n=n->next; } return k;
}

bool isRegularObjArray( const JsonValue &a ) {
  // check if all JSON_OBJECTs in JSON_ARRAY have the same fields
  JsonValue o=a.toNode()->value; siz k, n; const char **keys;
  n=length(o); keys=new const char*[n];
  k=0; for(auto j:o) keys[k++]=j->key;
  for( auto i:a ) {
    if(length(i->value)!=n) return false; k=0;
    for(auto j:i->value) if(strcmp(j->key,keys[k++])) return false;
  }
  delete [] keys; return true;
}

mxArray* json( const JsonValue &o ) {
  // convert JsonValue to Matlab mxArray
  siz k, m, n; mxArray *M; const char **keys;
  switch( o.getTag() ) {
    case JSON_NUMBER:
      return mxCreateDoubleScalar(o.toNumber());
    case JSON_STRING:
      return mxCreateString(o.toString());
    case JSON_ARRAY: {
      if(!o.toNode()) return mxCreateDoubleMatrix(1,0,mxREAL);
      JsonValue o0=o.toNode()->value; JsonTag tag=o0.getTag();
      n=length(o); bool isRegular=true;
      for(auto i:o) isRegular=isRegular && i->value.getTag()==tag;
      if( isRegular && tag==JSON_OBJECT && isRegularObjArray(o) ) {
        m=length(o0); keys=new const char*[m];
        k=0; for(auto j:o0) keys[k++]=j->key;
        M = mxCreateStructMatrix(1,n,m,keys);
        k=0; for(auto i:o) { m=0; for(auto j:i->value)
          mxSetFieldByNumber(M,k,m++,json(j->value)); k++; }
        delete [] keys; return M;
      } else if( isRegular && tag==JSON_NUMBER ) {
        M = mxCreateDoubleMatrix(1,n,mxREAL); double *p=mxGetPr(M);
        k=0; for(auto i:o) p[k++]=i->value.toNumber(); return M;
      } else {
        M = mxCreateCellMatrix(1,n);
        k=0; for(auto i:o) mxSetCell(M,k++,json(i->value));
        return M;
      }
    }
    case JSON_OBJECT:
      if(!o.toNode()) return mxCreateStructMatrix(1,0,0,NULL);
      n=length(o); keys=new const char*[n];
      k=0; for(auto i:o) keys[k++]=i->key;
      M = mxCreateStructMatrix(1,1,n,keys); k=0;
      for(auto i:o) mxSetFieldByNumber(M,0,k++,json(i->value));
      delete [] keys; return M;
    case JSON_TRUE:
      return mxCreateDoubleScalar(1);
    case JSON_FALSE:
      return mxCreateDoubleScalar(0);
    case JSON_NULL:
      return mxCreateDoubleMatrix(0,0,mxREAL);
    default: return NULL;
  }
}

template<class T, class C> ostrm& json( ostrm &S, T *A, siz n ) {
  // convert numeric array to JSON string with casting
  if(n==0) { S<<"[]"; return S; } if(n==1) { S<<C(A[0]); return S; }
  S<<"["; for(siz i=0; i<n-1; i++) S<<C(A[i])<<",";
  S<<C(A[n-1]); S<<"]"; return S;
}

template<class T> ostrm& json( ostrm &S, T *A, siz n ) {
  // convert numeric array to JSON string without casting
  return json<T,T>(S,A,n);
}

ostrm& json( ostrm &S, const char *A ) {
  // convert char array to JSON string (handle escape characters)
  #define RPL(a,b) case a: { S << b; A++; break; }
  S << "\""; while( *A>0 ) switch( *A ) {
    RPL('"',"\\\""); RPL('\\',"\\\\"); RPL('/',"\\/"); RPL('\b',"\\b");
    RPL('\f',"\\f"); RPL('\n',"\\n"); RPL('\r',"\\r"); RPL('\t',"\\t");
    default: S << *A; A++;
  }
  S << "\""; return S;
}

ostrm& json( ostrm& S, const JsonValue *o ) {
  // convert JsonValue to JSON string
  switch( o->getTag() ) {
    case JSON_NUMBER: S << o->toNumber(); return S;
    case JSON_TRUE:   S << "true"; return S;
    case JSON_FALSE:  S << "false"; return S;
    case JSON_NULL:   S << "null"; return S;
    case JSON_STRING: return json(S,o->toString());
    case JSON_ARRAY:
      S << "["; for(auto i:*o) {
        json(S,&i->value) << (i->next ? "," : ""); }
      S << "]"; return S;
    case JSON_OBJECT:
      S << "{"; for(auto i:*o) {
        json(S,i->key) << ":";
        json(S,&i->value) << (i->next ? "," : ""); }
      S << "}"; return S;
    default: return S;
  }
}

ostrm& json( ostrm& S, const mxArray *M ) {
  // convert Matlab mxArray to JSON string
  siz i, j, m, n=mxGetNumberOfElements(M);
  void *A=mxGetData(M); ostrm *nms;
  switch( mxGetClassID(M) ) {
    case mxDOUBLE_CLASS:  return json(S,(double*)   A,n);
    case mxSINGLE_CLASS:  return json(S,(float*)    A,n);
    case mxINT64_CLASS:   return json(S,(int64_t*)  A,n);
    case mxUINT64_CLASS:  return json(S,(uint64_t*) A,n);
    case mxINT32_CLASS:   return json(S,(int32_t*)  A,n);
    case mxUINT32_CLASS:  return json(S,(uint32_t*) A,n);
    case mxINT16_CLASS:   return json(S,(int16_t*)  A,n);
    case mxUINT16_CLASS:  return json(S,(uint16_t*) A,n);
    case mxINT8_CLASS:    return json<int8_t,int32_t>(S,(int8_t*) A,n);
    case mxUINT8_CLASS:   return json<uint8_t,uint32_t>(S,(uint8_t*) A,n);
    case mxLOGICAL_CLASS: return json<uint8_t,uint32_t>(S,(uint8_t*) A,n);
    case mxCHAR_CLASS:    return json(S,mxArrayToString(M));
    case mxCELL_CLASS:
      S << "["; for(i=0; i<n-1; i++) json(S,mxGetCell(M,i)) << ",";
      if(n>0) json(S,mxGetCell(M,n-1)); S << "]"; return S;
    case mxSTRUCT_CLASS:
      if(n==0) { S<<"{}"; return S; } m=mxGetNumberOfFields(M);
      if(m==0) { S<<"["; for(i=0; i<n; i++) S<<"{},"; S<<"]"; return S; }
      if(n>1) S<<"["; nms=new ostrm[m];
      for(j=0; j<m; j++) json(nms[j],mxGetFieldNameByNumber(M,j));
      for(i=0; i<n; i++) for(j=0; j<m; j++) {
        if(j==0) S << "{"; S << nms[j].str() << ":";
        json(S,mxGetFieldByNumber(M,i,j)) << ((j<m-1) ? "," : "}");
        if(j==m-1 && i<n-1) S<<",";
      }
      if(n>1) S<<"]"; delete [] nms; return S;
    default:
      mexErrMsgTxt( "Unknown type." ); return S;
  }
}

mxArray* mxCreateStringRobust( const char* str ) {
  // convert char* to Matlab string (robust version of mxCreateString)
  mxArray *M; ushort *c; mwSize n[2]={1,strlen(str)};
  M=mxCreateCharArray(2,n); c=(ushort*) mxGetData(M);
  for( siz i=0; i<n[1]; i++ ) c[i]=str[i]; return M;
}

char* mxArrayToStringRobust( const mxArray *M ) {
  // convert Matlab string to char* (robust version of mxArrayToString)
  if(!mxIsChar(M)) mexErrMsgTxt("String expected.");
  ushort *c=(ushort*) mxGetData(M); char* str; siz n;
  n=mxGetNumberOfElements(M); str=(char*) mxMalloc(n+1);
  for( siz i=0; i<n; i++ ) str[i]=c[i]; str[n]=0; return str;
}

void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] )
{
  char action[1024]; if(!nr) mexErrMsgTxt("Inputs expected.");
  mxGetString(pr[0],action,1024); nr--; pr++;
  char *endptr; JsonValue val; JsonAllocator allocator;
  if( nl>1 ) mexErrMsgTxt("One output expected.");
  
  if(!strcmp(action,"convert")) {
    if( nr!=1 ) mexErrMsgTxt("One input expected.");
    if( mxGetClassID(pr[0])==mxCHAR_CLASS ) {
      // object = mexFunction( string )
      char *str = mxArrayToStringRobust(pr[0]);
      int status = jsonParse(str, &endptr, &val, allocator);
      if( status != JSON_OK) mexErrMsgTxt(jsonStrError(status));
      pl[0] = json(val); mxFree(str);
    } else {
      // string = mexFunction( object )
      ostrm S; S << std::setprecision(12); json(S,pr[0]);
      pl[0]=mxCreateStringRobust(S.str().c_str());
    }
    
  } else if(!strcmp(action,"split")) {
    // strings = mexFunction( string, k )
    if( nr!=2 ) mexErrMsgTxt("Two input expected.");
    char *str = mxArrayToStringRobust(pr[0]);
    int status = jsonParse(str, &endptr, &val, allocator);
    if( status != JSON_OK) mexErrMsgTxt(jsonStrError(status));
    if( val.getTag()!=JSON_ARRAY ) mexErrMsgTxt("Array expected");
    siz i=0, t=0, n=length(val), k=(siz) mxGetScalar(pr[1]);
    k=(k>n)?n:(k<1)?1:k; k=ceil(n/ceil(double(n)/k));
    pl[0]=mxCreateCellMatrix(1,k); ostrm S; S<<std::setprecision(12);
    for(auto o:val) {
      if(!t) { S.str(std::string()); S << "["; t=ceil(double(n)/k); }
      json(S,&o->value); t--; if(!o->next) t=0; S << (t ? "," : "]");
      if(!t) mxSetCell(pl[0],i++,mxCreateStringRobust(S.str().c_str()));
    }
    
  } else if(!strcmp(action,"merge")) {
    // string = mexFunction( strings )
    if( nr!=1 ) mexErrMsgTxt("One input expected.");
    if(!mxIsCell(pr[0])) mexErrMsgTxt("Cell array expected.");
    siz n = mxGetNumberOfElements(pr[0]);
    ostrm S; S << std::setprecision(12); S << "[";
    for( siz i=0; i<n; i++ ) {
      char *str = mxArrayToStringRobust(mxGetCell(pr[0],i));
      int status = jsonParse(str, &endptr, &val, allocator);
      if( status != JSON_OK) mexErrMsgTxt(jsonStrError(status));
      if( val.getTag()!=JSON_ARRAY ) mexErrMsgTxt("Array expected");
      for(auto j:val) json(S,&j->value) << (j->next ? "," : "");
      mxFree(str); if(i<n-1) S<<",";
    }
    S << "]"; pl[0]=mxCreateStringRobust(S.str().c_str());
    
  } else mexErrMsgTxt("Invalid action.");
}
