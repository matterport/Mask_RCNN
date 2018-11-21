/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
#include "mex.h"
#include "maskApi.h"
#include <string.h>

void checkType( const mxArray *M, mxClassID id ) {
  if(mxGetClassID(M)!=id) mexErrMsgTxt("Invalid type.");
}

mxArray* toMxArray( const RLE *R, siz n ) {
  const char *fs[] = {"size", "counts"};
  mxArray *M=mxCreateStructMatrix(1,n,2,fs);
  for( siz i=0; i<n; i++ ) {
    mxArray *S=mxCreateNumericMatrix(1,2,mxDOUBLE_CLASS,mxREAL);
    mxSetFieldByNumber(M,i,0,S); double *s=mxGetPr(S);
    s[0]=R[i].h; s[1]=R[i].w; char *c=rleToString(R+i);
    mxSetFieldByNumber(M,i,1,mxCreateString(c)); free(c);
  }
  return M;
}

RLE* frMxArray( const mxArray *M, siz *n, bool same ) {
  const char *fs[] = {"size", "counts"}; siz i, j, m, h, w, O[2];
  const char *err="Invalid RLE struct array.";
  *n=mxGetNumberOfElements(M); RLE *R; rlesInit(&R,*n); if(!(*n)) return R;
  if(!mxIsStruct(M) || mxGetNumberOfFields(M)!=2) mexErrMsgTxt(err);
  for( i=0; i<2; i++ ) { O[i]=2; for( j=0; j<2; j++ ) {
    if(!strcmp(mxGetFieldNameByNumber(M,j),fs[i])) O[i]=j; }}
  for( i=0; i<2; i++ ) if(O[i]>1) mexErrMsgTxt(err);
  for( i=0; i<*n; i++ ) {
    mxArray *S, *C; double *s; void *c;
    S=mxGetFieldByNumber(M,i,O[0]); checkType(S,mxDOUBLE_CLASS);
    C=mxGetFieldByNumber(M,i,O[1]); s=mxGetPr(S); c=mxGetData(C);
    h=(siz)s[0]; w=(siz)s[1]; m=mxGetNumberOfElements(C);
    if(same && i>0 && (h!=R[0].h || w!=R[0].w)) mexErrMsgTxt(err);
    if( mxGetClassID(C)==mxDOUBLE_CLASS ) {
      rleInit(R+i,h,w,m,0);
      for(j=0; j<m; j++) R[i].cnts[j]=(uint)((double*)c)[j];
    } else if( mxGetClassID(C)==mxUINT32_CLASS ) {
      rleInit(R+i,h,w,m,(uint*)c);
    } else if( mxGetClassID(C)==mxCHAR_CLASS ) {
      char *c=mxMalloc(sizeof(char)*(m+1)); mxGetString(C,c,m+1);
      rleFrString(R+i,c,h,w); mxFree(c);
    }
    else mexErrMsgTxt(err);
  }
  return R;
}

void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] )
{
  char action[1024]; RLE *R=0; siz h=0, w=0, n=0;
  mxGetString(pr[0],action,1024); nr--; pr++;
  
  if(!strcmp(action,"encode")) {
    checkType(pr[0],mxUINT8_CLASS); byte *M=(byte*) mxGetData(pr[0]);
    const mwSize *ds=mxGetDimensions(pr[0]); n=mxGetN(pr[0])/ds[1];
    rlesInit(&R,n); rleEncode(R,M,ds[0],ds[1],n); pl[0]=toMxArray(R,n);
    
  } else if(!strcmp(action,"decode")) {
    R=frMxArray(pr[0],&n,1); mwSize ds[3];
    ds[0]=n?R[0].h:0; ds[1]=n?R[0].w:0; ds[2]=n;
    pl[0]=mxCreateNumericArray(3,ds,mxUINT8_CLASS,mxREAL);
    byte *M=(byte*) mxGetPr(pl[0]); rleDecode(R,M,n);
    
  } else if(!strcmp(action,"merge")) {
    R=frMxArray(pr[0],&n,1); RLE M;
    bool intersect = (nr>=2) ? (mxGetScalar(pr[1])>0) : false;
    rleMerge(R,&M,n,intersect); pl[0]=toMxArray(&M,1); rleFree(&M);
    
  } else if(!strcmp(action,"area")) {
    R=frMxArray(pr[0],&n,0);
    pl[0]=mxCreateNumericMatrix(1,n,mxUINT32_CLASS,mxREAL);
    uint *a=(uint*) mxGetPr(pl[0]); rleArea(R,n,a);
    
  } else if(!strcmp(action,"iou")) {
    if(nr>2) checkType(pr[2],mxUINT8_CLASS); siz nDt, nGt;
    byte *iscrowd = nr>2 ? (byte*) mxGetPr(pr[2]) : NULL;
    if(mxIsStruct(pr[0]) || mxIsStruct(pr[1])) {
      RLE *dt=frMxArray(pr[0],&nDt,1), *gt=frMxArray(pr[1],&nGt,1);
      pl[0]=mxCreateNumericMatrix(nDt,nGt,mxDOUBLE_CLASS,mxREAL);
      double *o=mxGetPr(pl[0]); rleIou(dt,gt,nDt,nGt,iscrowd,o);
      rlesFree(&dt,nDt); rlesFree(&gt,nGt);
    } else {
      checkType(pr[0],mxDOUBLE_CLASS); checkType(pr[1],mxDOUBLE_CLASS);
      double *dt=mxGetPr(pr[0]); nDt=mxGetN(pr[0]);
      double *gt=mxGetPr(pr[1]); nGt=mxGetN(pr[1]);
      pl[0]=mxCreateNumericMatrix(nDt,nGt,mxDOUBLE_CLASS,mxREAL);
      double *o=mxGetPr(pl[0]); bbIou(dt,gt,nDt,nGt,iscrowd,o);
    }
    
  } else if(!strcmp(action,"nms")) {
    siz n; uint *keep; double thr=(double) mxGetScalar(pr[1]);
    if(mxIsStruct(pr[0])) {
      RLE *dt=frMxArray(pr[0],&n,1);
      pl[0]=mxCreateNumericMatrix(1,n,mxUINT32_CLASS,mxREAL);
      keep=(uint*) mxGetPr(pl[0]); rleNms(dt,n,keep,thr);
      rlesFree(&dt,n);
    } else {
      checkType(pr[0],mxDOUBLE_CLASS);
      double *dt=mxGetPr(pr[0]); n=mxGetN(pr[0]);
      pl[0]=mxCreateNumericMatrix(1,n,mxUINT32_CLASS,mxREAL);
      keep=(uint*) mxGetPr(pl[0]); bbNms(dt,n,keep,thr);
    }
    
  } else if(!strcmp(action,"toBbox")) {
    R=frMxArray(pr[0],&n,0);
    pl[0]=mxCreateNumericMatrix(4,n,mxDOUBLE_CLASS,mxREAL);
    BB bb=mxGetPr(pl[0]); rleToBbox(R,bb,n);
    
  } else if(!strcmp(action,"frBbox")) {
    checkType(pr[0],mxDOUBLE_CLASS);
    double *bb=mxGetPr(pr[0]); n=mxGetN(pr[0]);
    h=(siz)mxGetScalar(pr[1]); w=(siz)mxGetScalar(pr[2]);
    rlesInit(&R,n); rleFrBbox(R,bb,h,w,n); pl[0]=toMxArray(R,n);
    
  } else if(!strcmp(action,"frPoly")) {
    checkType(pr[0],mxCELL_CLASS); n=mxGetNumberOfElements(pr[0]);
    h=(siz)mxGetScalar(pr[1]); w=(siz)mxGetScalar(pr[2]); rlesInit(&R,n);
    for(siz i=0; i<n; i++) {
      mxArray *XY=mxGetCell(pr[0],i); checkType(XY,mxDOUBLE_CLASS);
      siz k=mxGetNumberOfElements(XY)/2; double *xy=mxGetPr(XY);
      rleFrPoly(R+i,xy,k,h,w);
    }
    RLE M; rleMerge(R,&M,n,0); pl[0]=toMxArray(&M,1); rleFree(&M);
    
  } else mexErrMsgTxt("Invalid action.");
  if( R!=0 ) { rlesFree(&R,n); R=0; }
}
