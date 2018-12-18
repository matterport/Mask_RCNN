--[[----------------------------------------------------------------------------

Interface for accessing the Common Objects in COntext (COCO) dataset.

For an overview of the API please see http://mscoco.org/dataset/#download.
CocoApi.lua (this file) is modeled after the Matlab CocoApi.m:
  https://github.com/pdollar/coco/blob/master/MatlabAPI/CocoApi.m

The following API functions are defined in the Lua API:
  CocoApi    - Load COCO annotation file and prepare data structures.
  getAnnIds  - Get ann ids that satisfy given filter conditions.
  getCatIds  - Get cat ids that satisfy given filter conditions.
  getImgIds  - Get img ids that satisfy given filter conditions.
  loadAnns   - Load anns with the specified ids.
  loadCats   - Load cats with the specified ids.
  loadImgs   - Load imgs with the specified ids.
  showAnns   - Display the specified annotations.
Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
For detailed usage information please see cocoDemo.lua.

LIMITATIONS: the following API functions are NOT defined in the Lua API:
  loadRes    - Load algorithm results and create API for accessing them.
  download   - Download COCO images from mscoco.org server.
In addition, currently the getCatIds() and getImgIds() do not accept filters.
getAnnIds() can be called using getAnnIds({imgId=id}) and getAnnIds({catId=id}).

Note: loading COCO JSON annotations to Lua tables is quite slow. Hence, a call
to CocApi(annFile) converts the annotations to a custom 'flattened' format that
is more efficient. The first time a COCO JSON is loaded, the conversion is
invoked (this may take up to a minute). The converted data is then stored in a
t7 file (the code must have write permission to the dir of the JSON file).
Future calls of cocoApi=CocApi(annFile) take a fraction of a second. To view the
created data just inspect cocoApi.data of a created instance of the CocoApi.

Common Objects in COntext (COCO) Toolbox.      version 3.0
Data, paper, and tutorials available at:  http://mscoco.org/
Code written by Pedro O. Pinheiro and Piotr Dollar, 2016.
Licensed under the Simplified BSD License [see coco/license.txt]

------------------------------------------------------------------------------]]

local json = require 'cjson'
local coco = require 'coco.env'

local TensorTable = torch.class('TensorTable',coco)
local CocoSeg = torch.class('CocoSeg',coco)
local CocoApi = torch.class('CocoApi',coco)

--------------------------------------------------------------------------------

--[[ TensorTable is a lightweight data structure for storing variable size 1D
tensors. Tables of tensors are slow to save/load to disk. Instead, TensorTable
stores all the data in a single long tensor (along with indices into the tensor)
making serialization fast. A TensorTable may only contain 1D same-type torch
tensors or strings. It supports only creation from a table and indexing. ]]

function TensorTable:__init( T )
  local n = #T; assert(n>0)
  local isStr = torch.type(T[1])=='string'
  assert(isStr or torch.isTensor(T[1]))
  local c=function(s) return torch.CharTensor(torch.CharStorage():string(s)) end
  if isStr then local S=T; T={}; for i=1,n do T[i]=c(S[i]) end end
  local ms, idx = torch.LongTensor(n), torch.LongTensor(n+1)
  for i=1,n do ms[i]=T[i]:numel() end
  idx[1]=1; idx:narrow(1,2,n):copy(ms); idx=idx:cumsum()
  local type = string.sub(torch.type(T[1]),7,-1)
  local data = torch[type](idx[n+1]-1)
  if isStr then type='string' end
  for i=1,n do if ms[i]>0 then data:sub(idx[i],idx[i+1]-1):copy(T[i]) end end
  if ms:eq(ms[1]):all() and ms[1]>0 then data=data:view(n,ms[1]); idx=nil end
  self.data, self.idx, self.type = data, idx, type
end

function TensorTable:__index__( i )
  if torch.type(i)~='number' then return false end
  local d, idx, type = self.data, self.idx, self.type
  if idx and idx[i]==idx[i+1] then
    if type=='string' then d='' else d=torch[type]() end
  else
    if idx then d=d:sub(idx[i],idx[i+1]-1) else d=d[i] end
    if type=='string' then d=d:clone():storage():string() end
  end
  return d, true
end

--------------------------------------------------------------------------------

--[[ CocoSeg is an efficient data structure for storing COCO segmentations. ]]

function CocoSeg:__init( segs )
  local polys, pIdx, sizes, rles, p, isStr = {}, {}, {}, {}, 0, 0
  for i,seg in pairs(segs) do if seg.size then isStr=seg.counts break end end
  isStr = torch.type(isStr)=='string'
  for i,seg in pairs(segs) do
    pIdx[i], sizes[i] = {}, {}
    if seg.size then
      sizes[i],rles[i] = seg.size,seg.counts
    else
      if isStr then rles[i]='' else rles[i]={} end
      for j=1,#seg do p=p+1; pIdx[i][j],polys[p] = p,seg[j] end
    end
    pIdx[i],sizes[i] = torch.LongTensor(pIdx[i]),torch.IntTensor(sizes[i])
    if not isStr then rles[i]=torch.IntTensor(rles[i]) end
  end
  for i=1,p do polys[i]=torch.DoubleTensor(polys[i]) end
  self.polys, self.pIdx = coco.TensorTable(polys), coco.TensorTable(pIdx)
  self.sizes, self.rles = coco.TensorTable(sizes), coco.TensorTable(rles)
end

function CocoSeg:__index__( i )
  if torch.type(i)~='number' then return false end
  if self.sizes[i]:numel()>0 then
    return {size=self.sizes[i],counts=self.rles[i]}, true
  else
    local ids, polys = self.pIdx[i], {}
    for i=1,ids:numel() do polys[i]=self.polys[ids[i]] end
    return polys, true
  end
end

--------------------------------------------------------------------------------

--[[ CocoApi is the API to the COCO dataset, see main comment for details. ]]

function CocoApi:__init( annFile )
  assert( string.sub(annFile,-4,-1)=='json' and paths.filep(annFile) )
  local torchFile = string.sub(annFile,1,-6) .. '.t7'
  if not paths.filep(torchFile) then self:__convert(annFile,torchFile) end
  local data = torch.load(torchFile)
  self.data, self.inds = data, {}
  for k,v in pairs({images='img',categories='cat',annotations='ann'}) do
    local M = {}; self.inds[v..'IdsMap']=M
    if data[k] then for i=1,data[k].id:size(1) do M[data[k].id[i]]=i end end
  end
end

function CocoApi:__convert( annFile, torchFile )
  print('convert: '..annFile..' --> .t7 [please be patient]')
  local tic = torch.tic()
  -- load data and decode json
  local data = torch.CharStorage(annFile):string()
  data = json.decode(data); collectgarbage()
  -- transpose and flatten each field in the coco data struct
  local convert = {images=true, categories=true, annotations=true}
  for field, d in pairs(data) do if convert[field] then
    print('converting: '..field)
    local n, out = #d, {}
    if n==0 then d,n={d},1 end
    for k,v in pairs(d[1]) do
      local t, isReg = torch.type(v), true
      for i=1,n do isReg=isReg and torch.type(d[i][k])==t end
      if t=='number' and isReg then
        out[k] = torch.DoubleTensor(n)
        for i=1,n do out[k][i]=d[i][k] end
      elseif t=='string' and isReg then
        out[k]={}; for i=1,n do out[k][i]=d[i][k] end
        out[k] = coco.TensorTable(out[k])
      elseif t=='table' and isReg and torch.type(v[1])=='number' then
        out[k]={}; for i=1,n do out[k][i]=torch.DoubleTensor(d[i][k]) end
        out[k] = coco.TensorTable(out[k])
        if not out[k].idx then out[k]=out[k].data end
      else
        out[k]={}; for i=1,n do out[k][i]=d[i][k] end
        if k=='segmentation' then out[k] = coco.CocoSeg(out[k]) end
      end
      collectgarbage()
    end
    if out.id then out.idx=torch.range(1,out.id:size(1)) end
    data[field] = out
    collectgarbage()
  end end
  -- create mapping from cat/img index to anns indices for that cat/img
  print('convert: building indices')
  local makeMap = function( type, type_id )
    if not data[type] or not data.annotations then return nil end
    local invmap, n = {}, data[type].id:size(1)
    for i=1,n do invmap[data[type].id[i]]=i end
    local map = {}; for i=1,n do map[i]={} end
    data.annotations[type_id..'x'] = data.annotations[type_id]:clone()
    for i=1,data.annotations.id:size(1) do
      local id = invmap[data.annotations[type_id][i]]
      data.annotations[type_id..'x'][i] = id
      table.insert(map[id],data.annotations.id[i])
    end
    for i=1,n do map[i]=torch.LongTensor(map[i]) end
    return coco.TensorTable(map)
  end
  data.annIdsPerImg = makeMap('images','image_id')
  data.annIdsPerCat = makeMap('categories','category_id')
  -- save to disk
  torch.save( torchFile, data )
  print(('convert: complete [%.2f s]'):format(torch.toc(tic)))
end

function CocoApi:getAnnIds( filters )
  if not filters then filters = {} end
  if filters.imgId then
    return self.data.annIdsPerImg[self.inds.imgIdsMap[filters.imgId]] or {}
  elseif filters.catId then
    return self.data.annIdsPerCat[self.inds.catIdsMap[filters.catId]] or {}
  else
    return self.data.annotations.id
  end
end

function CocoApi:getCatIds()
  return self.data.categories.id
end

function CocoApi:getImgIds()
  return self.data.images.id
end

function CocoApi:loadAnns( ids )
  return self:__load(self.data.annotations,self.inds.annIdsMap,ids)
end

function CocoApi:loadCats( ids )
  return self:__load(self.data.categories,self.inds.catIdsMap,ids)
end

function CocoApi:loadImgs( ids )
  return self:__load(self.data.images,self.inds.imgIdsMap,ids)
end

function CocoApi:showAnns( img, anns )
  local n, h, w = #anns, img:size(2), img:size(3)
  local MaskApi, clrs = coco.MaskApi, torch.rand(n,3)*.6+.4
  local O = img:clone():contiguous():float()
  if n==0 then anns,n={anns},1 end
  if anns[1].keypoints then for i=1,n do if anns[i].iscrowd==0 then
    local sk, kp, j, k = self:loadCats(anns[i].category_id)[1].skeleton
    kp=anns[i].keypoints; k=kp:size(1); j=torch.range(1,k,3):long(); k=k/3;
    local x,y,v = kp:index(1,j), kp:index(1,j+1), kp:index(1,j+2)
    for _,s in pairs(sk) do if v[s[1]]>0 and v[s[2]]>0 then
      MaskApi.drawLine(O,x[s[1]],y[s[1]],x[s[2]],y[s[2]],.75,clrs[i])
    end end
    for j=1,k do if v[j]==1 then MaskApi.drawCirc(O,x[j],y[j],4,{0,0,0}) end end
    for j=1,k do if v[j]>0 then MaskApi.drawCirc(O,x[j],y[j],3,clrs[i]) end end
  end end end
  if anns[1].segmentation or anns[1].bbox then
    local Rs, alpha = {}, anns[1].keypoints and .25 or .4
    for i=1,n do
      Rs[i]=anns[i].segmentation
      if Rs[i] and #Rs[i]>0 then Rs[i]=MaskApi.frPoly(Rs[i],h,w) end
      if not Rs[i] then Rs[i]=MaskApi.frBbox(anns[i].bbox,h,w)[1] end
    end
    MaskApi.drawMasks(O,MaskApi.decode(Rs),nil,alpha,clrs)
  end
  return O
end

function CocoApi:__load( data, map, ids )
  if not torch.isTensor(ids) then ids=torch.LongTensor({ids}) end
  local out, idx = {}, nil
  for i=1,ids:numel() do
    out[i], idx = {}, map[ids[i]]
    for k,v in pairs(data) do out[i][k]=v[idx] end
  end
  return out
end
