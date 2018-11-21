classdef CocoApi
  % Interface for accessing the Microsoft COCO dataset.
  %
  % Microsoft COCO is a large image dataset designed for object detection,
  % segmentation, and caption generation. CocoApi.m is a Matlab API that
  % assists in loading, parsing and visualizing the annotations in COCO.
  % Please visit http://mscoco.org/ for more information on COCO, including
  % for the data, paper, and tutorials. The exact format of the annotations
  % is also described on the COCO website. For example usage of the CocoApi
  % please see cocoDemo.m. In addition to this API, please download both
  % the COCO images and annotations in order to run the demo.
  %
  % An alternative to using the API is to load the annotations directly
  % into a Matlab struct. This can be achieved via:
  %  data = gason(fileread(annFile));
  % Using the API provides additional utility functions. Note that this API
  % supports both *instance* and *caption* annotations. In the case of
  % captions not all functions are defined (e.g. categories are undefined).
  %
  % The following API functions are defined:
  %  CocoApi    - Load COCO annotation file and prepare data structures.
  %  getAnnIds  - Get ann ids that satisfy given filter conditions.
  %  getCatIds  - Get cat ids that satisfy given filter conditions.
  %  getImgIds  - Get img ids that satisfy given filter conditions.
  %  loadAnns   - Load anns with the specified ids.
  %  loadCats   - Load cats with the specified ids.
  %  loadImgs   - Load imgs with the specified ids.
  %  showAnns   - Display the specified annotations.
  %  loadRes    - Load algorithm results and create API for accessing them.
  %  download   - Download COCO images from mscoco.org server.
  % Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
  % Help on each functions can be accessed by: "help CocoApi>function".
  %
  % See also CocoApi>CocoApi, CocoApi>getAnnIds, CocoApi>getCatIds,
  % CocoApi>getImgIds, CocoApi>loadAnns, CocoApi>loadCats,
  % CocoApi>loadImgs, CocoApi>showAnns, CocoApi>loadRes, CocoApi>download
  %
  % Microsoft COCO Toolbox.      version 2.0
  % Data, paper, and tutorials available at:  http://mscoco.org/
  % Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
  % Licensed under the Simplified BSD License [see coco/license.txt]
  
  properties
    data    % COCO annotation data structure
    inds    % data structures for fast indexing
  end
  
  methods
    function coco = CocoApi( annFile )
      % Load COCO annotation file and prepare data structures.
      %
      % USAGE
      %  coco = CocoApi( annFile )
      %
      % INPUTS
      %  annFile   - COCO annotation filename
      %
      % OUTPUTS
      %  coco      - initialized coco object
      fprintf('Loading and preparing annotations... '); clk=clock;
      if(isstruct(annFile)), coco.data=annFile; else
        coco.data=gason(fileread(annFile)); end
      is.imgIds = [coco.data.images.id]';
      is.imgIdsMap = makeMap(is.imgIds);
      if( isfield(coco.data,'annotations') )
        ann=coco.data.annotations; o=[ann.image_id];
        if(isfield(ann,'category_id')), o=o*1e10+[ann.category_id]; end
        [~,o]=sort(o); ann=ann(o); coco.data.annotations=ann;
        s={'category_id','area','iscrowd','id','image_id'};
        t={'annCatIds','annAreas','annIscrowd','annIds','annImgIds'};
        for f=1:5, if(isfield(ann,s{f})), is.(t{f})=[ann.(s{f})]'; end; end
        is.annIdsMap = makeMap(is.annIds);
        is.imgAnnIdsMap = makeMultiMap(is.imgIds,...
          is.imgIdsMap,is.annImgIds,is.annIds,0);
      end
      if( isfield(coco.data,'categories') )
        is.catIds = [coco.data.categories.id]';
        is.catIdsMap = makeMap(is.catIds);
        if(isfield(is,'annCatIds')), is.catImgIdsMap = makeMultiMap(...
            is.catIds,is.catIdsMap,is.annCatIds,is.annImgIds,1); end
      end
      coco.inds=is; fprintf('DONE (t=%0.2fs).\n',etime(clock,clk));
      
      function map = makeMap( keys )
        % Make map from key to integer id associated with key.
        if(isempty(keys)), map=containers.Map(); return; end
        map=containers.Map(keys,1:length(keys));
      end
      
      function map = makeMultiMap( keys, keysMap, keysAll, valsAll, sqz )
        % Make map from keys to set of vals associated with each key.
        js=values(keysMap,num2cell(keysAll)); js=[js{:}];
        m=length(js); n=length(keys); k=zeros(1,n);
        for i=1:m, j=js(i); k(j)=k(j)+1; end; vs=zeros(n,max(k)); k(:)=0;
        for i=1:m, j=js(i); k(j)=k(j)+1; vs(j,k(j))=valsAll(i); end
        map = containers.Map('KeyType','double','ValueType','any');
        if(sqz), for j=1:n, map(keys(j))=unique(vs(j,1:k(j))); end
        else for j=1:n, map(keys(j))=vs(j,1:k(j)); end; end
      end
    end
    
    function ids = getAnnIds( coco, varargin )
      % Get ann ids that satisfy given filter conditions.
      %
      % USAGE
      %  ids = coco.getAnnIds( params )
      %
      % INPUTS
      %  params     - filtering parameters (struct or name/value pairs)
      %               setting any filter to [] skips that filter
      %   .imgIds     - [] get anns for given imgs
      %   .catIds     - [] get anns for given cats
      %   .areaRng    - [] get anns for given area range (e.g. [0 inf])
      %   .iscrowd    - [] get anns for given crowd label (0 or 1)
      %
      % OUTPUTS
      %  ids        - integer array of ann ids
      def = {'imgIds',[],'catIds',[],'areaRng',[],'iscrowd',[]};
      [imgIds,catIds,ar,iscrowd] = getPrmDflt(varargin,def,1);
      if( length(imgIds)==1 )
        t = coco.loadAnns(coco.inds.imgAnnIdsMap(imgIds));
        if(~isempty(catIds)), t = t(ismember([t.category_id],catIds)); end
        if(~isempty(ar)), a=[t.area]; t = t(a>=ar(1) & a<=ar(2)); end
        if(~isempty(iscrowd)), t = t([t.iscrowd]==iscrowd); end
        ids = [t.id];
      else
        ids=coco.inds.annIds; K = true(length(ids),1); t = coco.inds;
        if(~isempty(imgIds)), K = K & ismember(t.annImgIds,imgIds); end
        if(~isempty(catIds)), K = K & ismember(t.annCatIds,catIds); end
        if(~isempty(ar)), a=t.annAreas; K = K & a>=ar(1) & a<=ar(2); end
        if(~isempty(iscrowd)), K = K & t.annIscrowd==iscrowd; end
        ids=ids(K);
      end
    end
    
    function ids = getCatIds( coco, varargin )
      % Get cat ids that satisfy given filter conditions.
      %
      % USAGE
      %  ids = coco.getCatIds( params )
      %
      % INPUTS
      %  params     - filtering parameters (struct or name/value pairs)
      %               setting any filter to [] skips that filter
      %   .catNms     - [] get cats for given cat names
      %   .supNms     - [] get cats for given supercategory names
      %   .catIds     - [] get cats for given cat ids
      %
      % OUTPUTS
      %  ids        - integer array of cat ids
      if(~isfield(coco.data,'categories')), ids=[]; return; end
      def={'catNms',[],'supNms',[],'catIds',[]}; t=coco.data.categories;
      [catNms,supNms,catIds] = getPrmDflt(varargin,def,1);
      if(~isempty(catNms)), t = t(ismember({t.name},catNms)); end
      if(~isempty(supNms)), t = t(ismember({t.supercategory},supNms)); end
      if(~isempty(catIds)), t = t(ismember([t.id],catIds)); end
      ids = [t.id];
    end
    
    function ids = getImgIds( coco, varargin )
      % Get img ids that satisfy given filter conditions.
      %
      % USAGE
      %  ids = coco.getImgIds( params )
      %
      % INPUTS
      %  params     - filtering parameters (struct or name/value pairs)
      %               setting any filter to [] skips that filter
      %   .imgIds     - [] get imgs for given ids
      %   .catIds     - [] get imgs with all given cats
      %
      % OUTPUTS
      %  ids        - integer array of img ids
      def={'imgIds',[],'catIds',[]}; ids=coco.inds.imgIds;
      [imgIds,catIds] = getPrmDflt(varargin,def,1);
      if(~isempty(imgIds)), ids=intersect(ids,imgIds); end
      if(isempty(catIds)), return; end
      t=values(coco.inds.catImgIdsMap,num2cell(catIds));
      for i=1:length(t), ids=intersect(ids,t{i}); end
    end
    
    function anns = loadAnns( coco, ids )
      % Load anns with the specified ids.
      %
      % USAGE
      %  anns = coco.loadAnns( ids )
      %
      % INPUTS
      %  ids        - integer ids specifying anns
      %
      % OUTPUTS
      %  anns       - loaded ann objects
      ids = values(coco.inds.annIdsMap,num2cell(ids));
      anns = coco.data.annotations([ids{:}]);
    end
    
    function cats = loadCats( coco, ids )
      % Load cats with the specified ids.
      %
      % USAGE
      %  cats = coco.loadCats( ids )
      %
      % INPUTS
      %  ids        - integer ids specifying cats
      %
      % OUTPUTS
      %  cats       - loaded cat objects
      if(~isfield(coco.data,'categories')), cats=[]; return; end
      ids = values(coco.inds.catIdsMap,num2cell(ids));
      cats = coco.data.categories([ids{:}]);
    end
    
    function imgs = loadImgs( coco, ids )
      % Load imgs with the specified ids.
      %
      % USAGE
      %  imgs = coco.loadImgs( ids )
      %
      % INPUTS
      %  ids        - integer ids specifying imgs
      %
      % OUTPUTS
      %  imgs       - loaded img objects
      ids = values(coco.inds.imgIdsMap,num2cell(ids));
      imgs = coco.data.images([ids{:}]);
    end
    
    function hs = showAnns( coco, anns )
      % Display the specified annotations.
      %
      % USAGE
      %  hs = coco.showAnns( anns )
      %
      % INPUTS
      %  anns       - annotations to display
      %
      % OUTPUTS
      %  hs         - handles to segment graphic objects
      n=length(anns); if(n==0), return; end
      r=.4:.2:1; [r,g,b]=ndgrid(r,r,r); cs=[r(:) g(:) b(:)];
      cs=cs(randperm(size(cs,1)),:); cs=repmat(cs,100,1);
      if( isfield( anns,'keypoints') )
        for i=1:n
          a=anns(i); if(isfield(a,'iscrowd') && a.iscrowd), continue; end
          seg={}; if(isfield(a,'segmentation')), seg=a.segmentation; end
          k=a.keypoints; x=k(1:3:end)+1; y=k(2:3:end)+1; v=k(3:3:end);
          k=coco.loadCats(a.category_id); k=k.skeleton; c=cs(i,:); hold on
          p={'FaceAlpha',.25,'LineWidth',2,'EdgeColor',c}; % polygon
          for j=seg, xy=j{1}+.5; fill(xy(1:2:end),xy(2:2:end),c,p{:}); end
          p={'Color',c,'LineWidth',3}; % skeleton
          for j=k, s=j{1}; if(all(v(s)>0)), line(x(s),y(s),p{:}); end; end
          p={'MarkerSize',8,'MarkerFaceColor',c,'MarkerEdgeColor'}; % pnts
          plot(x(v>0),y(v>0),'o',p{:},'k');
          plot(x(v>1),y(v>1),'o',p{:},c); hold off;
        end
      elseif( any(isfield(anns,{'segmentation','bbox'})) )
        if(~isfield(anns,'iscrowd')), [anns(:).iscrowd]=deal(0); end
        if(~isfield(anns,'segmentation')), S={anns.bbox}; %#ok<ALIGN>
          for i=1:n, x=S{i}(1); w=S{i}(3); y=S{i}(2); h=S{i}(4);
            anns(i).segmentation={[x,y,x,y+h,x+w,y+h,x+w,y]}; end; end
        S={anns.segmentation}; hs=zeros(10000,1); k=0; hold on;
        pFill={'FaceAlpha',.4,'LineWidth',3};
        for i=1:n
          if(anns(i).iscrowd), C=[.01 .65 .40]; else C=rand(1,3); end
          if(isstruct(S{i})), M=double(MaskApi.decode(S{i})); k=k+1;
            hs(k)=imagesc(cat(3,M*C(1),M*C(2),M*C(3)),'Alphadata',M*.5);
          else for j=1:length(S{i}), P=S{i}{j}+.5; k=k+1;
              hs(k)=fill(P(1:2:end),P(2:2:end),C,pFill{:}); end
          end
        end
        hs=hs(1:k); hold off;
      elseif( isfield(anns,'caption') )
        S={anns.caption};
        for i=1:n, S{i}=[int2str(i) ') ' S{i} '\newline']; end
        S=[S{:}]; title(S,'FontSize',12);
      end
    end
    
    function cocoRes = loadRes( coco, resFile )
      % Load algorithm results and create API for accessing them.
      %
      % The API for accessing and viewing algorithm results is identical to
      % the CocoApi for the ground truth. The single difference is that the
      % ground truth results are replaced by the algorithm results.
      %
      % USAGE
      %  cocoRes = coco.loadRes( resFile )
      %
      % INPUTS
      %  resFile    - COCO results filename
      %
      % OUTPUTS
      %  cocoRes    - initialized results API
      fprintf('Loading and preparing results...     '); clk=clock;
      cdata=coco.data; R=gason(fileread(resFile)); m=length(R);
      valid=ismember([R.image_id],[cdata.images.id]);
      if(~all(valid)), error('Results provided for invalid images.'); end
      t={'segmentation','bbox','keypoints','caption'}; t=t{isfield(R,t)};
      if(strcmp(t,'caption'))
        for i=1:m, R(i).id=i; end; imgs=cdata.images;
        cdata.images=imgs(ismember([imgs.id],[R.image_id]));
      else
        assert(all(isfield(R,{'category_id','score',t})));
        s=cat(1,R.(t)); if(strcmp(t,'bbox')), a=s(:,3).*s(:,4); end
        if(strcmp(t,'segmentation')), a=MaskApi.area(s); end
        if(strcmp(t,'keypoints')), x=s(:,1:3:end)'; y=s(:,2:3:end)';
          a=(max(x)-min(x)).*(max(y)-min(y)); end
        for i=1:m, R(i).area=a(i); R(i).id=i; end
      end
      fprintf('DONE (t=%0.2fs).\n',etime(clock,clk));
      cdata.annotations=R; cocoRes=CocoApi(cdata);
    end
    
    function download( coco, tarDir, maxn )
      % Download COCO images from mscoco.org server.
      %
      % USAGE
      %  coco.download( tarDir, [maxn] )
      %
      % INPUTS
      %  tarDir     - COCO results filename
      %  maxn       - maximum number of images to download
      fs={coco.data.images.file_name}; n=length(fs);
      if(nargin==3), n=min(n,maxn); end; [fs,o]=sort(fs);
      urls={coco.data.images.coco_url}; urls=urls(o); do=true(1,n);
      for i=1:n, fs{i}=[tarDir '/' fs{i}]; do(i)=~exist(fs{i},'file'); end
      fs=fs(do); urls=urls(do); n=length(fs); if(n==0), return; end
      if(~exist(tarDir,'dir')), mkdir(tarDir); end; t=tic;
      m='downloaded %i/%i images (t=%.1fs)\n'; o=weboptions('Timeout',60);
      for i=1:n, websave(fs{i},urls{i},o); fprintf(m,i,n,toc(t)); end
    end
  end
  
end
