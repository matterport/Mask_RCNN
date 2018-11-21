classdef CocoUtils
  % Utility functions for testing and validation of COCO code.
  %
  % The following utility functions are defined:
  %  convertPascalGt    - Convert ground truth for PASCAL to COCO format.
  %  convertImageNetGt  - Convert ground truth for ImageNet to COCO format.
  %  convertPascalDt    - Convert detections on PASCAL to COCO format.
  %  convertImageNetDt  - Convert detections on ImageNet to COCO format.
  %  validateOnPascal   - Validate COCO eval code against PASCAL code.
  %  validateOnImageNet - Validate COCO eval code against ImageNet code.
  %  generateFakeDt     - Generate fake detections from ground truth.
  %  validateMaskApi    - Validate MaskApi against Matlab functions.
  %  gasonSplit         - Split JSON file into multiple JSON files.
  %  gasonMerge         - Merge JSON files into single JSON file.
  % Help on each functions can be accessed by: "help CocoUtils>function".
  %
  % See also CocoApi MaskApi CocoEval CocoUtils>convertPascalGt
  % CocoUtils>convertImageNetGt CocoUtils>convertPascalDt
  % CocoUtils>convertImageNetDt CocoUtils>validateOnPascal
  % CocoUtils>validateOnImageNet CocoUtils>generateFakeDt
  % CocoUtils>validateMaskApi CocoUtils>gasonSplit CocoUtils>gasonMerge
  %
  % Microsoft COCO Toolbox.      version 2.0
  % Data, paper, and tutorials available at:  http://mscoco.org/
  % Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
  % Licensed under the Simplified BSD License [see coco/license.txt]
  
  methods( Static )
    function convertPascalGt( dataDir, year, split, annFile )
      % Convert ground truth for PASCAL to COCO format.
      %
      % USAGE
      %  CocoUtils.convertPascalGt( dataDir, year, split, annFile )
      %
      % INPUTS
      %  dataDir    - dir containing VOCdevkit/
      %  year       - dataset year (e.g. '2007')
      %  split      - dataset split (e.g. 'val')
      %  annFile    - annotation file for writing results
      if(exist(annFile,'file')), return; end
      fprintf('Converting PASCAL VOC dataset...     '); clk=tic;
      dev=[dataDir '/VOCdevkit/']; addpath(genpath([dev '/VOCcode']));
      VOCinit; C=VOCopts.classes'; catsMap=containers.Map(C,1:length(C));
      f=fopen([dev '/VOC' year '/ImageSets/Main/' split '.txt']);
      is=textscan(f,'%s %*s'); is=is{1}; fclose(f); n=length(is);
      data=CocoUtils.initData(C,n);
      for i=1:n, nm=[is{i} '.jpg'];
        f=[dev '/VOC' year '/Annotations/' is{i} '.xml'];
        R=PASreadrecord(f); hw=R.imgsize([2 1]); O=R.objects;
        id=is{i}; id(id=='_')=[]; id=str2double(id);
        ignore=[O.difficult]; bbs=cat(1,O.bbox);
        t=catsMap.values({O.class}); catIds=[t{:}]; iscrowd=ignore*0;
        data=CocoUtils.addData(data,nm,id,hw,catIds,ignore,iscrowd,bbs);
      end
      f=fopen(annFile,'w'); fwrite(f,gason(data)); fclose(f);
      fprintf('DONE (t=%0.2fs).\n',toc(clk));
    end
    
    function convertImageNetGt( dataDir, year, split, annFile )
      % Convert ground truth for ImageNet to COCO format.
      %
      % USAGE
      %  CocoUtils.convertImageNetGt( dataDir, year, split, annFile )
      %
      % INPUTS
      %  dataDir    - dir containing ILSVRC*/ folders
      %  year       - dataset year (e.g. '2013')
      %  split      - dataset split (e.g. 'val')
      %  annFile    - annotation file for writing results
      if(exist(annFile,'file')), return; end
      fprintf('Converting ImageNet dataset...       '); clk=tic;
      dev=[dataDir '/ILSVRC' year '_devkit/'];
      addpath(genpath([dev '/evaluation/']));
      t=[dev '/data/meta_det.mat'];
      t=load(t); synsets=t.synsets(1:200); catNms={synsets.name};
      catsMap=containers.Map({synsets.WNID},1:length(catNms));
      if(~strcmp(split,'val')), blacklist=cell(1,2); else
        f=[dev '/data/' 'ILSVRC' year '_det_validation_blacklist.txt'];
        f=fopen(f); blacklist=textscan(f,'%d %s'); fclose(f);
        t=catsMap.values(blacklist{2}); blacklist{2}=[t{:}];
      end
      if(strcmp(split,'train'))
        dl=@(i) [dev '/data/det_lists/' split '_pos_' int2str(i) '.txt'];
        is=cell(1,200); for i=1:200, f=fopen(dl(i));
          is{i}=textscan(f,'%s %*s'); is{i}=is{i}{1}; fclose(f); end
        is=unique(cat(1,is{:})); n=length(is);
      else
        f=fopen([dev '/data/det_lists/' split '.txt']);
        is=textscan(f,'%s %*s'); is=is{1}; fclose(f); n=length(is);
      end
      data=CocoUtils.initData(catNms,n);
      for i=1:n
        f=[dataDir '/ILSVRC' year '_DET_bbox_' split '/' is{i} '.xml'];
        R=VOCreadxml(f); R=R.annotation; nm=[is{i} '.JPEG'];
        hw=str2double({R.size.height R.size.width});
        if(~isfield(R,'object')), catIds=[]; bbs=[]; else
          O=R.object; t=catsMap.values({O.name}); catIds=[t{:}];
          b=[O.bndbox]; bbs=str2double({b.xmin; b.ymin; b.xmax; b.ymax})';
        end
        j=blacklist{2}(blacklist{1}==i); m=numel(j); b=[0 0 hw(2) hw(1)];
        catIds=[j catIds]; bbs=[repmat(b,m,1); bbs]; %#ok<AGROW>
        ignore=ismember(catIds,j); iscrowd=ignore*0; iscrowd(1:m)=1;
        data=CocoUtils.addData(data,nm,i,hw,catIds,ignore,iscrowd,bbs);
      end
      f=fopen(annFile,'w'); fwrite(f,gason(data)); fclose(f);
      fprintf('DONE (t=%0.2fs).\n',toc(clk));
    end
    
    function convertPascalDt( srcFiles, tarFile )
      % Convert detections on PASCAL to COCO format.
      %
      % USAGE
      %  CocoUtils.convertPascalDt( srcFiles, tarFile )
      %
      % INPUTS
      %  srcFiles   - source detection file(s) in PASCAL format
      %  tarFile    - target detection file in COCO format
      if(exist(tarFile,'file')), return; end; R=[];
      for i=1:length(srcFiles), f=fopen(srcFiles{i},'r');
        R1=textscan(f,'%d %f %f %f %f %f'); fclose(f);
        [~,~,x0,y0,x1,y1]=deal(R1{:}); b=[x0-1 y0-1 x1-x0+1 y1-y0+1];
        b(:,3:4)=max(b(:,3:4),1); b=mat2cell(b,ones(1,size(b,1)),4);
        R=[R; struct('image_id',num2cell(R1{1}),'bbox',b,...
          'category_id',i,'score',num2cell(R1{2}))]; %#ok<AGROW>
      end
      f=fopen(tarFile,'w'); fwrite(f,gason(R)); fclose(f);
    end
    
    function convertImageNetDt( srcFile, tarFile )
      % Convert detections on ImageNet to COCO format.
      %
      % USAGE
      %  CocoUtils.convertImageNetDt( srcFile, tarFile )
      %
      % INPUTS
      %  srcFile    - source detection file in ImageNet format
      %  tarFile    - target detection file in COCO format
      if(exist(tarFile,'file')), return; end; f=fopen(srcFile,'r');
      R=textscan(f,'%d %d %f %f %f %f %f'); fclose(f);
      [~,~,~,x0,y0,x1,y1]=deal(R{:}); b=[x0-1 y0-1 x1-x0+1 y1-y0+1];
      b(:,3:4)=max(b(:,3:4),1); bbox=mat2cell(b,ones(1,size(b,1)),4);
      R=struct('image_id',num2cell(R{1}),'bbox',bbox,...
        'category_id',num2cell(R{2}),'score',num2cell(R{3}));
      f=fopen(tarFile,'w'); fwrite(f,gason(R)); fclose(f);
    end
    
    function validateOnPascal( dataDir )
      % Validate COCO eval code against PASCAL code.
      %
      % USAGE
      %  CocoUtils.validateOnPascal( dataDir )
      %
      % INPUTS
      %  dataDir    - dir containing VOCdevkit/
      split='val'; year='2007'; thrs=0:.001:1; T=length(thrs);
      dev=[dataDir '/VOCdevkit/']; addpath(genpath([dev '/VOCcode/']));
      d=pwd; cd(dev); VOCinit; cd(d); O=VOCopts; O.testset=split;
      O.detrespath=[O.detrespath(1:end-10) split '_%s.txt'];
      catNms=O.classes; K=length(catNms); ap=zeros(K,1);
      for i=1:K, [R,P]=VOCevaldet(O,'comp3',catNms{i},0); R1=[R; inf];
        P1=[P; 0]; for t=1:T, ap(i)=ap(i)+max(P1(R1>=thrs(t)))/T; end; end
      srcFile=[dev '/results/VOC' year '/Main/comp3_det_' split];
      resFile=[srcFile '.json']; annFile=[dev '/VOC2007/' split '.json'];
      sfs=cell(1,K); for i=1:K, sfs{i}=[srcFile '_' catNms{i} '.txt']; end
      CocoUtils.convertPascalGt(dataDir,year,split,annFile);
      CocoUtils.convertPascalDt(sfs,resFile);
      D=CocoApi(annFile); R=D.loadRes(resFile); E=CocoEval(D,R);
      p=E.params; p.recThrs=thrs; p.iouThrs=.5; p.areaRng=[0 inf];
      p.useSegm=0; p.maxDets=inf; E.params=p; E.evaluate(); E.accumulate();
      apCoco=squeeze(mean(E.eval.precision,2)); deltas=abs(apCoco-ap);
      fprintf('AP delta: mean=%.2e median=%.2e max=%.2e\n',...
        mean(deltas),median(deltas),max(deltas))
      if(max(deltas)>1e-2), msg='FAILED'; else msg='PASSED'; end
      warning(['Eval code *' msg '* validation!']);
    end
    
    function validateOnImageNet( dataDir )
      % Validate COCO eval code against ImageNet code.
      %
      % USAGE
      %  CocoUtils.validateOnImageNet( dataDir )
      %
      % INPUTS
      %  dataDir    - dir containing ILSVRC*/ folders
      warning(['Set pixelTolerance=0 in line 30 of eval_detection.m '...
        '(and delete cache) otherwise AP will differ by >1e-4!']);
      year='2013'; dev=[dataDir '/ILSVRC' year '_devkit/'];
      fs = { [dev 'evaluation/demo.val.pred.det.txt']
        [dataDir '/ILSVRC' year '_DET_bbox_val/']
        [dev 'data/meta_det.mat']
        [dev 'data/det_lists/val.txt']
        [dev 'data/ILSVRC' year '_det_validation_blacklist.txt']
        [dev 'data/ILSVRC' year '_det_validation_cache.mat'] };
      addpath(genpath([dev 'evaluation/']));
      ap=eval_detection(fs{:})';
      resFile=[fs{1}(1:end-3) 'json'];
      annFile=[dev 'data/ILSVRC' year '_val.json'];
      CocoUtils.convertImageNetDt(fs{1},resFile);
      CocoUtils.convertImageNetGt(dataDir,year,'val',annFile)
      D=CocoApi(annFile); R=D.loadRes(resFile); E=CocoEval(D,R);
      p=E.params; p.recThrs=0:.0001:1; p.iouThrs=.5; p.areaRng=[0 inf];
      p.useSegm=0; p.maxDets=inf; E.params=p; E.evaluate(); E.accumulate();
      apCoco=squeeze(mean(E.eval.precision,2)); deltas=abs(apCoco-ap);
      fprintf('AP delta: mean=%.2e median=%.2e max=%.2e\n',...
        mean(deltas),median(deltas),max(deltas))
      if(max(deltas)>1e-4), msg='FAILED'; else msg='PASSED'; end
      warning(['Eval code *' msg '* validation!']);
    end
    
    function generateFakeDt( coco, dtFile, varargin )
      % Generate fake detections from ground truth.
      %
      % USAGE
      %  CocoUtils.generateFakeDt( coco, dtFile, varargin )
      %
      % INPUTS
      %  coco       - instance of CocoApi containing ground truth
      %  dtFile     - target file for writing detection results
      %  params     - parameters (struct or name/value pairs)
      %   .n          - [100] number images for which to generate dets
      %   .fn         - [.20] false negative rate (0<fn<1)
      %   .fp         - [.10] false positive rate (0<fp<fn)
      %   .sigma      - [.10] translation noise (relative to object width)
      %   .seed       - [0] random seed for reproducibility
      %   .type       - ['bbox'] can be 'bbox', 'segm', or 'keypoints'
      fprintf('Generating fake detection data...    '); clk=tic;
      def={'n',100,'fn',.20,'fp',.10,'sigma',.10,'seed',0,'type','bbox'};
      opts=getPrmDflt(varargin,def,1); n=opts.n;
      if(strcmp(opts.type,'segm')), opts.type='segmentation'; end
      assert(any(strcmp(opts.type,{'bbox','segmentation','keypoints'})));
      rstream = RandStream('mrg32k3a','Seed',opts.seed); k=n*100;
      R=struct('image_id',[],'category_id',[],opts.type,[],'score',[]);
      imgIds=sort(coco.getImgIds()); imgIds=imgIds(1:n); R=repmat(R,1,k);
      imgs=coco.loadImgs(imgIds); catIds=coco.getCatIds(); k=0;
      for i=1:n
        A=coco.loadAnns(coco.getAnnIds('imgIds',imgIds(i),'iscrowd',0));
        m=length(A); h=imgs(i).height; w=imgs(i).width;
        for j=1:m, t=rand(rstream);
          if(t<opts.fp), catId=catIds(randi(rstream,length(catIds)));
          elseif(t<opts.fn), continue; else catId=A(j).category_id; end
          bb=A(j).bbox; dx=round(randn(rstream)*opts.sigma*bb(3));
          if( strcmp(opts.type,'bbox') )
            x0=max(0,bb(1)+dx); x1=min(w-1,bb(1)+bb(3)+dx-1);
            bb(1)=x0; bb(3)=x1-x0+1; if(bb(3)==0), continue; end; o=bb;
          elseif( strcmp(opts.type,'segmentation') )
            M=MaskApi.decode(MaskApi.frPoly(A(j).segmentation,h,w)); T=M*0;
            T(:,max(1,1+dx):min(w,w+dx))=M(:,max(1,1-dx):min(w,w-dx));
            if(nnz(T)==0), continue; end; o=MaskApi.encode(T);
          elseif( strcmp(opts.type,'keypoints') )
            o=A(j).keypoints; v=o(3:3:end)>0; if(~any(v)), continue; end
            x=o(1:3:end); y=o(2:3:end); x(~v)=mean(x(v)); y(~v)=mean(y(v));
            x=max(0,min(w-1,x+dx)); o(1:3:end)=x; o(2:3:end)=y;
          end
          k=k+1; R(k).image_id=imgIds(i); R(k).category_id=catId;
          R(k).(opts.type)=o; R(k).score=round(rand(rstream)*1000)/1000;
        end
      end
      R=R(1:k); f=fopen(dtFile,'w'); fwrite(f,gason(R)); fclose(f);
      fprintf('DONE (t=%0.2fs).\n',toc(clk));
    end
    
    function validateMaskApi( coco )
      % Validate MaskApi against Matlab functions.
      %
      % USAGE
      %  CocoUtils.validateMaskApi( coco )
      %
      % INPUTS
      %  coco       - instance of CocoApi containing ground truth
      S=coco.data.annotations; S=S(~[S.iscrowd]); S={S.segmentation};
      h=1000; n=1000; Z=cell(1,n); A=Z; B=Z; M=Z; IB=zeros(1,n);
      fprintf('Running MaskApi implementations...   '); clk=tic;
      for i=1:n, A{i}=MaskApi.frPoly(S{i},h,h); end
      Ia=MaskApi.iou(A{1},[A{:}]);
      fprintf('DONE (t=%0.2fs).\n',toc(clk));
      fprintf('Running Matlab implementations...    '); clk=tic;
      for i=1:n, M1=0; for j=1:length(S{i}), x=S{i}{j}+.5;
          M1=M1+poly2mask(x(1:2:end),x(2:2:end),h,h); end
        M{i}=uint8(M1>0); B{i}=MaskApi.encode(M{i});
        IB(i)=sum(sum(M{1}&M{i}))/sum(sum(M{1}|M{i}));
      end
      fprintf('DONE (t=%0.2fs).\n',toc(clk));
      if(isequal(A,B)&&isequal(Ia,IB)),
        msg='PASSED'; else msg='FAILED'; end
      warning(['MaskApi *' msg '* validation!']);
    end
    
    function gasonSplit( name, k )
      % Split JSON file into multiple JSON files.
      %
      % Splits file 'name.json' into multiple files 'name-*.json'. Only
      % works for JSON arrays. Memory efficient. Inverted by gasonMerge().
      %
      % USAGE
      %  CocoUtils.gasonSplit( name, k )
      %
      % INPUTS
      %  name       - file containing JSON array (w/o '.json' ext)
      %  k          - number of files to split JSON into
      s=gasonMex('split',fileread([name '.json']),k); k=length(s);
      for i=1:k, f=fopen(sprintf('%s-%06i.json',name,i),'w');
        fwrite(f,s{i}); fclose(f); end
    end
    
    function gasonMerge( name )
      % Merge JSON files into single JSON file.
      %
      % Merge files 'name-*.json' into single file 'name.json'. Only works
      % for JSON arrays. Memory efficient. Inverted by gasonSplit().
      %
      % USAGE
      %  CocoUtils.gasonMerge( name )
      %
      % INPUTS
      %  name       - files containing JSON arrays (w/o '.json' ext)
      s=dir([name '-*.json']); s=sort({s.name}); k=length(s);
      p=fileparts(name); for i=1:k, s{i}=fullfile(p,s{i}); end
      for i=1:k, s{i}=fileread(s{i}); end; s=gasonMex('merge',s);
      f=fopen([name '.json'],'w'); fwrite(f,s); fclose(f);
    end
  end
  
  methods( Static, Access=private )
    function data = initData( catNms, n )
      % Helper for convert() functions: init annotations.
      m=length(catNms); ms=num2cell(1:m);
      I = struct('file_name',0,'height',0,'width',0,'id',0);
      C = struct('supercategory','none','id',ms,'name',catNms);
      A = struct('segmentation',0,'area',0,'iscrowd',0,...
        'image_id',0,'bbox',0,'category_id',0,'id',0,'ignore',0);
      I=repmat(I,1,n); A=repmat(A,1,n*20);
      data = struct('images',I,'type','instances',...
        'annotations',A,'categories',C,'nImgs',0,'nAnns',0);
    end
    
    function data = addData( data,nm,id,hw,catIds,ignore,iscrowd,bbs )
      % Helper for convert() functions: add annotations.
      data.nImgs=data.nImgs+1;
      data.images(data.nImgs)=struct('file_name',nm,...
        'height',hw(1),'width',hw(2),'id',id);
      for j=1:length(catIds), data.nAnns=data.nAnns+1; k=data.nAnns;
        b=bbs(j,:); b=b-1; b(3:4)=b(3:4)-b(1:2)+1;
        x1=b(1); x2=b(1)+b(3); y1=b(2); y2=b(2)+b(4);
        S={{[x1 y1 x1 y2 x2 y2 x2 y1]}}; a=b(3)*b(4);
        data.annotations(k)=struct('segmentation',S,'area',a,...
          'iscrowd',iscrowd(j),'image_id',id,'bbox',b,...
          'category_id',catIds(j),'id',k,'ignore',ignore(j));
      end
      if( data.nImgs == length(data.images) )
        data.annotations=data.annotations(1:data.nAnns);
        data=rmfield(data,{'nImgs','nAnns'});
      end
    end
  end
  
end
