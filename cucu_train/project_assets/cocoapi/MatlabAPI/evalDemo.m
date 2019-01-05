%% Demo demonstrating the algorithm result formats for COCO

%% select results type for demo (either bbox or segm)
type = {'segm','bbox','keypoints'}; type = type{1}; % specify type here
fprintf('Running demo for *%s* results.\n\n',type);

%% initialize COCO ground truth api
dataDir='../'; prefix='instances'; dataType='val2014';
if(strcmp(type,'keypoints')), prefix='person_keypoints'; end
annFile=sprintf('%s/annotations/%s_%s.json',dataDir,prefix,dataType);
cocoGt=CocoApi(annFile);

%% initialize COCO detections api
resFile='%s/results/%s_%s_fake%s100_results.json';
resFile=sprintf(resFile,dataDir,prefix,dataType,type);
cocoDt=cocoGt.loadRes(resFile);

%% visialuze gt and dt side by side
imgIds=sort(cocoGt.getImgIds()); imgIds=imgIds(1:100);
imgId = imgIds(randi(100)); img = cocoGt.loadImgs(imgId);
I = imread(sprintf('%s/images/val2014/%s',dataDir,img.file_name));
figure(1); subplot(1,2,1); imagesc(I); axis('image'); axis off;
annIds = cocoGt.getAnnIds('imgIds',imgId); title('ground truth')
anns = cocoGt.loadAnns(annIds); cocoGt.showAnns(anns);
figure(1); subplot(1,2,2); imagesc(I); axis('image'); axis off;
annIds = cocoDt.getAnnIds('imgIds',imgId); title('results')
anns = cocoDt.loadAnns(annIds); cocoDt.showAnns(anns);

%% load raw JSON and show exact format for results
fprintf('results structure have the following format:\n');
res = gason(fileread(resFile)); disp(res)

%% the following command can be used to save the results back to disk
if(0), f=fopen(resFile,'w'); fwrite(f,gason(res)); fclose(f); end

%% run COCO evaluation code (see CocoEval.m)
cocoEval=CocoEval(cocoGt,cocoDt,type);
cocoEval.params.imgIds=imgIds;
cocoEval.evaluate();
cocoEval.accumulate();
cocoEval.summarize();

%% generate Derek Hoiem style analyis of false positives (slow)
if(0), cocoEval.analyze(); end
