function [ bboxes, scores ] = runDetector( im, weight, bias, window_size, cell_size, block_size, nbins )
%RUNDETECTOR Given an image, runs the SVM detector and outputs bounding
% boxes and scores
% Arguments:
%          im - the image matrix
%          weight - weight matrix trained by SVM
%          bias - bias vector trained by SVM
%          window_size - an array which contains the height and width
%                       of the sliding window
%          cell_size - each cell will be of size (cell_size, cell_size)
%                       pixels
%          block_size - each block will be of size (block_size, block_size)
%                       cells
%          nbins - number of histogram bins
% Returns:
%          bboxes - D x 4 bounding boxes that tell [xmin ymin xmax ymax]
%               per bounding box
%          scores - the SVM scores associated with each bounding box in bboxes
%

width = window_size(2); height = window_size(1);
total_block_size = block_size * cell_size;
num_blocks = [floor((window_size(1)-2)/(total_block_size/2)) - 1, ...
	floor((window_size(2)-2)/(total_block_size/2)) - 1];

top_bboxes = [];
top_scores = [];

test_features = computeHOGFeatures(im,cell_size,block_size,nbins);
scores = zeros(size(test_features,1)-num_blocks(1), size(test_features,2)-num_blocks(2));

for h=1:size(test_features,1)-num_blocks(1)
	for w=1:size(test_features,2)-num_blocks(2)
		features = test_features(h:h+num_blocks(2)-1, w:w+num_blocks(2)-1,:);
		features = reshape(features, 1,[]); 
		scores(h,w) = features*weight + bias;
	end
end

%Create Nx4 bbox and scores matrices only if positive
scores2 = reshape(scores, [],1);
a = repmat((1:size(scores,1))', size(scores,2),1)-1;
a=a*cell_size+1;
b = floor((0:size(scores2)-1)'/size(scores,1))+1;
b = b*cell_size;
bbox = [b,a,b+width,a+height];
ind = scores2 > 1;
bbox = bbox(ind,:);
scores3 = scores2(ind,:);
top_bboxes = [top_bboxes; bbox];
top_scores = [top_scores; scores3];

% only for part d)
nmax = nonmaxSuppress(top_bboxes, top_scores, size(im));
top_bboxes = top_bboxes(nmax,:);
top_scores = top_scores(nmax,:);

bboxes = top_bboxes;
scores = top_scores;
end





