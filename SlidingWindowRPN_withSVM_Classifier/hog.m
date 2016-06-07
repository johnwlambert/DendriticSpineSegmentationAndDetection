close all; clear all; clc

cd( '/Users/johnlambert/Documents/Stanford_2015-2016/SpringQuarter_2015-2016/CS_231a/FinalProject/HOG_SlidingWindow/hog' );

run('vlfeat-0.9.20/toolbox/vl_setup');
%% PART A: Checking Image Gradient
%im = im2single(imread('data/simple.jpg'));
%[grad_angle, grad_magnitude] = computeGradient(im)
%% PART B: Generating HOG Features
%im = im2single(imread('data/car.jpg'));
%im = im2single(imread('data/June_6_2011_415_5_Series005_z024_ch00.jpg'));
% im = im2single(imread('data/June_6_2011_415-5_Series008_z009_ch00.tif'));

% vary cell size to change the output feature vector
% block size and number of bins are generally defaulted to 2 and 9
cell_size = 8;
block_size = 2;
nbins = 9;

%car_hog_feat = computeHOGFeatures(im, cell_size, block_size, nbins);
%show_hog(car_hog_feat);
%% PART C & D: Sliding Window & Nonmaximal Suppression, Setup
rng(1);
window_size = [36 36];
cell_size = 6;

% train the svm
features_pos = get_positive_features('data/caltech_faces/Caltech_CropFaces', cell_size, window_size, block_size, nbins);
num_negative_examples = 10; %Higher will work strictly better, but takes longer
features_neg = get_random_negative_features( 'data/train_non_face_scenes', cell_size, window_size, block_size, nbins, num_negative_examples);
X = [features_pos;features_neg;];
Y = [ones(size(features_pos,1),1); -1*ones(size(features_neg,1),1)];
[weight bias] = vl_svmtrain(X', Y', .0001);

%% PART C & D: Sliding Window & Nonmaximal Suppression, Simple Detector
im = im2single(imread('data/people.jpg'));
[bboxes, scores] = runDetector(im, weight, bias, window_size, cell_size, block_size, nbins);

% Plot boxes
bboxes = [bboxes(:,1:2), bboxes(:,3:4)-bboxes(:,1:2)];
figure(2);hold on;
imshow(im);
for i =1:size(scores)
    rectangle('Position', bboxes(i,:), 'EdgeColor','r', 'LineWidth', 2);
end

%% PART C & D: Sliding Window & Nonmaximal Suppression, Complex Detector
im = im2single(rgb2gray(imread('data/people3.jpg')));
[bboxes, scores] = runDetector(im, weight, bias, window_size, cell_size, block_size, nbins);

% Plot boxes
bboxes = [bboxes(:,1:2), bboxes(:,3:4)-bboxes(:,1:2)];
figure(3);
imshow(im);hold on;
for i =1:size(scores)
    rectangle('Position', bboxes(i,:), 'EdgeColor','r', 'LineWidth', 2);
end
