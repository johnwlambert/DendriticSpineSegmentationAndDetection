function [ angles, magnitudes ] = computeGradient( im )
%COMPUTEGRADIENT Given an image, computes the pixel gradients
% Arguments:
%          im - an image matrix read in by im2read (size H X W X C)
%
% Returns:
%          angles - (H-2) x (W-2) matrix of gradient angles
%          magnitudes - (H-2) x (W-2) matrix of gradient magnitudes
%

height=size(im,1);
width=size(im,2);

% compute gradient in x and y direction
gradx = zeros(height-2, width-2);
grady = zeros(height-2, width-2);

for i=1:height-2
	A = im(i,2:end-1,:)-im(i+2,2:end-1,:); [~,index] = max(abs(A),[],3);
	absmax = zeros(size(A,1), size(A,2)); for j =1:numel(index)
	absmax(:,j) = A(:,j,index(j));
	end
	grady(i,:)=absmax;
end
for i=1:width-2
	A = im(2:end-1,i,:)-im(2:end-1,i+2,:); 
	[~,index] = max(abs(A),[],3);
	absmax = zeros(size(A,1), size(A,2)); 
	for j =1:numel(index)
		absmax(j,:) = A(j,:,index(j));
	end
	gradx(:,i)=absmax;
end

% compute angle and magnitude
angles=atand(gradx./grady);
angles=90 - angles;
magnitudes=sqrt(gradx.^2 + grady.^2);

% Remove redundant pixels in an image.
angles(isnan(angles))=0;
magnitudes(isnan(magnitudes))=0;
end
