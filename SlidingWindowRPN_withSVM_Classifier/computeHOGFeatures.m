function [ features ] = computeHOGFeatures( im, cell_size, block_size, nbins )
%COMPUTEHOGFEATURES Computes the histogram of gradients features
% Arguments:
%          im - the image matrix
%          cell_size - each cell will be of size (cell_size, cell_size)
%                       pixels
%          block_size - each block will be of size (block_size, block_size)
%                       cells
%          nbins - number of histogram bins
% Returns:
%          features - the hog features of the image (H_blocks x W_blocks x cell_size*cell_size*nbins)
%
% Generating the HoG features can be done as follows:
height=size(im,1)-2;
width=size(im,2)-2;

[angles, magnitudes] = computeGradient(im);

total_block_size = block_size * cell_size;
features = zeros(floor(height/(total_block_size/2)) - 1, ...
	floor(width/(total_block_size/2)) -1, nbins*block_size*block_size);
%iterate over the blocks, 50% overlap
for w=1:(total_block_size/2):width-total_block_size+1
	for h = 1:(total_block_size/2):height-total_block_size+1
		block_features = [];
		block_magnitude = magnitudes(h:h+total_block_size-1, ...
			w:w+total_block_size-1);
		block_angle = angles(h:h+total_block_size-1, ...
			w:w+total_block_size-1);

		% iterate over the cells
		for i=0:block_size-1
			for j=0:block_size-1
				cell_magnitude = block_magnitude(i*cell_size+1:(i+1)*cell_size, j*cell_size+1:(j+1)*cell_size);
				cell_angle = block_angle(i*cell_size+1:(i+1)*cell_size, ...
					j*cell_size+1:(j+1)*cell_size);

				histograms = zeros(1, nbins);

				% iterate over the pixels
				for p = 1:cell_size
					for q = 1:cell_size
						ang = cell_angle(p,q);
						if ang >= 180
							ang = ang - 180;
						end
						% interpolate the votes
						lower_idx = round(ang / (180/nbins));
						upper_idx = lower_idx + 1;

						lower_ang = (lower_idx-1) * (180/nbins) + 90/nbins;
						upper_ang = (upper_idx-1) * (180/nbins) + 90/nbins;
						if upper_idx > nbins
							upper_idx = 1;
						end
						if lower_idx == 0
							lower_idx = nbins;
						end
						lower_ang = abs(ang - lower_ang);
						upper_ang = abs(ang - upper_ang);
						lower_percent = upper_ang / (lower_ang+upper_ang); 
						upper_percent = lower_ang / (lower_ang+upper_ang); 
						histograms(lower_idx) = histograms(lower_idx) + lower_percent * cell_magnitude(p,q); 
						histograms(upper_idx) = histograms(upper_idx) + upper_percent * cell_magnitude(p,q);
					end
				end
				block_features = [block_features histograms];

			end
		end
		block_features=block_features/sqrt(norm(block_features)^2+.01);
		features(ceil(h/(total_block_size/2)),ceil(w/(total_block_size/2)),:) = block_features;

	end
end

features(isnan(features))=0; %Removing Infinitiy values
end
