%%  John's Dendritic Spine Voxelization

function voxels = GenerateDendriticSpineVoxels() % should specify folders here, etc.

clc; clear all; close all;
cd('/Users/johnlambert/Documents/Stanford_2015-2016/SpringQuarter_2015-2016/CS_231A/FinalProject');
allFiles = dir('/Users/johnlambert/Documents/Stanford_2015-2016/SpringQuarter_2015-2016/CS_231A/FinalProject/RoiSet_June_6_2011_415-5_s8');
allNames = { allFiles.name };
%disp( allNames );
numFilesInDir = size(allNames,2);
coordinates = [];
dirString = '/Users/johnlambert/Documents/Stanford_2015-2016/SpringQuarter_2015-2016/CS_231A/FinalProject/RoiSet_June_6_2011_415-5_s8';

maxZSliceForZStack = -1
for i = 1:100
    zSliceNumAsString = num2str( i );
    if size( zSliceNumAsString ) == 1
        zSliceNumAsString = strcat( '0', zSliceNumAsString );
    end
    imageName = sprintf( 'June_6_2011__415-5__Series008_z0%s_ch00.tif', zSliceNumAsString );
    imagePath = strcat( 'June_6_2011_415-5_Series008/', imageName );
    if exist( imagePath , 'file') == 2
        maxZSliceForZStack = i;
    end
end

% NEED TO DETERMINE IF ALWAYS 50, OR SOMETIMES 38
voxels = zeros( 1024, 1024, maxZSliceForZStack + 1 ); % Will give me 52,428,800 voxels

% I MAP THE ZERO'TH Z-SLICE TO THE NUMBER 1

for fileIdx= 3:numFilesInDir
    disp( 'one roi file...');
    disp( fileIdx );
    s = strcat( dirString, '/' );
    s = strcat( s , allNames(1, fileIdx) );
    [sROI] = ReadImageJROI( s );
    fullstring = char( allNames(1, fileIdx) );
    substring = fullstring(1:4);
    zSliceNumber = str2num( substring );
	x = sROI{1,1}.mnCoordinates(:,1);
    y = sROI{1,1}.mnCoordinates(:,2); 
    k = convhull( x , y);
    %plot( x(k), y(k), 'c.' );
    zSliceNumAsString = num2str( zSliceNumber);
    if size( zSliceNumAsString ) == 1
        zSliceNumAsString = strcat( '0', zSliceNumAsString );
    end
    imageName = sprintf( 'June_6_2011__415-5__Series008_z0%s_ch00.tif', zSliceNumAsString );
    imagePath = strcat( 'June_6_2011_415-5_Series008/', imageName );
    if exist( imagePath , 'file') == 2
        I = imread( imagePath);
        segmentedRegion = roipoly(I, (x(k))', (y(k))' );
    end
    for h=1:1024
        for w=1:1024
            if segmentedRegion(h,w) == 1
                maxAbove = zSliceNumber + 4;
                minBelow = zSliceNumber - 4;
                if maxAbove > (maxZSliceForZStack + 1)
                    maxAbove = (maxZSliceForZStack + 1);
                end
                if minBelow < 0
                    minBelow = 0;
                end
                zSliceIndices = minBelow:maxAbove;
                voxels(h,w, [ zSliceIndices ] ) = 1;
            end
        end
    end
end
% coordinates = [];
% for x = 1:size(voxels,1)
%     for y = 1: size(voxels,2)
%         for z = 1:size(voxels,3)
%             if voxels(y,x,z) == 1
%                 coordinates = [ coordinates ; [ x, y, z ] ];
%             end
%         end
%     end
% end
% X = coordinates;
% S = repmat([5,5,5],numel( coordinates ),1);
% s = S(:);
% % Create a 3-D scatter plot and use view to change the angle of the axes in the figure.
% figure;
% scatter3( coordinates(:,1) , coordinates(:,2), coordinates(:,3) );
% axis([0 1024 0 1024 0 50 ]);
% grid on;
% view( 10, 10);
% title( '3D-Voxelization of 39 Z-Slices' );
% xlabel( 'X axis' ), ylabel( 'Y axis' ), zlabel( 'Z axis' );
% text(0,0,0, 'Origin');

disp( 'size of voxels' )
disp( size(voxels,1) );
disp( size(voxels,2) );
disp( size(voxels,3) );
%imshow( voxels(:,:,9) );
save('dendriticSpines.mat','voxels' ,'-v7.3')
end






        %figure;
        %imshow(I);
        %hold on;
        %imshow( segmentedRegion ); % Compute Binary Pixel ROI from Image
        %hold off;







    %disp( allNames(1, fileIdx) );
    %disp( zSliceNumber );
    %numCoordinates = size( segmentedRegion , 1);
    %copies_zSliceNumber = zSliceNumber * ones( numCoordinates, 1 );
    %threeDimCoordsForSlice = [ sROI{1,1}.mnCoordinates copies_zSliceNumber ];
    %plot(x(k),y(k),'LineWidth',3,'Color','r');
    %plot( x(k) , y(k) ,'c','LineWidth',2);
    %hold off;
%   plot(x(k),y(k),'r-',x,y,'b*')
    %coordinates = [ coordinates ; threeDimCoordsForSlice ];
















% numRoiPoints = size( coordinates, 1 );
% maxZSlice = coordinates( numRoiPoints, 3 );
% for imageIdx = 0:maxZSlice
%     zSliceNumber = num2str( imageIdx );
%     if size(zSliceNumber) == 1
%         zSliceNumber = strcat( '0', zSliceNumber );
%     end
%     imageName = sprintf( 'June_6_2011__415-5__Series008_z0%s_ch00.tif', zSliceNumber);
%     imagePath = strcat( 'June_6_2011_415-5_Series008/', imageName );
%     if exist( imagePath , 'file') == 2
%         figure( 'position', [0, 0, 600, 600 ] );
%         imshow( imagePath );
%         hold on;
%         % ADD LOGIC FOR CASE WHEN -4 WILL BE OUT OF BOUNDS...
%         indicesBelow = find( coordinates( : , 3 ) >= (imageIdx - 4) );
%         indicesAbove = find( coordinates( : , 3 ) <= (imageIdx + 4) );
%         indices = intersect( indicesBelow , indicesAbove );
%         
%         pointsThatCorrespondToThisImage = coordinates( indices, : );
%         x = pointsThatCorrespondToThisImage(:,1)
%         y = pointsThatCorrespondToThisImage(:,2)
%         if size( pointsThatCorrespondToThisImage,1) ~= 0
%             
% %             x = [63 186 54 190 63]; % TURN THIS INTO REGION, GET VOXELS
% %             y = [60 60 209 204 60];
% %             bw = poly2mask(x,y,256,256);
% %             imshow(bw)
% %             hold on
% %             plot(x,y,'b','LineWidth',2)
% %             hold off
%             
%             %POLYGON
%             plot(x(k),y(k),'r-',x,y,'b*')
%         else
%             plot( x, y ,  'r.') % '.',
%         end
% %         hold on;
% %         centerOfMassIndices = find( XYPoints( :, 3) == imageIdx );
% %         centerOfMassPointsForThisImage = XYPoints( centerOfMassIndices, : );
% %         plot( centerOfMassPointsForThisImage(:,1), centerOfMassPointsForThisImage(:,2) , 'c.' , 'markersize', 50 );
%         %disp( 'centerOfMassIndices' );
%         %disp( centerOfMassIndices );
%     end
% end




% SCROLL OVER ONE DIR
% PLOT AT THAT Z-COORDINATE THE slice

%for i = [train_indices, test_indices]
    %pathname = sprintf('car_epfl/images/%04d.jpg',data(i,1)); 
%end

% read in data from 2D files.
% Read in the points for the first image.
% figure;
% plot3( 2 , 2, 2 , 2, 2, 3  )
% clc; close all;

% [X,Y,Z] = sphere(16);
% X = [ 2, 2, 2 ]
% Y = [ 3, 3, 3 ]
% Z = [ 1, 2, 3 ]
% x = [0.5*X(:); 0.75*X(:); X(:)];
% y = [0.5*Y(:); 0.75*Y(:); Y(:)];
% z = [0.5*Z(:); 0.75*Z(:); Z(:)];
% Define vector s to specify the marker sizes.

% res = max(voxels(2,:) - voxels(1,:));
% ux = unique(voxels(:,1));
% uy = unique(voxels(:,2));
% uz = unique(voxels(:,3));
% 
% % Expand the model by one step in each direction
% ux = [ux(1)-res; ux; ux(end)+res];
% uy = [uy(1)-res; uy; uy(end)+res];
% uz = [uz(1)-res; uz; uz(end)+res];
% 
% % Convert to a grid
% [X,Y,Z] = meshgrid( ux, uy, uz );
% 
% % Create an empty voxel grid, then fill only those elements in voxels
% V = zeros( size( X ) );
% N = size(voxels,1);
% for ii=1:N
%     ix = (ux == voxels(ii,1));
%     iy = (uy == voxels(ii,2));
%     iz = (uz == voxels(ii,3));
%     V(iy,ix,iz) = 1;
% end
% 
% % Now draw it
% ptch = patch( isosurface( X, Y, Z, V, 0.5 ) );
% isonormals( X, Y, Z, V, ptch )
% set( ptch, 'FaceColor', 'r', 'EdgeColor', 'none' );
% 
% set(gca,'DataAspectRatio',[1 1 1]);
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% view(-140,22)
% lighting( 'gouraud' )
% camlight( 'right' )
% axis( 'tight' )


% fileid = fopen('roi_points_June_6_2011_415-5_Series008_z009_ch00.txt','r');
% PointNo = str2num(fgets(fileid));
% temp = fscanf(fileid,'%f');
% Points1 = zeros(PointNo, 2);
% for i = 1 : PointNo
%     Points1(i,2) = temp(i*2);
%     Points1(i,1) = temp(i*2-1);
% end
% fclose(fileid);



% fileid = fopen( 'xy_coords_June_6_415-5_Series8.txt' ,'r');
% temp = fscanf(fileid,'%f');
% PointNo = 122;
% XYPoints = zeros( PointNo , 3);
% for i = 1 : PointNo
%     XYPoints(i,3) = temp(i*3 - 2); 
%     XYPoints(i,1) = temp(i*3 - 1);
%     XYPoints(i,2) = temp(i*3);
% end
% fclose(fileid);


%                 voxels(h,w,zSliceNumber - 3 ) = 1;
%                 voxels(h,w,zSliceNumber - 2 ) = 1;
%                 voxels(h,w,zSliceNumber - 1 ) = 1;
%                 voxels(h,w,zSliceNumber + 0 ) = 1;
%                 voxels(h,w,zSliceNumber + 1 ) = 1;
%                 voxels(h,w,zSliceNumber + 2 ) = 1;
%                 voxels(h,w,zSliceNumber + 3 ) = 1;
%                 voxels(h,w,zSliceNumber + 4 ) = 1;
%                 voxels(h,w,zSliceNumber + 5 ) = 1;

