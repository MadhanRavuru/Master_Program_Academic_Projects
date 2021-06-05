clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = '../data/detection';
% path to object ply file
object_path = '../data/teabox.ply';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Load the SIFT model from the previous task
load('sift_model.mat');


% TODO: setup camera intrinsic parameters using cameraParameters()
IntrinsicMatrix = [2960.37845 0 0; 0 2960.37845 0; 1841.68855 1235.23369 1]; 
camera_params = cameraParameters('IntrinsicMatrix',IntrinsicMatrix);

%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);


%% Match SIFT features of new images to the SIFT model with features computed in the task 1
% You should use VLFeat function vl_ubcmatch()

% Place SIFT keypoints and descriptors of new images here
keypoints=cell(num_files,1);
descriptors=cell(num_files,1);
% Place matches between new SIFT features and SIFT features from the SIFT
% model here
sift_matches=cell(num_files,1);

% Default threshold for SIFT keypoints matching: 1.5 
% When taking higher value, match is only recognized if similarity is very high
threshold_ubcmatch = 1.5; 


 for i=1:num_files
     fprintf('Calculating and matching sift features for image: %d \n', i)
      
% % %     TODO: Prepare the image (img) for vl_sift() function
     img = imread(char(Filenames(i)));
     [keypoints{i}, descriptors{i}] = vl_sift(single(rgb2gray(img)));
% %     Match features between SIFT model and SIFT features from new image
     sift_matches{i} = vl_ubcmatch(descriptors{i}, model.descriptors, threshold_ubcmatch); 
 end
 

% % % Save sift features, descriptors and matches and load them when you rerun the code to save time
   save('sift_matches.mat', 'sift_matches');
   save('detection_keypoints.mat', 'keypoints')
   save('detection_descriptors.mat', 'descriptors')

 load('sift_matches.mat')
 load('detection_keypoints.mat')
 load('detection_descriptors.mat')


%% PnP and RANSAC 
% Implement the RANSAC algorithm featuring also the following arguments:
% Reprojection error threshold for inlier selection - 'threshold_ransac'  
% Number of RANSAC iterations - 'ransac_iterations'

% Pseudocode
% i Randomly select a sample of 4 data points from S and estimate the pose using PnP.
% ii Determine the set of data points Si from all 2D-3D correspondences 
%   where the reprojection error (Euclidean distance) is below the threshold (threshold_ransac). 
%   The set Si is the consensus set of the sample and defines the inliers of S.
% iii If the number of inliers is greater than we have seen so far,
%   re-estimate the pose using Si and store it with the corresponding number of inliers.
% iv Repeat the above mentioned procedure for N iterations (ransac_iterations).

% For PnP you can use estimateWorldCameraPose() function
% but only use it with 4 points and set the 'MaxReprojectionError' to the
% value of 10000 so that all these 4 points are considered to be inliers

% Place camera orientations, locations and best inliers set for every image here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);
best_inliers_set = cell(num_files, 1);

ransac_iterations = 500; 
threshold_ransac = 8;

for i = 1:num_files
    fprintf('Running PnP+RANSAC for image: %d \n', i)
   
%     TODO: Implement the RANSAC algorithm here

    
    for iter = 1:ransac_iterations
       
        
        image_points = zeros(4,2);
        world_points = zeros(4,3);
    
        % randomnly select 4 samples from sift matches
        num_samples = 4;  
    
        perm = randperm(size(sift_matches{i}(1,:),2)) ;
        sel = perm(1:num_samples);
        sample_1 = zeros(4,1);
        sample_2 = zeros(4,1);
        for j = 1:num_samples
            sample_1(j) = sift_matches{i}(1,sel(j));  % detection sample set
            sample_2(j) = sift_matches{i}(2,sel(j));  % model sample set
        end
        
        for j = 1:num_samples
            image_points(j,1) = keypoints{i}(1,sample_1(j)); %image u-coordinates for detection sample
            image_points(j,2) = keypoints{i}(2,sample_1(j)); %image v-coordinates for detection sample
            world_points(j,:) = model.coord3d(sample_2(j),:); %model coordinates for sample
        end
    
        max_reproj_err = 10000;
        
        % exception handling 
        try
            [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i)] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', max_reproj_err);
        catch ME
            if(strcmp(ME.identifier,'Could not find enough inliers in imagePoints and worldPoints.'))    
                break;
            end
        end
        
         % Getting 2d-3d correspondences from R,T
        [points] = project3d2image(model.coord3d(sift_matches{i}(2,:),:)', camera_params, cam_in_world_orientations(:,:,i),cam_in_world_locations(:, :, i));
      
        sift_matched_2d = keypoints{i}(1:2,sift_matches{i}(1,:)); % matching keypoints in 2D
        inlier_count = 0;
        indices = [];                        %This stores the indices of sift matches fulfilling dist < threshold_ransac
        for m = 1 : size(points,2)
            dist = norm(sift_matched_2d(:,m) - points(:,m));
            if(dist < threshold_ransac)
                inlier_count = inlier_count + 1;
                indices = cat(1,indices,m);    
            end
        end
        
        num_inliers = inlier_count;
        if(iter == 1)
            prev_inliers = num_inliers;
            best_orientation = cam_in_world_orientations(:,:,i);
            best_translation = cam_in_world_locations(:,:,i);
            inliers_set_indices = indices;
        elseif(num_inliers > prev_inliers)    % Updating the results    
            prev_inliers = num_inliers;
            best_orientation = cam_in_world_orientations(:,:,i);
            best_translation = cam_in_world_locations(:,:,i);
            inliers_set_indices = indices;
        end
    end
    %Storing the best results for each file
    cam_in_world_orientations(:,:,i) = best_orientation;
    cam_in_world_locations(:,:,i) = best_translation;
    best_inliers_set{i} = inliers_set_indices;  
   
end


 


%% Visualize inliers and the bounding box

% You can use the visualizations below or create your own one
% But be sure to present the bounding boxes drawn on the image to verify
% the camera pose

edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];

for i=1:num_files
    
    figure()
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    
%   Plot inliers set
    PlotInlierOutlier(best_inliers_set{i}, camera_params, sift_matches{i}, model.coord3d, keypoints{i}, cam_in_world_orientations(:,:,i), cam_in_world_locations(:,:,i))
%   Plot bounding box
    points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
    end
    hold off;
end
