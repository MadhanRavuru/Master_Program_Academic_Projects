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

% for i=1:num_files
%     fprintf('Calculating and matching sift features for image: %d \n', i)
%     
% %     TODO: Prepare the image (img) for vl_sift() function
%     img = imread(char(Filenames(i)));
%     [keypoints{i}, descriptors{i}] = vl_sift(single(rgb2gray(img)));
% %     Match features between SIFT model and SIFT features from new image
%     sift_matches{i} = vl_ubcmatch(descriptors{i}, model.descriptors, threshold_ubcmatch); 
% end


% Save sift features, descriptors and matches and load them when you rerun the code to save time
% save('sift_matches.mat', 'sift_matches');
% save('detection_keypoints.mat', 'keypoints')
% save('detection_descriptors.mat', 'descriptors')

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

ransac_iterations = 100; 
threshold_ransac = 4;

for i = 1:10
    fprintf('Running PnP+RANSAC for image: %d \n', i)
   
%     TODO: Implement the RANSAC algorithm here
    [C,ia] = unique(sift_matches{i}(2,:)); % Getting indices of unique matches
    unique_matches2d = sift_matches{i}(1,ia);
    unique_matches3d = sift_matches{i}(2,ia);
    flag=0;
    while(flag==0)  
        image_points = zeros(4,2);
        world_points = zeros(4,3);
        num_samples = 4;
        perm = randperm(size(unique_matches2d,2)) ;
        sel = perm(1:num_samples);
        sample_1 = zeros(4,1);
        sample_2 = zeros(4,1);
        for j = 1:num_samples
            sample_1(j) = unique_matches2d(1,sel(j));  % detection sample
            sample_2(j) = unique_matches3d(1,sel(j));  % model sample
        end
        
        for j = 1:num_samples
            image_points(j,1) = keypoints{i}(1,sample_1(j));
            image_points(j,2) = keypoints{i}(2,sample_1(j));
            world_points(j,:) = model.coord3d(sample_2(j),:);
        end
    
        max_reproj_err = 10000;
        try
            [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i)] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', max_reproj_err);
            flag = 1;
        catch ME
            if(strcmp(ME.identifier,'Could not find enough inliers in imagePoints and worldPoints.'))    
                continue;
            end
        end 
    end
    for iter = 1:50
        % Getting 2d-3d correspondences from R,T
        P = IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i) -cam_in_world_orientations(:,:,i)*cam_in_world_locations(:,:,i).'];
        matching_3d = model.coord3d(unique_matches3d,:); % matching 3d points
        model_image = zeros(size(matching_3d,1),2);   % 2D correspondences
        for k=1:size(matching_3d,1)
            xcoor_3d = matching_3d(k,:);
            xcoor_3d(k,4) = 1;    % homogeneous
            model_img = P*xcoor_3d(k,:)';
            model_img(1) = model_img(1)/model_img(3);
            model_img(2) = model_img(2)/model_img(3);
            model_image(k,1) = model_img(1);
            model_image(k,2) = model_img(2);
            xcoor_3d = [];
        end    
    
        % Ransac
        
        sift_matched_2d = keypoints{i}(1:2,unique_matches2d); % matching keypoints in 2D
        sift_matched_2d = sift_matched_2d';
        inlier_count = 0;
        inliers_set = [];
        for m = 1 : size(model_image,1)
            dist = norm(sift_matched_2d(m,:) - model_image(m,:));
            if(dist < threshold_ransac)
                inlier_count = inlier_count + 1;
                inliers_set = cat(1,inliers_set,model_image(m,:));
            end
        end
        num_inliers = inlier_count;
        if(num_inliers > 1)
            prev_inliers = num_inliers;
            best_orientation = cam_in_world_orientations(:,:,i);
            best_translation = cam_in_world_locations(:,:,i);
            inliers_set_indices = sample_1;
            model_world = zeros(size(inliers_set,1),3);
            for p=1:size(inliers_set,1)
                copy = inliers_set(p,:);
                copy(p,3:4) = 1;
                copy = copy';
                model_world(p,:) = inv(P)*copy;
            end    
            [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i)] = estimateWorldCameraPose(inliers_set, model_world, camera_params, 'MaxReprojectionError', max_reproj_err);
            
        elseif(num_inliers > prev_inliers)
            prev_inliers = num_inliers;
            best_orientation = cam_in_world_orientations(:,:,i);
            best_translation = cam_in_world_locations(:,:,i);
            inliers_set_indices = sample_1;
            model_world = zeros(size(inliers_set,1),3);
            for p=1:size(inliers_set,1)
                copy = inliers_set(p,:);
                copy(p,3:4) = 1;
                copy = copy';
                model_world(k,:) = inv(P)*copy;
            end    
            [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i)] = estimateWorldCameraPose(inliers_set, model_world, camera_params, 'MaxReprojectionError', max_reproj_err);
            
        end
    end


end



%% Visualize inliers and the bounding box

% You can use the visualizations below or create your own one
% But be sure to present the bounding boxes drawn on the image to verify
% the camera pose

edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];

for i=1:10
    
    figure()
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    
%   Plot inliers set
   % PlotInlierOutlier(best_inliers_set{i}, camera_params, sift_matches{i}, model.coord3d, keypoints{i}, cam_in_world_orientations(:,:,i), cam_in_world_locations(:,:,i))
%   Plot bounding box
    points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
    end
    hold off;
end