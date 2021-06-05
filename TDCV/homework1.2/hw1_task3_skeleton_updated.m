clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = '../data/validation/img';
% path to object ply file
object_path = '../data/teabox.ply';
% path to results folder
results_path = '../data/validation/results';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Create directory for results
if ~exist(results_path,'dir') 
    mkdir(results_path); 
end

% Load Ground Truth camera poses for the validation sequence
% Camera orientations and locations in the world coordinate system
load('gt_valid.mat')

% TODO: setup camera parameters (camera_params) using cameraParameters()
IntrinsicMatrix = [2960.37845 0 0; 0 2960.37845 0; 1841.68855 1235.23369 1]; 
camera_params = cameraParameters('IntrinsicMatrix',IntrinsicMatrix);


%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);

% Place predicted camera orientations and locations in the world coordinate system for all images here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

%% Detect SIFT keypoints in all images

% You will need vl_sift() and vl_ubcmatch() functions
% download vlfeat (http://www.vlfeat.org/download.html) and unzip it somewhere
% Don't forget to add vlfeat folder to MATLAB path
run('../vlfeat-0.9.21/toolbox/vl_setup');
% Place SIFT keypoints and corresponding descriptors for all images here
keypoints = cell(num_files,1); 
descriptors = cell(num_files,1); 

% for i=1:length(Filenames)
%     fprintf('Calculating sift features for image: %d \n', i)
%     
% %    TODO: Prepare the image (img) for vl_sift() function
%     img = imread(char(Filenames(i)));
%     [keypoints{i}, descriptors{i}] = vl_sift(single(rgb2gray(img))) ;
% end

% Save sift features and descriptors and load them when you rerun the code to save time
% save('sift_descriptors.mat', 'descriptors')
% save('sift_keypoints.mat', 'keypoints')

 load('sift_descriptors.mat');
 load('sift_keypoints.mat');

%% Initialization: Compute camera pose for the first image

% As the initialization step for tracking
% you need to compute the camera pose for the first image 
% The first image and it's camera pose will be your initial frame 
% and initial camera pose for the tracking process

% You can use estimateWorldCameraPose() function or your own implementation
% of the PnP+RANSAC from previous tasks

% You can get correspondences for PnP+RANSAC either using your SIFT model from the previous tasks
% or by manually annotating corners (e.g. with mark_images() function)


% TODO: Estimate camera position for the first image


% num_points =8;
% labeled_points = mark_image('../data/validation/img/color_000006.JPG', num_points);
% save('labeled_points.mat', 'labeled_points')
load('labeled_points.mat')


image_points = [labeled_points(1:5, :) ; labeled_points(7:8, :)];
world_points = [vertices(1:5, :) ; vertices(7:8, :)];
threshold_ubcmatch = 1.5;

[init_orientation, init_location] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', 4 );

cam_in_world_orientations(:,:, 1) = init_orientation;
cam_in_world_locations(:,:, 1) = init_location;


%cam_in_world_orientations(:,:,1) = gt_valid.orientations(:,:,1);
%cam_in_world_locations(:,:,1) = gt_valid.locations(:,:,1);
fprintf('Completed calculating pose for the first image and storing it in cam_in_world_orientations and cam_in_world_locations\n')

% Visualise the pose for the initial frame
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
figure()
hold on;
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
title(sprintf('Initial Image Camera Pose'));
%   Plot bounding box
points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,1), cam_in_world_locations(:, :, 1));
for j=1:12
    plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
end
hold off;


%% IRLS nonlinear optimisation

% Now you need to implement the method of iteratively reweighted least squares (IRLS)
% to optimise reprojection error between consecutive image frames

% Method steps:
% 1) Back-project SIFT keypoints from the initial frame (image i) to the object using the
% initial camera pose and the 3D ray intersection code from the task 1. 
% This will give you 3D coordinates (in the world coordinate system) of the
% SIFT keypoints from the initial frame (image i) that correspond to the object
% 2) Find matches between descriptors of back-projected SIFT keypoints from the initial frame (image i) and the
% SIFT keypoints from the subsequent frame (image i+1) using vl_ubcmatch() from VLFeat library
% 3) Project back-projected SIFT keypoints onto the subsequent frame (image i+1) using 3D coordinates from the
% step 1 and the initial camera pose 
% 4) Compute the reprojection error between 2D points of SIFT
% matches for the subsequent frame (image i+1) and 2D points of projected matches
% from step 3
% 5) Implement IRLS: for each IRLS iteration compute Jacobian of the reprojection error with respect to the pose
% parameters and update the camera pose for the subsequent frame (image i+1)
% 6) Now the subsequent frame (image i+1) becomes the initial frame for the
% next subsequent frame (image i+2) and the method continues until camera poses for all
% images are estimated

num_samples = 33000;
threshold_irls = 0.005; % update threshold for IRLS
N = 20; % number of iterations
threshold_ubcmatch = 6; % matching threshold for vl_ubcmatch()
box_points = [0.165 0 0.093; 0 0 0.093; 0 0.063 0.093; 0.165 0.063 0.093; 0.165 0 0; 0 0 0; 0 0.063 0; 0.165 0.063 0];

for i=2:num_files
    model2.coord3d = [];
    model2.descriptors = [];
%     Randomly select a number of SIFT keypoints
    perm = randperm(size(keypoints{i-1},2)) ;
    sel = perm(1:num_samples);
    
%    Section to be deleted starts here
    P = IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i-1) -cam_in_world_orientations(:,:,i-1)*cam_in_world_locations(:,:,i-1).'];
    Q = P(:,1:3);
    q = P(:,4);
    orig = -inv(Q)*q; % this corresponds to C
%   
    for j=1:num_samples

        % Perform intersection between a ray and the object
        % You can use TriangleRayIntersection to find intersections
        % Pay attention at the visible faces from the given camera position
        m(1) = keypoints{i-1}(1,sel(j));
        m(2) = keypoints{i-1}(2,sel(j));
        m(3) = 1;
        dir = inv(Q)*m';
        vert1 = box_points(faces(:,1)+1,:);
        vert2 = box_points(faces(:,2)+1,:);
        vert3 = box_points(faces(:,3)+1,:);
        [intersect, t, u, v, xcoor] = TriangleRayIntersection(orig, dir, vert1, vert2, vert3);
        if sum(intersect)==0
            continue;
        end
        a = intersect.*t;
        a(a==0) = inf;
        [M,I]=min(a);
        model2.coord3d = cat(1,model2.coord3d,xcoor(I,:,:));
        if(isfinite(M))
        model2.descriptors = cat(2,model2.descriptors,descriptors{i-1}(:,sel(j)));
        end
    end
    
    fprintf('Matching sift features for image: %d \n', i)
    % %     Match features between previous frame SIFT model and SIFT features from new image
    sift_matches{i} = vl_ubcmatch(descriptors{i}, model2.descriptors, threshold_ubcmatch);
    
    % Getting 2d-3d correspondences from R,T
    world_points =  model2.coord3d(sift_matches{i}(2,:),:); %selecting 3D coordinates of i-1 to project them to image plane in i
    sift_matched_2d = keypoints{i}(1:2,sift_matches{i}(1,:)); % 2D points in i
    [cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i)] = IRLS_FiniteDiff(cam_in_world_orientations(:,:,i-1),cam_in_world_locations(:, :, i-1) , IntrinsicMatrix, world_points', sift_matched_2d,  N, threshold_irls);
   
end
% We suggest you to validate the correctness of the Jacobian implementation
% either using Symbolic toolbox or finite differences approach

% TODO: Implement IRLS method for the reprojection error optimisation
% You can start with these parameters to debug your solution 
% but you should also experiment with their different values


%% Plot camera trajectory in 3D world CS + cameras

figure()
% Predicted trajectory
visualise_trajectory(vertices, edges, cam_in_world_orientations, cam_in_world_locations, 'Color', 'b');
hold on;
% Ground Truth trajectory
visualise_trajectory(vertices, edges, gt_valid.orientations, gt_valid.locations, 'Color', 'g');
hold off;
title('\color{green}Ground Truth trajectory \color{blue}Predicted trajectory')
%% Visualize bounding boxes

figure()
for i=1:num_files
    
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    % Ground Truth Bounding Boxes
    points_gt = project3d2image(vertices',camera_params, gt_valid.orientations(:,:,i), gt_valid.locations(:, :, i));
    % Predicted Bounding Boxes
    points_pred = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points_gt(1, edges(:, j)), points_gt(2, edges(:,j)), 'color', 'g');
        plot(points_pred(1, edges(:, j)), points_pred(2, edges(:,j)), 'color', 'b');
    end
    hold off;
    
    filename = fullfile(results_path, strcat('image', num2str(i), '.png'));
    saveas(gcf, filename)
end

%% Bonus part

% Save estimated camera poses for the validation sequence using Vision TUM trajectory file
% format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Then estimate Absolute Trajectory Error (ATE) and Relative Pose Error for
% the validation sequence using python tools from: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
% In this task you should implement you own function to convert rotation matrix to quaternion

% Save estimated camera poses for the test sequence using Vision TUM 
% trajectory file format

% Attach the file with estimated camera poses for the test sequence to your code submission
% If your code and results are good you will get a bonus for this exercise
% We are expecting the mean absolute translational error (from ATE) to be
% approximately less than 1cm

% TODO: Estimate ATE and RPE for validation and test sequences

