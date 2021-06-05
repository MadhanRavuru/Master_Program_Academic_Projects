clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = '../data/init_texture';
% path to object ply file
object_path = '../data/teabox.ply';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Coordinate System is right handed and placed at lower left corner of tea
% box. Using coordinates of teabox model, vertex numbering is visualized in
% image vertices.png

imshow('vertices.png')
title('Vertices numbering')

%% Label images
% You can use this function to label corners of the model on all images
% This function will give an array with image coordinates for all points
% Be careful that some points may not be visible on the image and so this
% will result in NaN values in the output array
% Don't forget to filter NaNs later
num_points = 8;
% //labeled_points = mark_image(path_img_dir, num_points);


% Save labeled points and load them when you rerun the code to save time
% //save('labeled_points.mat', 'labeled_points')
load('labeled_points.mat')

%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);

%% Check corners labeling by plotting labels
for i=1:length(Filenames)
    figure()
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit')
    title(sprintf('Image: %d', i))
    hold on
    for point_idx = 1:8
        x = labeled_points(point_idx,1,i);
        y = labeled_points(point_idx,2,i); 
        if ~isnan(x)
            plot(x,y,'x', 'LineWidth', 3, 'MarkerSize', 15)
            text(x,y, char(num2str(point_idx)), 'FontSize',12)
        end
    end
end


%% Call estimateWorldCameraPose to perform PnP

% Place estimated camera orientation and location here to use
% visualisations later
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

%   NaN values in each image
nan_img = {[1:4, 7:8], [1:4, 6:8], [1:4, 6:7], [1:7], [1:6], [1:6, 8], [1:5, 8], [1:5, 7:8]};
    
world_points = [0.165 0 0.093; 0 0 0.093; 0 0.063 0.093; 0.165 0.063 0.093; 0.165 0 0; 0 0 0; 0 0.063 0; 0.165 0.063 0];

% iterate over the images
for i=1:num_files
    
    fprintf('Estimating pose for image: %d \n', i)

%   TODO: Estimate camera pose for every image
%     In order to estimate pose of the camera using the function bellow you need to:
%   - Prepare image_points and corresponding world_points
%   - Setup camera_params using cameraParameters() function
%   - Define max_reproj_err - take a look at the documentation and
%   experiment with different values of this parameter 
    image_points = labeled_points(:,:,i);
    %   Prepare image_points
    % FILTER NaN values in labeled_points
    prep_filter_points = labeled_points(:, :,i);
    filtered_points = prep_filter_points(isfinite(prep_filter_points(:, 1)), :);

    %   Prepare world_points
    filtered_world_points = world_points(nan_img{i}, :);

    IntrinsicMatrix = [2960.37845 0 0; 0 2960.37845 0; 1841.68855 1235.23369 1]; 
    camera_params = cameraParameters('IntrinsicMatrix',IntrinsicMatrix);
    max_reproj_err = 4;
    [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i)] = estimateWorldCameraPose(filtered_points, filtered_world_points, camera_params, 'MaxReprojectionError', max_reproj_err);
    
end

%% Visualize computed camera poses

% Edges of the object bounding box
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
visualise_cameras(vertices, edges, cam_in_world_orientations, cam_in_world_locations);

%% Detect SIFT keypoints in the images

% You will need vl_sift() and vl_plotframe() functions
% download vlfeat (http://www.vlfeat.org/download.html) and unzip it somewhere
% Don't forget to add vlfeat folder to MATLAB path

run('vlfeat/toolbox/vl_setup');
% Place SIFT keypoints and corresponding descriptors for all images here
keypoints = cell(num_files,1); 
descriptors = cell(num_files,1); 

for i=1:length(Filenames)
    fprintf('Calculating sift features for image: %d \n', i)

%    TODO: Prepare the image (img) for vl_sift() function
    img = imread(char(Filenames(i)));
    [keypoints{i}, descriptors{i}] = vl_sift(single(rgb2gray(img))) ;
end

% When you rerun the code, you can load sift features and descriptors to

% Save sift features and descriptors and load them when you rerun the code to save time
save('sift_descriptors.mat', 'descriptors')
save('sift_keypoints.mat', 'keypoints')
 load('sift_descriptors.mat');
 load('sift_keypoints.mat');


% Visualisation of sift features for the first image
figure()
hold on;
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
vl_plotframe(keypoints{1}(:,:), 'linewidth',2);
title('SIFT features')
hold off;



%% Build SIFT model
% Filter SIFT features that correspond to the features of the object

% Project a 3D ray from camera center through a SIFT keypoint
% Compute where it intersects the object in the 3D space
% You can use TriangleRayIntersection() function here

% Your SIFT model should only consists of SIFT keypoints that correspond to
% SIFT keypoints of the object
% Don't forget to visualise the SIFT model with the respect to the cameras
% positions


% num_samples - number of SIFT points that is randomly sampled for every image
% Leave the value of 1000 to retain reasonable computational time for debugging
% In order to contruct the final SIFT model that will be used later, consider
% increasing this value to get more SIFT points in your model
num_samples=14500;
size_total_sift_points=num_samples*num_files;

% Visualise cameras and model SIFT keypoints
fig = visualise_cameras(vertices, edges, cam_in_world_orientations, cam_in_world_locations);
hold on

% Place model's SIFT keypoints coordinates and descriptors here
model.coord3d = [];
model.descriptors = [];


for i=1:num_files
    
%     Randomly select a number of SIFT keypoints
    perm = randperm(size(keypoints{i},2)) ;
    sel = perm(1:num_samples);
    
%    Section to be deleted starts here
    P = IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i) -cam_in_world_orientations(:,:,i)*cam_in_world_locations(:,:,i).'];
    Q = P(:,1:3);
    q = P(:,4);
    orig = -inv(Q)*q; % this corresponds to C
%   
    for j=1:num_samples
        
    % TODO: Perform intersection between a ray and the object
    % You can use TriangleRayIntersection to find intersections
    % Pay attention at the visible faces from the given camera position
    
    % Getting u, v coordinate for the sample 
    m(1) = keypoints{i}(1,sel(j));   
    m(2) = keypoints{i}(2,sel(j));
    m(3) = 1;                        % Homogeneous
    dir = inv(Q)*m';
    vert1 = world_points(faces(:,1)+1,:);
    vert2 = world_points(faces(:,2)+1,:);
    vert3 = world_points(faces(:,3)+1,:);
    [intersect, t, u, v, xcoor] = TriangleRayIntersection(orig, dir, vert1, vert2, vert3);
    if sum(intersect)==0       % No intersection
        continue;
    end
    a = intersect.*t;
    a(a==0) = inf;
    [M,I]=min(a);          % M and I store respectively the minimum distance and its index (so that we only look at front face and not back)
    model.coord3d = cat(1,model.coord3d,xcoor(I,:,:));
    if(isfinite(M))
    model.descriptors = cat(2,model.descriptors,descriptors{i}(:,sel(j)));
    end
    end  
        
end

hold off
xlabel('x');
ylabel('y');
zlabel('z');

% Save your sift model for the future tasks
save('sift_model.mat', 'model');

%% Visualise only the SIFT model
figure()
scatter3(model.coord3d(:,1), model.coord3d(:,2), model.coord3d(:,3), 'o', 'b');
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');

