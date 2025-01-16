% s_data_association_nv.m
% This script loops through NovaSAR image folders and associates AIS
% data to SAR object detections. The loop can process data for up to one
% month after loading a month's worth of AIS data, before there is a need
% to update the AIS data.

clear
clc

%% Load datasets
% Initialise tables for spreadsheet tracker
excel_table = [];
ais_assign_all = [];
ais_unassign_all = [];
ais_beacons_all = [];

% Reference ellipsoid
wgs84 = wgs84Ellipsoid('km');

% AIS data (Spire)
    % Specify the base path depending on the operating system
    if ispc  % For Windows
        ais_path = "C:\Users\mkers\Desktop";
    elseif isunix  % For Unix
        ais_path = "/vol/research/SSC-SRS_Data/eE-Surrey-NEREUS (2023) AIS updated with GFW/mat";
    else
        error("Specify the base path to the AIS data.");
    end

    % Load the .mat file data into a struct
    ais_filename = "202311.mat";
    ais_file_loc = fullfile(ais_path,ais_filename); % replace ais_file_loc with ais_path ?
    if isfile(ais_file_loc)
        try
            ais_struct = load(ais_file_loc);
        catch ME
            warning("Failed to load AIS .mat file: %s\nError message: %s",ais_file_loc,ME.message);
        end
    else
        warning("File does not exist: %s",ais_file_loc);
    end

    % Rename the table
    ais_original = ais_struct.ais;
    clear ais_struct

    % Calculate the time window
    default_window = 60; % min
    time_window = f_calculateTimeWindow(ais_original,default_window);

% Land mask
    % Specify the base path depending on the operating system
    if ispc  % For Windows
        mask_path = "C:\Users\mkers\OneDrive - University of Surrey (1)\Projects\NEREUS\Processing\Study Area\QGIS\Processing\Land mask";
    elseif isunix  % For Unix
        mask_path = "/user/HS301/mr0052/Downloads/OneDrive_1_17-05-2023";
    else
        error("Specify the base path to the land mask.");
    end
    mask_filename = "land_polygons_clip_reproject_m_buffer_250m_epsg4326.shp";
    mask_file_loc = fullfile(mask_path,mask_filename);

% Offshore infrastructure
    % Read the offshore infrastructure data
    infrastructure_path = "C:\Users\mkers\OneDrive - University of Surrey (1)\Resources\GFW\Datasets";
    infrastructure_filename = "offshore_infrastructure_v20231106.csv";
    infrastructure_loc = fullfile(infrastructure_path,infrastructure_filename);
    infrastructure = readtable(infrastructure_loc);
    
    % Convert infrastructure geodetic coordinates to Cartesian coordinates
    [infrastructure_x,infrastructure_y,~] = geodetic2ecef(wgs84,...
        infrastructure.lat,infrastructure.lon,0);

% Trained classification model
% load()

%% Load SAR data
% Specify the base path depending on the operating system
if ispc  % For Windows
    im_path = "Q:\NovaSAR\Mauritius 2022-2024\NovaSAR-Data-unzipped";
elseif isunix  % For Unix
    im_path = "/vol/research/SSC-SRS_Data/NovaSAR/Mauritius 2022-2024/NovaSAR-Data-unzipped";
else
    error("Specify the base path to the SAR data.");
end

% List the folder contents
im_folders = dir(im_path);
im_folders(ismember({im_folders.name},{'.','..'})) = []; % remove . and ..

start_loop = tic;
% Select the image folder
for f = 181 : 188%length(im_folders)
    folder = im_folders(f).name;

    % List the folder contents
    im_items = dir(fullfile(im_path,folder));
    im_items(ismember({im_items.name},{'.','..'})) = []; % remove . and ..

    % Create a logical vector telling which is a directory
    dir_flag = [im_items.isdir];

    % Extract only non-empty directories
    subfolders = im_items(dir_flag);

    % Create a cell array containing the subfolder names
    subfolder_names = {subfolders.name};

    %% Loop through and process each image's subfolder
    for s_f = 1 : length(subfolders)
        %% Read the QL image file, the image metadata and the detections
        % Define the base path
        base_path = fullfile(im_path,folder,subfolder_names(s_f));

        % Define the QL image filenames
        % im_filename_HH = fullfile(base_path,"QL_image_HH.tif");
        % im_filename_VV = fullfile(base_path,"QL_image_VV.tif");

        % Check which QL image band (HH or VV) exists
        % (If both are found, prioritise HH over VV)
        % if exist(im_filename_HH,"file")
        %     im_file_loc = im_filename_HH;
        % elseif exist(im_filename_VV,"file")
        %     im_file_loc = im_filename_VV;
        % end

        % Read the QL image file if a valid file was found
        % if isfile(im_file_loc)
        %     fprintf(1,'Now reading subfolder %d of %d for image folder %d\n',s_f,length(subfolders),f);
        %     try
        %         QL_image = readgeoraster(im_file_loc,"Bands",1,"OutputType","double");
        %     catch ME
        %         warning("Error in:\n%s\nError message: %s\nPassing to next iteration of loop.",im_file_loc,ME.message);
        %         continue;
        %     end
        % else
        %     warning("File does not exist:\n%s\nPassing to next iteration of loop.",im_file_loc);
        %     continue;
        % end

        % Read the image metadata
        metadata_filename = "metadata.xml";
        metadata_file_loc = fullfile(base_path,metadata_filename);
        if isfile(metadata_file_loc)
            try
                S = xml2struct(metadata_file_loc);
            catch ME
                warning("Failed to parse XML file:\n%s\nError message: %s\nAssigning an empty array.",metadata_file_loc,ME.message);
                S = []; % Return an empty structure
            end
        else
            warning("File does not exist:\n%s\nAssigning an empty array.",metadata_file_loc);
            S = []; % Return an empty structure
        end

        % Read the detections file
        dets_filename = "objects_morph0_v1.csv"; % old: objects_multilook0_morph0_v1.csv
        dets_file_loc = fullfile(base_path,dets_filename);
        if isfile(dets_file_loc)
            try
                sar = readmatrix(dets_file_loc);
                sar = array2table(sar,"VariableNames",["lat" "lon" "length"]);
            catch ME
                warning("Error in:\n%s\nError message: %s\nAssigning an empty array.",dets_file_loc,ME.message);
                sar = []; % Return an empty array
            end
        else
            warning("File does not exist:\n%s\nAssigning an empty array.",dets_file_loc);
            sar = []; % Return an empty array
        end

        %% Pre-processing: Extract data from metadata
        % Get the latitude, longitude, line and pixel tie-point grid data
        n_tie_points = length(S.metadata.geographicInformation.TiePoint);
        lat = zeros(n_tie_points,1);
        lon = zeros(n_tie_points,1);
        row = zeros(n_tie_points,1);
        col = zeros(n_tie_points,1);
        for ii = 1 : n_tie_points
            lat(ii,1) = str2double(S.metadata.geographicInformation.TiePoint{1,ii}.Latitude.Text);
            lon(ii,1) = str2double(S.metadata.geographicInformation.TiePoint{1,ii}.Longitude.Text);
            row(ii,1) = str2double(S.metadata.geographicInformation.TiePoint{1,ii}.Line.Text);
            col(ii,1) = str2double(S.metadata.geographicInformation.TiePoint{1,ii}.Pixel.Text);
        end
        x = col + 1; % MATLAB starts at (1,1) instead of (0,0)
        y = row + 1;

        % Create the SAR footprint/bounding box
        b = boundary(lon,lat);
        bbox_lat = lat(b);
        bbox_lon = lon(b);
        bbox_lat = flipud(bbox_lat);
        bbox_lon = flipud(bbox_lon); % https://uk.mathworks.com/matlabcentral/answers/456451-why-does-geoshow-not-color-the-correct-part-of-the-map
        kmlwritepolygon(fullfile(base_path,"\bbox.kml"),bbox_lat,bbox_lon);

        % Convert the SAR footprint/bounding box to Cartesian coordinates
        [bbox_x,bbox_y,~] = geodetic2ecef(wgs84,bbox_lat,bbox_lon,0);

        % Get the SAR datetime
        sar_datetime = datetime(S.metadata.Sourceu_Attributes.RawDataStartTime.Text);

        % Load the geolocation grid and the incidence angle
        % load(fullfile(base_path,"latq.mat"))
        % load(fullfile(base_path,"lonq.mat"))
        % load(fullfile(base_path,"inc_angle.mat"))

        %% SAR data processing
        % Discrimination
        % Remove detections within 500 m of infrastructure
        infra = infrastructure;
        infra_x = infrastructure_x;
        infra_y = infrastructure_y;
        infra_dist_threshold = 500; % metres

        % Filter infrastructure to SAR bounding box to speed up cost matrix
        [bbox_in,bbox_on] = inpolygon(infra_x,infra_y,bbox_x,bbox_y);
        bbox_out_idx = find(~bbox_in);
        infra(bbox_out_idx,:) = [];

        if ~isempty(sar)
            % Prepare a cost matrix
            sar_infra_dist = f_2DCostMatrixFormation([sar.lat sar.lon],...
                [infra.lat infra.lon],'geodesic');

            % Remove detections within 500 m of infrastructure
            sar_infra_close = sar_infra_dist <= infra_dist_threshold;
            [sar_infra_close_r,~] = find(sar_infra_close);
            sar(sar_infra_close_r,:) = [];

            % Merge duplicate detections
            threshold_distance = 127.50; % Merging threshold distance in metres
            sar = f_mergeDetections(sar,threshold_distance,@f_2DCostMatrixFormation);

            % Remove detections with a length of one pixel or less
            min_target_size = str2double(S.metadata.Imageu_Attributes.SampledPixelSpacing.Text); % metres
            sar(sar.length <= min_target_size,:) = [];

            % Remove detections with a length of X pixels or greater
            max_target_size = 600; % metres
            sar(sar.length >= max_target_size,:) = [];
        end

        % Classify detections using pretrained model
        % TBD

        %% AIS data processing
        ais = ais_original;

        % Temporal filtering: SAR date and time
        ais = f_temporalFilter(ais,sar_datetime,time_window);

        % Spatial filtering: Remove AIS data outside a guard footprint
            % Add buffer to bounding box polygon (guard footprint)
            buff_width = 0.2500; % deg
            [bbox_lat_b,bbox_lon_b] = bufferm(bbox_lat,bbox_lon,buff_width,'outPlusInterior');

            % Convert guard footprint to Cartesian coordinates
            [bbox_x_b,bbox_y_b,~] = geodetic2ecef(wgs84,bbox_lat_b,bbox_lon_b,0);

            % Convert AIS data to Cartesian coordinates
            [ais.x,ais.y,~] = geodetic2ecef(wgs84,ais.lat,ais.lon,0);

            % Test if AIS data is inside guard footprint
            [guard_in,guard_on] = inpolygon(ais.x,ais.y,bbox_x_b,bbox_y_b);
            guard_out_idx = find(~guard_in);
            ais(guard_out_idx,:) = [];

        % Spatio-temporal alignment: Interpolation and azimuth-shift compensation
            % Interpolate AIS data to the SAR image datetime and add the
            % implied speed and bearing
            ais = f_interpData(ais,sar_datetime); % TODO: replace with bilstm model

            % (Reverse) Azimuth image shift compensation
            % state_vectors = S.metadata.OrbitData.StateVector;
            % ais = f_azimuthShift(ais,latq,lonq,inc_angle,state_vectors);

        % Spatial filtering: SAR footprint, land mask and infrastructure
            % Remove AIS data outside SAR footprint
            % Test if AIS data is inside SAR footprint
            [foot_in,foot_on] = inpolygon(ais.x,ais.y,bbox_x,bbox_y);
            foot_out_idx = find(~foot_in);
            ais(foot_out_idx,:) = [];

            % Remove AIS data inside land mask
                % Define latitude and longitude limits
                latlim = [min(bbox_lat)-0.1 max(bbox_lat)+0.1];
                lonlim = [min(bbox_lon)-0.1 max(bbox_lon)+0.1];

                % Read land mask shapefile
                mask = shaperead(mask_file_loc,"BoundingBox",...
                    [lonlim(1) latlim(1); lonlim(2) latlim(2)]);

                % Convert land mask to Cartesian coordinates
                [mask_x,mask_y,~] = geodetic2ecef(wgs84,[mask.Y],[mask.X],0);

                % Test if AIS data is inside land mask
                if ~isempty(mask)
                    [mask_in,mask_on] = inpolygon(ais.x,ais.y,mask_x,mask_y); % ais.lon,ais.lat,[mask.X],[mask.Y]
                    mask_in_idx = find(mask_in);
                    ais(mask_in_idx,:) = [];
                end

            % Remove AIS data within 500 m of infrastructure
            ais_infra_dist = f_2DCostMatrixFormation([ais.lat ais.lon],...
                [infra.lat infra.lon],'geodesic');
            ais_infra_close = ais_infra_dist <= infra_dist_threshold;
            [ais_infra_close_r,~] = find(ais_infra_close);
            ais(ais_infra_close_r,:) = [];

        % Identify and exclude AIS beacons
            % Check if an entry is a beacon
            ais.is_beacon = false(height(ais),1);
            for i = 1 : height(ais)
                ais.is_beacon(i) = f_checkIfBeacon(ais(i,:));
            end
    
            % Separate the data into beacons and vessels
            ais_beacons = ais(ais.is_beacon,:);
            ais = ais(~ais.is_beacon,:);

        % Data resolver
            % Update the data with standard missing values and return the
            % MMSI and IMO of entries missing length, width and vessel_type
            [ais, missing_data_ids] = f_findMissingDataIds(ais);
    
            % Look up the MMSI or IMO in a public vessel database
            db_vals = f_databaseLookup(missing_data_ids,"mmsi");
    
            % Fill the missing data from the public database
            ais = f_updateMissingData(ais,missing_data_ids,db_vals);

        %% Show data
        figure('Position',[100 100 1120 840])
        worldmap([min(lat)-0.1 max(lat)+0.1],[min(lon)-0.1 max(lon)+0.1])

        % QL_image_R = georefcells([min(lat) max(lat)],[min(lon) max(lon)],size(QL_image));
        % geoshow(QL_image,colormap("gray"),QL_image_R,"DisplayType","image")

        geoshow(bbox_lat,bbox_lon,'DisplayType','polygon','FaceColor','y','FaceAlpha',.3)

        if ~isempty(mask)
            geoshow(mask.Y,mask.X,'Color',[0.4660 0.6740 0.1880])
        end

        if ~isempty(ais)
            geoshow(ais.lat,ais.lon,'DisplayType','point','MarkerEdgeColor',[0.6350 0.0780 0.1840],'Marker','x')
        end

        if ~isempty(sar)
            geoshow(sar.lat,sar.lon,'DisplayType','point','MarkerEdgeColor',[0 0.4470 0.7410],'Marker','x')
        end

        if ~isempty(ais_beacons)
            geoshow(ais_beacons.lat,ais_beacons.lon,'DisplayType','point','MarkerEdgeColor',[0.4660 0.6740 0.1880],'Marker','x')
        end

        %% Data association
        if ~isempty(ais) & ~isempty(sar)
            % Prepare a cost matrix for the SAR and AIS data
            cost_matrix = f_2DCostMatrixFormation([ais.lat ais.lon],...
                [sar.lat sar.lon],'geodesic',wgs84); % km

            % Determine the cost of unassignment/gating parameter
            cost_of_unassign = ( convvel(15,'kts','m/s') * (time_window/2*60) ) ...
                / 1000; % km

            % Perform the assignment using the k-best algorithm
            m = 1;
            %start_assign = tic;
            [assignk,~,~] = assignkbest(cost_matrix,cost_of_unassign,m,'jv');
            %end_assign_time = toc(start_assign);
            assign = cat(1,assignk{:});
            assign = unique(assign,'rows');

            % Determine the number of competing assignments
            n_competing_assign = size(assign,1) - size(assignk{1,1},1);

            % Determine the assignments' confidence levels
            % TBD

            % Resolve the duplicate/competing assignments
            % TBD
        else
            assign = [];
        end

        % Obtain the final assignments
        if ~isempty(assign)
            ais_assign = ais(assign(:,1),:);
            sar_assign = sar(assign(:,2),:);
        else
            ais_assign = [];
            sar_assign = [];
        end

        % Obtain the final unassignments
        if ~isempty(assign)
            ais_unassign_idx = setdiff(1:size(ais,1),assign(:,1));
            ais_unassign = ais(ais_unassign_idx,:);
            sar_unassign_idx = setdiff(1:size(sar,1),assign(:,2));
            sar_unassign = sar(sar_unassign_idx,:);
        else
            ais_unassign = ais;
            sar_unassign = sar;
        end

        %% Export the data to csv
        if ~isempty(ais_assign)
            ais_assign.folder = repmat(f,[size(ais_assign,1) 1]);
            ais_assign.subfolder = repmat(s_f,[size(ais_assign,1) 1]);
            writetable(ais_assign,fullfile(base_path,"/ais_assign.csv")) % based on AIS_D
        end

        if ~isempty(sar_assign)
            writetable(sar_assign,fullfile(base_path,"/sar_assign.csv"))
        end

        if ~isempty(ais_unassign)
            ais_unassign.folder = repmat(f,[size(ais_unassign,1) 1]);
            ais_unassign.subfolder = repmat(s_f,[size(ais_unassign,1) 1]);
            writetable(ais_unassign,fullfile(base_path,"/ais_unassign.csv"))
        end

        if ~isempty(sar_unassign)
            writetable(sar_unassign,fullfile(base_path,"/sar_unassign.csv"))
        end

        if ~isempty(ais_beacons)
            ais_beacons.folder = repmat(f,[size(ais_beacons,1) 1]);
            ais_beacons.subfolder = repmat(s_f,[size(ais_beacons,1) 1]);
            writetable(ais_beacons,fullfile(base_path,"/ais_beacons.csv"))
        end

        %% Table for 'NovaSAR_processing_status.xlsx'
        if ~isempty(ais_assign)
            ais_assign_all = [ais_assign_all; ais_assign];
        end

        if ~isempty(ais_unassign)
            ais_unassign_all = [ais_unassign_all; ais_unassign];
        end

        if ~isempty(ais_beacons)
            ais_beacons_all = [ais_beacons_all; ais_beacons];
        end

        excel_table = [excel_table; f s_f size(ais_assign,1) size(sar_assign,1) ...
            size(ais_unassign,1) size(sar_unassign,1) size(ais_beacons,1)];

    end

end
end_loop_time = toc(start_loop);

%% Supporting local functions
function time_window = f_calculateTimeWindow(ais,default_window)
%F_CALCULATETIMEWINDOW Calculate time window for F_TEMPORALFILTER
%   This function takes in AIS data, ais (assumed to be a table with
%   'datetime' and 'mmsi' columns), and a default time window in minutes,
%   default_window (datetime).
%
%   It calculates the median reporting interval for each vessel and then
%   determines the overall mean of these intervals across all vessels to
%   define a time window for filtering the AIS data relative to the SAR
%   image acquisition time. The time window is adjusted by a factor
%   (e.g. 1.5) to ensure it's sufficiently broad to capture relevant data,
%   but not so wide as to unnecessarily increase the computational load.
%   The max function is used to ensure that the time window is at least as
%   large as a default minimum value (e.g. the default_window parameter).

% Identify unique vessels
unique_vessels = unique(ais.mmsi);

% Initialise array to hold median interval for each vessel
vessel_median_interval = NaN(length(unique_vessels),1);

for i = 1 : length(unique_vessels)
    vessel_id = unique_vessels(i);

    % Select data for the current vessel
    vessel_data = ais(ais.mmsi == vessel_id,:);

    % Ensure data is sorted by time for the current vessel
    vessel_data = sortrows(vessel_data,'datetime');

    % Calculate reporting intervals (in minutes) for the current vessel
    if height(vessel_data) > 1 % Ensure there are at least two data points to calculate an interval
        time_diffs = minutes(diff(vessel_data.datetime));
        vessel_median_interval(i) = median(time_diffs,'omitnan');
    end
end

% Calculate the overall [mean]/median of median intervals and use the
% overall interval to define the time window, adjusted by some factor
if ~isempty(vessel_median_interval)
    overall_interval = mean(vessel_median_interval,'omitnan');
    time_window = max(overall_interval * 1.5, default_window);
else
    time_window = default_window;
end

end


function retained_detections = f_mergeDetections(detections,threshold_distance,f_2DCostMatrixFormation)
%F_MERGEDETECTIONS Merges close detections based on a custom cost matrix
% and retains the one with larger length.
%
% Inputs:
%   detections - A table with columns ['lat', 'lon', 'length'],
%                where 'lat' and 'lon' represent the geographical coordinates
%                of the detections and 'length' represents the maximum size
%                of the bounding box.
%   threshold_distance - The threshold distance within which detections are
%                considered for merging.
%   f_2DCostMatrixFormation - A function handle to the custom function for
%                calculating the cost matrix.
%
% Output:
%   retained_detections - The detections after merging, in the same table
%                format as the input.

% Use the custom function to calculate the cost (distance) matrix
D = f_2DCostMatrixFormation([detections.lat detections.lon],...
    [detections.lat detections.lon],'geodesic');

% Identify detections to merge based on the distance threshold
to_merge = D <= threshold_distance & D > 0; % Exclude self-comparison

% Initialise all detections as kept
is_kept = true(height(detections),1);

for i = 1 : height(detections)
    if ~is_kept(i)
        continue; % If detection i is already not kept, skip it
    end
    for j = i+1:height(detections)
        if to_merge(i,j)
            % Use the 'length' to determine which detection to retain
            if detections.length(i) >= detections.length(j)
                is_kept(j) = false; % Retain detection i, discard j
            else
                is_kept(i) = false; % Retain detection j, discard i
                break; % Since i is not kept, no need to compare it further with others
            end
        end
    end
end

% Filter detections to keep only those marked as kept
retained_detections = detections(is_kept,:);

% Alt: Compute centroid pairwise distance
% centroids_dist = pdist2(centroids,centroids);
% centroids_dist_t = 15;
% centroids_close = centroids_dist <= centroids_dist_t;
% centroids_close = centroids_close - eye(size(centroids_close));

% Remove centroids closer than threshold
% [centroids_close_r,centroids_close_c] = find(centroids_close);
% c_idx = [centroids_close_r centroids_close_c];
% if ~isempty(c_idx)
%     c_idx = sort(c_idx,2);
%     c_idx = unique(c_idx,"rows");
%     centroids(c_idx(:,2),:) = [];
%     length_in_metres(c_idx(:,2),:) = [];
% end

end


function ais_filtered = f_temporalFilter(ais,sar_datetime,time_window)
%F_TEMPORALFILTER Filter AIS data to SAR image acquisition time
%   This function takes in AIS data, ais (assumed to be a table with a
%   'datetime' column), the SAR image acquisition time, sar_datetime
%   (datetime) and the pre-calculated time window, time_window.

% Filter AIS data based on the pre-calculated time window
start_time = sar_datetime - minutes(time_window/2);
end_time = sar_datetime + minutes(time_window/2);
ais_filtered = ais(ais.datetime >= start_time & ais.datetime <= end_time, :);

% Alt:
% ais = table2timetable(ais); % Convert table to timetable
% time_interval = 60; % min
% t1 = sar_datetime - minutes(time_interval/2);
% t2 = sar_datetime + minutes(time_interval/2);
% tr = timerange(t1,t2,'closed');
% ais = ais(tr,:);
% ais = timetable2table(ais);

end


function ais_out = f_interpData(ais_inp,interp_time)
%F_INTERPDATA Interpolate AIS data to the SAR image datetime and add the
% implied speed and bearing.
%
%   AIS_OUT = F_INTERPDATA(AIS_INP,INTERP_TIME)
%       Inputs: AIS_INP (AIS input table), INTERP_TIME (SAR image
%           datetime (i.e. the interpolation point))
%       Output: AIS_OUT (AIS output table with size unique(ais_in.mmsi))
%
%   Example: ais = f_interpData(ais,sar_datetime);
%
%   TODO:
%       - see 'Create a table of the preserved columns with mode values'
%       - does ais_inp need to be sorted for interp1 ?

if isempty(ais_inp)
    ais_inp.speed_implied = NaN(size(ais_inp,1),1);
    ais_inp.bearing_implied = NaN(size(ais_inp,1),1);
    ais_out = ais_inp;
    return;
end

% Ensure that ais_inp has no duplicates (needed for interp1)
[~,u_idx] = unique(ais_inp(:,{'datetime','mmsi'}));
ais_inp = ais_inp(u_idx,:);

% Sort the rows of the table (needed for deriving the implied speed)
ais_inp = sortrows(ais_inp,'mmsi','ascend');

% Repeat the interp_time vector for each row in ais_inp
interp_time = repmat(interp_time,size(ais_inp,1),1);

% Convert times to a numerical value
%t_num = datenum(ais_inp.datetime);
%interp_time_num = datenum(interp_time);

% Define the groups and values to be interpolated
groups = findgroups(ais_inp.mmsi); % split data by MMSI
%values = [t_num ais_inp.lat ais_inp.lon interp_time_num];

% Derive the implied speed based on the datetime, latitude and longitude
speed_implied = splitapply(@fh_speedImplied,ais_inp.datetime,ais_inp.lat,...
    ais_inp.lon,ais_inp.sog,groups);

% Derive the implied bearing based on the latitude and longitude
bearing_implied = splitapply(@fh_bearingImplied,ais_inp.lat,ais_inp.lon,...
    ais_inp.cog,groups);

% Convert the cell arrays to numeric arrays
speed_implied = cell2mat(speed_implied);
bearing_implied = cell2mat(bearing_implied);

% Add the implied data to the table as a new column (alt: use 'join' by 'keys')
ais_inp.speed_implied = speed_implied;
ais_inp.bearing_implied = bearing_implied;

% Define the other columns to preserve
interp_var_names = {'datetime','lat','lon','sog','cog','heading','x','y',...
    'speed_implied','bearing_implied'};
interp_non_var_names = setdiff(ais_inp.Properties.VariableNames,interp_var_names);

% Interpolate values for each group at the specified time
interp_vals = splitapply(@fh_interpData,ais_inp.datetime,ais_inp.lat,ais_inp.lon,...
    ais_inp.sog,ais_inp.cog,ais_inp.heading,ais_inp.x,ais_inp.y,...
    ais_inp.speed_implied,ais_inp.bearing_implied,interp_time,groups); % @f_interp_func,values,groups

% Extract the first row of each cell of the cell array
interp_vals_firstRows = cellfun(@(x) x(1,:),interp_vals,'UniformOutput',false);

% Convert cell array to matrix
interp_vals_firstRows = cell2mat(interp_vals_firstRows);

% Convert times back to datetime format
interp_vals_firstRows_time = datetime(interp_vals_firstRows(:,1),'ConvertFrom','datenum');

% Create a table with matrix and datetimes
interp_table = table(interp_vals_firstRows_time,...
    interp_vals_firstRows(:,2),interp_vals_firstRows(:,3),...
    interp_vals_firstRows(:,4),interp_vals_firstRows(:,5),...
    interp_vals_firstRows(:,6),interp_vals_firstRows(:,7),...
    interp_vals_firstRows(:,8),interp_vals_firstRows(:,9),...
    interp_vals_firstRows(:,10),'VariableNames',interp_var_names);

% Create a table of the preserved columns with mode values
%preserve_table = splitapply(@mode,ais_inp{:,'accuracy'},groups); % do for all interp_non_var_names cols
%ais_out = [interp_table preserve_table];
%ais_out = ais_out(:,ais_inp.Properties.VariableNames);

% Concatenate the interpolated first rows with the preserved columns
[~,u_idx_2] = unique(ais_inp.mmsi);
ais_out = [ais_inp(u_idx_2,interp_non_var_names) interp_table];
ais_out = ais_out(:,ais_inp.Properties.VariableNames);

end


function speed_implied = fh_speedImplied(t,lat,lon,sog)
%speedImplied Calculate the implied speed (m/s) from lat and lon.
%
%   SPEED_IMPLIED = SPEEDIMPLIED(T,LAT,LON,SOG)
%       Inputs: T [datetime], LAT [double], LON [double] and SOG [double]
%       Output: SPEED_IMPLIED [double]
%
%   Example: ais.speed_implied = speedImplied(ais.datetime,ais.lat,ais.lon,ais.sog);

if length(t) < 2 % Check if group has fewer than two points
    speed_implied = {sog / 1.9438444924574}; % kts to m/s
else
    % Convert the datetime (t) to a duration in seconds
    t = seconds(t - t(1));

    % Calculate the time differences between consecutive rows in seconds
    t_diff = diff(t);

    % Calculate the distances between consecutive rows in metres
    dist = distance(lat(1:end-1),lon(1:end-1),lat(2:end),lon(2:end),wgs84Ellipsoid);

    % Include the first entry from the SOG column*
    sog_1 = sog(1) / 1.9438444924574; % knots to metres per second

    % Calculate the speeds by dividing the distances by the time differences
    speed_implied = {[sog_1; dist ./ t_diff]};

    % Convert the cell array to a numeric array
    %speed_implied = cell2mat(speed_implied);
end

end

%*needed for the case when interp_time is between the first and second
% point and interpolation is carried out


function bearing_implied = fh_bearingImplied(lat,lon,cog)
%bearingImplied Calculate the implied bearing (deg) from lat and lon.
%
%   BEARING_IMPLIED = BEARINGIMPLIED(LAT,LON,COG)
%       Inputs: LAT [double], LON [double] and COG [double]
%       Output: BEARING_IMPLIED measured clockwise from True North [double]
%
%   Example: ais.bearing_implied = bearingImplied(ais.lat,ais.lon,ais.cog(1));

n_points = length(lat);
bearing_implied = zeros(1, n_points - 1);

for i = 1 : (n_points - 1)
    lat1 = lat(i);
    lon1 = lon(i);
    lat2 = lat(i + 1);
    lon2 = lon(i + 1);

    % Convert latitude and longitude from degrees to radians
    lat1 = deg2rad(lat1);
    lon1 = deg2rad(lon1);
    lat2 = deg2rad(lat2);
    lon2 = deg2rad(lon2);

    % Calculate the bearing using the Haversine formula
    dLon = lon2 - lon1;
    y = sin(dLon) * cos(lat2);
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon);

    % Convert the bearing from radians to degrees
    bearing = atan2(y,x);
    bearing = rad2deg(bearing);

    % Normalise the bearing to be in the range [0, 360] degrees
    bearing = mod(bearing + 360,360);

    % Store the bearing in the results array
    bearing_implied(i) = bearing;
end

% Include the first entry from the COG column
bearing_implied = {[cog(1) bearing_implied]'};

end


function interp_vals = fh_interpData(t,lat,lon,sog,cog,hdg,x,y,...
    speed_implied,bearing_implied,interp_time)
%FH_INTERPDATA Summary of this function goes here
%
%   X
%       Inputs:
%       Output:
%
%   Example:

if length(t) < 2 % Check if group has fewer than two points
    interp_vals = {[datenum(t) lat lon sog cog hdg x y speed_implied bearing_implied]};
else
    interp_lat = interp1(datenum(t),lat,datenum(interp_time),'spline');
    interp_lon = interp1(datenum(t),lon,datenum(interp_time),'spline');
    interp_sog = interp1(datenum(t),sog,datenum(interp_time),'makima',mean(sog)); % 'makima'*
    interp_cog = interp1(datenum(t),cog,datenum(interp_time),'nearest');
    interp_hdg = interp1(datenum(t),hdg,datenum(interp_time),'nearest');
    interp_x = interp1(datenum(t),x,datenum(interp_time),'spline');
    interp_y = interp1(datenum(t),y,datenum(interp_time),'spline');
    interp_speed_implied = interp1(datenum(t),speed_implied,datenum(interp_time),'makima',mean(speed_implied)); % 'makima'*
    interp_bearing_implied = interp1(datenum(t),bearing_implied,datenum(interp_time),'nearest'); % TODO: implement circular interpolation
    interp_vals = {[datenum(interp_time) interp_lat interp_lon interp_sog ...
        interp_cog interp_hdg interp_x interp_y interp_speed_implied interp_bearing_implied]};
end

end

%"If you specify the 'pchip', 'spline', or 'makima' interpolation
% methods, then the default behavior is 'extrap'. All other interpolation
% methods return NaN by default for query points outside the domain".

%*sometimes results in negative values (invalid) if 'extrap' is not
% specified as a scalar, in this case mean()

% if t < interp_time
% makima, extrap = nearest to first point
% else
% makima, extrap = nearest to last point


function ais_updated = f_azimuthShift(ais,latq,lonq,inc_angle,state_vectors)
%F_AZIMUTHSHIFT Apply azimuth shift to AIS positions
%   This function applies an azimuth shift correction to positions in an
%   AIS data table based on satellite state vectors and incidence angle
%   information.

% Extract the incidence angle at AIS data latitude and longitude
ais = f_extractIncidenceAngle(ais,latq,lonq,inc_angle);

% Calculate the average satellite height and speed from the state vectors
[average_sat_height,average_sat_speed] = f_calculateSatelliteMetrics(state_vectors);

% Determine platform heading (azimuth) relative to North
platform_heading_north = 97.86 - 90 + 180; % [0,359 deg clockwise from North]

% Determine vessel's implied bearing relative to satellite range direction (phi)
ais.phi = platform_heading_north - ais.bearing_implied + 90 + 180; % [0,359 deg anti-clockwise from Range]

% Calculate the azimuth shift in metres
ais.azimuth_shift = ( average_sat_height .* ais.speed_implied .* tand(ais.inc_angle) .* cosd(ais.phi) ) ./ average_sat_speed;

% Apply azimuth shift to AIS position in (opposite/reverse) azimuth
ais.azimuth_shift_dir = ais.bearing_implied - 90;
[ais.lat_shift,ais.lon_shift] = reckon(ais.lat,ais.lon,abs(ais.azimuth_shift),...
    ais.azimuth_shift_dir,wgs84Ellipsoid);

% Return the updated ais table
ais_updated = ais;

end


function ais_updated = f_extractIncidenceAngle(ais,latq,lonq,inc_angle)
%F_EXTRACTINCIDENCEANGLE Extract incidence angle for given latitude and longitude
%   This function outputs the 'ais' table with an added column for the
%   incidence angles. This function assumes that each latitude and
%   longitude pair in 'ais' corresponds to one location in the 'latq' and
%   'lonq' matrices. The function finds the closest point in 'latq' and
%   'lonq' matrices based on the sum of absolute differences, which works
%   well for regularly spaced grids.
%
%   INPUTS:
%       'ais' should be a table with at least two columns: 'lat' and 'lon',
%           representing the coordinates for which you want to find
%           incidence angles
%       'latq' and 'lonq' should be matrices of latitude and longitude
%           values for each pixel location
%       'inc_angle' should be a matrix of the incidence angles, with the
%           same dimensions as 'latq' and 'lonq'
%
%   Example: ais_updated = f_extractIncidenceAngle(ais,latq,lonq,inc_angle);

% Initialise a column for incidence angles in the table
ais.inc_angle = NaN(height(ais),1);

% Loop through each row in the ais table
for idx = 1 : height(ais)
    % Current lat and lon from ais table
    current_lat = ais.lat(idx);
    current_lon = ais.lon(idx);

    % Compute absolute differences
    lat_diffs = abs(latq - current_lat);
    lon_diffs = abs(lonq - current_lon);

    % Combine differences to find the closest point
    total_diffs = lat_diffs + lon_diffs;

    % Find the index of the minimum difference
    [~,min_idx] = min(total_diffs(:));

    % Convert index to subscript
    [row,col] = ind2sub(size(latq),min_idx);

    % Extract the corresponding incidence angle
    ais.inc_angle(idx) = inc_angle(row,col);
end

% Return the updated ais table
ais_updated = ais;

end


function [average_height, average_speed] = f_calculateSatelliteMetrics(state_vectors)
%F_CALCULATESATELLITEMETRICS Calculate average satellite height (in metres)
% and speed (in metres per second) from orbit state vectors
%   Detailed explanation goes here

% Earth's radius in meters
%earthRadius = 6371000; % Average radius in metres

% Initialise variables to accumulate height and speed
total_height = 0;
total_speed = 0;
n = length(state_vectors); % Number of state vectors

% Loop through each state vector
for i = 1 : n
    % Extract the current state vector
    state = state_vectors{i};

    % Calculate the height above Earth's surface
    height = sqrt(str2double(state.xPosition.Text)^2 +...
        str2double(state.yPosition.Text)^2 + str2double(state.zPosition.Text)^2)...
        - earthRadius;
    total_height = total_height + height;

    % Calculate the speed (magnitude of the velocity)
    speed = sqrt(str2double(state.xVelocity.Text)^2 +...
        str2double(state.yVelocity.Text)^2 + str2double(state.zVelocity.Text)^2);
    total_speed = total_speed + speed;
end

% Calculate the average height and speed
average_height = total_height / n;
average_speed = total_speed / n;

end


function is_beacon = f_checkIfBeacon(entry)
%F_CHECKIFBEACON Classify data into either real vessels or AIS beacons
% based on defined patterns.
%
%   IS_BEACON = F_CHECKIFBEACON(ENTRY)
%       Input: ENTRY [single row table]
%       Output: IS_BEACON [logical]
%
%   Example: ais.is_beacon(i) = f_checkIfBeacon(ais(i,:)); (Note that the
%   operations (regexp, cellfun) are not inherently vectorized for table
%   operations.)

% Define patterns and specific values for selected data fields
vessel_name_patterns = {'\dV', '\d{9}\s+\dV', '%', '\d+%', '(?i)BUOY', '(?i)NET', '(?i)LONGLINE', '(?i)TEST'};
callsign_patterns = {'\d+%'};

% Check vessel name
is_beacon = any(cellfun(@(pattern) ~isempty(regexp(entry.vessel_name, pattern, 'once')), vessel_name_patterns));

% Check callsign if not already identified as beacon
entry.callsign = string(entry.callsign); % Temporarily convert to a string
if ~is_beacon
    is_beacon = any(cellfun(@(pattern) ~isempty(regexp(entry.callsign, pattern, 'once')), callsign_patterns));
end

end


function [ais_updated, missing_data_ids] = f_findMissingDataIds(ais)
%F_FINDMISSINGDATAIDS Update the data with standard missing values and
% return the 'mmsi' and 'imo' of missing values in 'length', 'width'
% and 'vessel_type'.
%
%   [AIS_UPDATED, MISSING_DATA_IDS] = F_FINDMISSINGDATAIDS(AIS)
%       Input: AIS [table]
%       Outputs: AIS_UPDATED [table], MISSING_DATA_IDS [table]
%
%   Example: [ais, missing_data_ids] = f_findMissingDataIds(ais);

% Insert standard missing values
ais.length = standardizeMissing(ais.length,0);
ais.width = standardizeMissing(ais.width,0);
ais.vessel_type = standardizeMissing(ais.vessel_type,"Unknown");
ais.vessel_type = standardizeMissing(ais.vessel_type,"Not Available");
ais.vessel_type = standardizeMissing(ais.vessel_type,"UNAVAILABLE");

% Find indices of missing data in 'length', 'width' and 'vessel_type'
missing_length_idx = find(ismissing(ais.length));
missing_width_idx = find(ismissing(ais.width));
missing_vessel_type_idx = find(ismissing(ais.vessel_type));

% Combine indices
missing_indices = unique([missing_length_idx; missing_width_idx; missing_vessel_type_idx]);

% Retrieve corresponding 'mmsi' and 'imo' data
missing_data_ids = ais(missing_indices,{'mmsi','imo'});

% Return the updated table
ais_updated = ais;

end


function db_vals = f_databaseLookup(missing_data_ids,mmsi_or_imo)
%F_DATABASELOOKUP Look up the ID (choose either MMSI or IMO) in a public
% vessel database and return the dimensions and class information.
%
%   DB_VALS = F_DATABASELOOKUP(MISSING_DATA_IDS,MMSI_OR_IMO)
%       Inputs: MISSING_DATA_IDS [table], MMSI_OR_IMO [string]
%       Output: DB_VALS [table]
%
%   Example: db_vals = f_databaseLookup(missing_data_ids,"mmsi");

% Validate the 'mmsi_or_imo' input argument
valid_ids = ["imo", "mmsi"];
if ~ismember(mmsi_or_imo,valid_ids)
    error("Invalid ID type specified. Choose either ''mmsi'' or ''imo''.");
end

% Prepare the output table
db_vals = table('Size',[height(missing_data_ids) 3],...
    'VariableTypes',{'double','double','categorical'},...
    'VariableNames',{'length','width','vessel_type'});

% Read the HTML code from a URL using the webread function
for k = 1 : height(missing_data_ids)
    % Define the URL of the public vessel database
    url = "https://www.vesseltracker.com/en/vessels.html?term=";

    % Select the ID based on 'mmsi_or_imo'
    id = string(missing_data_ids{k,mmsi_or_imo});
    url = strcat(url,id);

    count = 0;
    err_count = 0;
    while count == err_count % when true try again (until error 403 resolves)
        try
            options = weboptions('Timeout',10);
            web_data = webread(url,options);
        catch ME
            fprintf('WEBREAD without success: %s\n',ME.message);
            err_count = err_count + 1;
        end
        count = count + 1;
    end

    % Parse the HTML code using htmlTree
    tree = htmlTree(web_data);

    % Find the relevant table in the HTML tree using findElement (the CSS
    % Selector can be found using 'Inspect Element' in Firefox)
    % selector = '.odd';
    % subtrees = findElement(tree,selector);

    % Extract the text from the subtrees using extractHTMLText
    % str = extractHTMLText(subtrees);

    % Extract substring "sizes"
    selector_size = '.odd > div:nth-child(7) > span:nth-child(1)';
    subtrees_size = findElement(tree,selector_size);
    try
        str_size = extractHTMLText(subtrees_size);
        if isempty(str_size)
            s = strcat("No information available for ID:"," ",id," ","(",mmsi_or_imo,").");
            disp(s)
            continue;
        end
    catch ME
        fprintf('Error: %s\n',ME.message);
        continue;
    end
    str_size = split(str_size,' ');
    length = str2double(str_size(1)); % error appears here not in try for no info
    beam = str2double(str_size(3));

    % Extract substring "type"
    selector_type = '.type';
    subtrees_type = findElement(tree,selector_type);
    str_type = extractHTMLText(subtrees_type);
    vessel_class = categorical(str_type(1));

    % Insert extracted information into output table
    db_vals{k,1} = length;
    db_vals{k,2} = beam;
    db_vals{k,3} = vessel_class;
end

end


function ais_updated = f_updateMissingData(ais,missing_data_ids,db_vals)
%F_UPDATEMISSINGDATA Update the original data with new values for where
% the data was initially missing.
%
%   AIS_UPDATED = F_UPDATEMISSINGDATA(AIS,MISSING_DATA_IDS,DB_VALS)
%       Inputs: AIS [table], MISSING_DATA_IDS [table], DB_VALS [table]
%       Output: AIS_UPDATED [table]
%
%   Example: ais = f_updateMissingData(ais,missing_data_ids,db_vals)

% Iterate through each missing data ID
for i = 1 : height(missing_data_ids)
    % Get the current ID
    current_id = missing_data_ids.mmsi(i);

    % Find the index in the original data
    idx = find(ais.mmsi == current_id);

    % Update 'length' if it is missing
    if ismissing(ais.length(idx))
        ais.length(idx) = db_vals.length(i);
    end

    % Update 'width' if it is missing
    if ismissing(ais.width(idx))
        ais.width(idx) = db_vals.width(i);
    end

    % Update 'vessel_type' if it is missing
    if ismissing(ais.vessel_type(idx))
        ais.vessel_type(idx) = db_vals.vessel_type(i);
    end
end

% Return the updated table
ais_updated = ais;
end


function cost_matrix = f_2DCostMatrixFormation(X,Y,dist,ellips,ellips2cart)
%F_2DCOSTMATRIXFORMATION Compute the 'cost' of each X coordinate matching
% a Y coordinate. The cost here is defined as the distance between the X
% and Y coordinate which is based on a selected association metric.
%
% INPUTS:
    % X           - m-by-n matrix
    % Y           - m-by-n matrix
    % dist        - distance or association metric (see below)
    % ellips      - (optional) provide a reference ellipsoid
    % ellips2cart - convert geodetic coordinates to ECEF Cartesian
    %               coordinates ('true' or 'false')
%
% OUTPUTS:
    % cost_matrix - size(X,1)-by-size(Y,1) matrix
%
% ASSOCIATION METRICS:
    % 'geodesic'       - Geodesic distance
    % 'euclidean'      - Euclidean distance
    % 'seuclidean'     - Standardized Euclidean distance
    % 'mazzarella2015' - Weighted features (0.7 pos, 0.2 hdg and 0.1 size)
    % 'custom'         - Custom distance (shown to give best results)
    % 'iou'            - IOU distance
    % 'ssim'           - Structural Similarity Index (SSIM) (xiu2019)
%
% FEATURES ([feat], [dataType]):
    % position (lat,lon), numeric
    % length, numeric
    % width, numeric
    % sog, numeric
    % cog/hdg, numeric
        % vachon2007report use cog
        % "True heading (optional)" - itu-r m.1371-5
        % 180 deg ambiguity - unreliable ?
    % class, categorical
    % under way/stationary ? (utilise azimuth shift), categorical
%
% Example: cost_matrix = f_2DCostMatrixFormation([ais.lat ais.lon],[sar.lat sar.lon],'geodesic',wgs84);

if(nargin<5)||isempty(ellips2cart)
    ellips2cart = false;
end

if(nargin<4||isempty(ellips))
    ellips = wgs84Ellipsoid; % wgs84Ellipsoid('km')
end

% Call lla2ecef/ellips2Cart/geodetic2ecef function % [X,Y,Z] = geodetic2ecef(spheroid,lat,lon,h)
if ellips2cart == true
    cartX = lla2ecef([X(:,1) X(:,2) zeros(size(X,1),1)]);
    cartY = lla2ecef([Y(:,1) Y(:,2) zeros(size(Y,1),1)]);
    X(:,1:2) = cartX(:,1:2);
    Y(:,1:2) = cartY(:,1:2);
end

% Preallocate a cost matrix
cost_matrix = zeros(size(X,1),size(Y,1));

% Association metrics
switch dist
    case 'geodesic'
        for i = 1:size(X,1)
            cost_matrix(i,:) = distance('gc',Y(:,1:2),X(i,1:2),ellips);
        end

    case 'euclidean'
        for i = 1:size(X,1)
            cost_matrix(i,:) = pdist2(Y(:,1:2),X(i,1:2),'euclidean');
        end

    case 'seuclidean'
        for i = 1:size(X,1)
            cost_matrix(i,:) = pdist2(Y(:,1:2),X(i,1:2),'seuclidean',std(Y,'omitnan'));
        end

    case 'mazzarella2015'
        for i = 1:size(X,1)
            cost_matrix(i,:) = 0.7 * ( distance('gc',Y(:,1:2),X(i,1:2),ellips) ) +...
                0.2 * ( abs(X(i,6) - Y(i,6)) ) +...
                0.1 * ( abs(X(i,3) - Y(i,3)) );
        end

    case 'custom' % Standardised using Y std values (default) % WIDTH DOMINANT
        for i = 1:size(X,1)
            cost_matrix(i,:) = pdist2(Y(:,1:2),X(i,1:2),'seuclidean',std(Y(:,1:2),'omitnan')) +...
                pdist2(Y(:,3),X(i,3),'seuclidean',std(Y(:,3),'omitnan')) +...
                pdist2(Y(:,4),X(i,4),'seuclidean',std(Y(:,4),'omitnan'));
        end

    case 'custom_mahal' % Mahalanobis distance
        cov12 = cov(Y(:,1:2),X(:,1:2),'omitrows');
        cov3 = cov(Y(:,3),X(:,3),'omitrows');
        cov4 = cov(Y(:,4),X(:,4),'omitrows');
        for i = 1:size(X,1)
            cost_matrix(i,:) = pdist2(Y(:,1:2),X(i,1:2),'mahalanobis') +...
                pdist2(Y(:,3),X(i,3),'mahalanobis',cov3(2)) +...
                pdist2(Y(:,4),X(i,4),'mahalanobis',cov4(2));
        end

    case 'custom_class'
        for i = 1:size(X,1)
            cost_matrix(i,:) = pdist2(Y(:,1:2),X(i,1:2),'seuclidean',std(Y(:,1:2),'omitnan')) +...
                pdist2(Y(:,3),X(i,3),'seuclidean',std(Y(:,3),'omitnan')) +...
                pdist2(Y(:,4),X(i,4),'seuclidean',std(Y(:,4),'omitnan')) +...
                min(abs(Y(:,7) - X(i,7)),0.1);
        end

    case 'custom_norm'
        a = zeros(size(X,1),size(Y,1));
        b = zeros(size(X,1),size(Y,1));
        c = zeros(size(X,1),size(Y,1));
        [X(:,1:4),X_center,X_scale] = normalize(X(:,1:4),'range');
        Y(:,1:4) = normalize(Y(:,1:4),'center',X_center,'scale',X_scale);
        for i = 1:size(X,1)
            a(i,:) = pdist2(Y(:,1:2),X(i,1:2));%,'seuclidean',std(Y(:,1:2),'omitnan'));
            b(i,:) = pdist2(Y(:,3),X(i,3));%,'seuclidean',std(Y(:,3),'omitnan'));
            c(i,:) = pdist2(Y(:,4),X(i,4));%,'seuclidean',std(Y(:,4),'omitnan'));
        end
        cost_matrix = 0.8*a + 0.15*b + 0.05*c;

    case 'custom_norm2' % same as 'custom_norm'
        for i = 1:size(X,1)
            cost_matrix(i,:) = 0.8*pdist2(Y(:,1:2),X(i,1:2),'seuclidean',std(Y(:,1:2),'omitnan')) +...
                0.1*pdist2(Y(:,3),X(i,3),'seuclidean',std(Y(:,3),'omitnan')) +...
                0.1*pdist2(Y(:,4),X(i,4),'seuclidean',std(Y(:,4),'omitnan'));
        end

    case 'iou'
        % Create M-by-5 matrices where M is the number of bounding boxes
        bb_width = zeros(size(X,1),1) + km2deg(11.1195);
        bb_height = zeros(size(X,1),1) + km2deg(11.1195);
        bb_X = [X(:,2) X(:,1) bb_width bb_height zeros(size(X,1),1)];
        bb_Y = [Y(:,2) Y(:,1) bb_width bb_height zeros(size(Y,1),1)];
        cost_matrix = 1 - bboxOverlapRatio(bb_X,bb_Y,"Union"); % rows = ais & cols = sar

    otherwise

end

end

% RESOURCES:
% http://www.econ.upf.edu/~michael/stanford/
% IOU distance cost matrix: https://arxiv.org/pdf/2105.07901.pdf

% NOTES:
% Non-standardised or non-normalised distance metrics will be dominated by
% features with the greatest units or scale (e.g. 0.05 deg vs 50 km)
