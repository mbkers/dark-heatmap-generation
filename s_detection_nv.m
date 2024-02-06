% s_detection_nv.m
% This script performs ship detection on NovaSAR imagery using a 2-D CFAR
% detector with (optional) morphological opening.

% Note if morph. needed, CFAR 'OutputFormat','CUT result' else 'Detection index'

clear
clc

%% Select the image
% Specify the path depending on the operating system
im_path = "";
if ispc  % For Windows
    im_path = "Q:\NovaSAR\Airbus DS NovaSAR 10-10-2022\NovaSAR-Data-unzipped";
elseif isunix  % For Unix
    im_path = "/vol/research/SSC-SRS_Data/NovaSAR/Airbus DS NovaSAR 10-10-2022/NovaSAR-Data-unzipped";
else
    error("Specify the path to the SAR data.");
end

% List the folder contents
im_folders = dir(im_path);
im_folders(ismember({im_folders.name},{'.','..'})) = []; % remove . and ..

start_loop = tic;
% Select the image folder
for f = 180 : 188%length(im_folders)
    folder = im_folders(f).name;

    % List the folder contents
    im_items = dir(fullfile(im_path,folder));
    im_items(ismember({im_items.name},{'.','..'})) = []; % remove . and ..

    % Create a logical vector telling which is a directory
    dir_flag = [im_items.isdir];

    % Extract only non-empty directories
    subfolders = im_items(dir_flag);
    %subfolders = subfolders(~cellfun('isempty',{subfolders}));

    % Create a cell array containing the subfolder names
    subfolder_names = {subfolders.name};

    %% Loop through and process each image's subfolder
    start_sub_loop = tic;
    for s_f = 1 : length(subfolders) % parfor
        %% Read the image file and its metadata
        % Define the base path
        base_path = fullfile(im_path,folder,subfolder_names(s_f));

        % Define the image filenames
        im_filename_HH = fullfile(base_path,"image_HH.tif");
        im_filename_VV = fullfile(base_path,"image_VV.tif");

        % Check which image band (HH or VV) exists
        % (If both are found, prioritise HH over VV)
        im_file_loc = "";
        if exist(im_filename_HH,"file")
            im_file_loc = im_filename_HH;
        elseif exist(im_filename_VV,"file")
            im_file_loc = im_filename_VV;
        end

        % Read the image file if a valid file was found
        if isfile(im_file_loc)
            fprintf(1,'Now reading subfolder %d of %d for image folder %d\n',s_f,length(subfolders),f);
            try
                A = readgeoraster(im_file_loc,"Bands",1,"OutputType","double");
                %info_raster = georasterinfo(im_file_loc); %info = imfinfo(im_file_loc);
                %info_geotiff = geotiffinfo(im_file_loc);
            catch ME
                warning("Error in:\n%s\nError message: %s\nPassing to next iteration of loop.",im_file_loc,ME.message);
                %continue;
            end
        else
            warning("File does not exist:\n%s\nPassing to next iteration of loop.",im_file_loc);
            %continue;
        end

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

        % Create the geolocation grid and interpolate up to the full resolution
        % (SNAP: 10 x 12 --> 13748 x 13748 pixels, (deg))
        tic
        [X,Y] = meshgrid(1:size(A,2),1:size(A,1)); % x,y | c,r | w,h
        latq = griddata(x,y,lat,X,Y);
        lonq = griddata(x,y,lon,X,Y);
        time_grid = toc;

        % Save the geolocation grid
        % save(fullfile(base_path,"latq.mat"),"latq")
        % save(fullfile(base_path,"lonq.mat"),"lonq")

        %% Land masking
        % Create a referencing object, R, for masking (note: R is not used for
        % extracting detection centroid locations*)
        latlim = [min(lat) max(lat)];
        lonlim = [min(lon) max(lon)];
        R = georefcells(latlim,lonlim,[size(A,2) size(A,1)],'ColumnsStartFrom','south','RowsStartFrom','east');

        % *The reason for this is because this method only works for a regular
        % quadrangle and so is less accurate (i.e. geographically rectangular
        % and aligned with parallels and meridians)

        % Select and read land mask
        % Specify the path depending on the operating system
        mask_path = "";
        if ispc  % For Windows
            mask_path = "C:\Users\mkers\OneDrive - University of Surrey (1)\Projects\NEREUS\Processing\Study Area\QGIS\Processing\Land mask";
        elseif isunix  % For Unix
            mask_path = "/user/HS301/mr0052/Downloads/OneDrive_1_17-05-2023";
        else
            error("Specify the path to the land mask.");
        end
        mask_filename = "land_polygons_clip_reproject_m_buffer_250m_epsg4326.shp";
        mask_file_loc = fullfile(mask_path,mask_filename);
        land_poly = shaperead(mask_file_loc,'BoundingBox',...
            [R.LongitudeLimits(1) R.LatitudeLimits(1); R.LongitudeLimits(2) R.LatitudeLimits(2)]); % 'UseGeoCoords',true

        % Trim shapefile polygons to image lon/lat limits
        [ry,rx] = maptrimp([land_poly.Y],[land_poly.X],R.LatitudeLimits,R.LongitudeLimits);

        % Remove data within land mask
        [rx,ry] = removeExtraNanSeparators(rx,ry);
        rx = round(rx,9); % Note: If mask shows errors, increase N in round(X,N). For example, increase N from 5 (default) to 6 (or higher) for complicated coastlines.
        ry = round(ry,9);
        [latcells,loncells] = polysplit(ry,rx);
        tic
        for m = 1 : size(latcells)
            mask = inpolygon(lonq,latq,loncells{m,1},latcells{m,1});
            %[ix,iy] = geographicToIntrinsic(R,latcells{m,1},loncells{m,1});
            %mask = poly2mask(ix,iy,size(A,1),size(A,2)); % R.RasterSize(1),R.RasterSize(2)
            A(mask) = NaN;
        end
        time_masking = toc;

        % Show data
        % figure
        % worldmap([min(lat)-0.1 max(lat)+0.1],[min(lon)-0.1 max(lon)+0.1])
        % geoshow(A,R)
        % for m_fig = 1 : size(latcells)
        %     geoshow(latcells{m_fig,1},loncells{m_fig,1})
        %     hold on
        % end

        %% Radiometric calibration
        % Define the parameters
        calibration_constant = str2double(S.metadata.Imageu_Attributes.CalibrationConstant.Text);
        inc_angle_coeffs = str2num(S.metadata.Imageu_Generationu_Parameters.IncAngleCoeffs.Text);

        % Call the calibration function
        tic
        [~,~,sigma_nought,sigma_nought_dB,inc_angle] =...
            f_calibration(A,calibration_constant,inc_angle_coeffs);
        time_cal = toc;

        % Resize the incidence angle array to the size of the image
        inc_angle = repmat(inc_angle,[size(A,1) 1]);

        % Save incidence angle array
        % save(fullfile(base_path,"inc_angle.mat"),"inc_angle")

        % Convert Amplitude to Intensity
        % I = A.^2;

        % Convert to Decibels (dB)
        % I_dB = 10*log10(I);

        %% (Optional) Multilook (increase SCR)
        multilook = 0; % false
        if multilook == 1 % true
            ml_factor_az = 2;
            ml_factor_rg = 2;
            tic
            I = f_multilooking(I,ml_factor_az,ml_factor_rg);
            time_ml = toc;
        end

        %% (Optional) Block processing
        % bim = blockedImage(I);

        %% Detection
        % Define and setup 2-D CFAR detector
        detector = phased.CFARDetector2D(...
            "Method","CA",...
            "TrainingBandSize",[11 11],... % Background window
            "GuardBandSize",[8 8],...
            "ProbabilityFalseAlarm",1e-7,... % 1e-8 (RADARSAT-2)
            "OutputFormat","CUT result",... % "Detection index"
            "ThresholdOutputPort",true);

        % (Optional) Block processing
        %block1 = I(1:1000,1:end);

        % Pad the image
        padding_size = [detector.GuardBandSize(2) + detector.TrainingBandSize(2), ...
            detector.GuardBandSize(1) + detector.TrainingBandSize(1)];
        sigma_nought = padarray(sigma_nought,padding_size,"replicate","both"); % I

        % Define the Cells Under Test (CUT)
        N = size(sigma_nought); % I % block1
        N_gr = detector.GuardBandSize(1);
        N_gc = detector.GuardBandSize(2);
        N_tr = detector.TrainingBandSize(1);
        N_tc = detector.TrainingBandSize(2);

        col_start = N_tc + N_gc + 1;
        col_end = N(2) - (N_tc + N_gc);
        row_start = N_tr + N_gr + 1;
        row_end = N(1) - (N_tr + N_gr);

        n_rows = row_end - row_start + 1;
        n_cols = col_end - col_start + 1;
        CUT_idx = zeros(2,n_rows*n_cols);
        for n = 0 : n_rows-1
            CUT_idx(1,(n*n_cols + 1:(n+1)*n_cols)) = row_start + n;
            CUT_idx(2,(n*n_cols + 1:(n+1)*n_cols)) = col_start:col_end;
        end

        % Compute a detection result for each CUT
        tic
        [dets,th] = detector(sigma_nought,CUT_idx); % I % block1 % ~10-20 min
        time_detect = toc;

        % Create a binary image of the detection results
        I_bw = zeros([n_rows n_cols]);
        for j = 1 : numel(dets)
            I_bw(CUT_idx(1,j)-padding_size(1),CUT_idx(2,j)-padding_size(2)) = dets(j,:);
        end

        % (Optional) Morphological opening
        morph = 0; % false
        if morph == 1 % true
            se_radius = 2; % 1 pixel ~ 14 m
            se = strel('disk',se_radius);
            I_bw = imopen(I_bw,se);
        end

        % Show data
        %figure, imshow(I_bw);
        figure, imshow(sigma_nought_dB,[]); % I_dB
        hold on

        % Extract centroids and bounding boxes
        cc = bwconncomp(I_bw,8);
        stats = regionprops(cc,'Centroid','BoundingBox');
        centroids = cat(1,stats.Centroid);
        centroids = ceil(centroids);
        bounding_boxes = cat(1,stats.BoundingBox);
        bounding_boxes = ceil(bounding_boxes);

        % Adjust centroids and bounding boxes depending if multilooked
        if multilook == true
            centroids = ceil(ml_factor_az*centroids);
            bounding_boxes = ceil(ml_factor_az*bounding_boxes);
        end

        % Determine the length of the object
        if ~isempty(bounding_boxes)
            % Extract the maximum size of the object in pixels
            length_in_pixels = max(bounding_boxes(:,3),bounding_boxes(:,4));

            % Convert length to metres using the pixel spacing
            length_in_metres = length_in_pixels * str2double(S.metadata.Imageu_Attributes.SampledPixelSpacing.Text);
        end

        % Remove centroid duplicates
            % Compute centroid pairwise distance
            centroids_dist = pdist2(centroids,centroids);
            centroids_dist_t = 15;
            centroids_close = centroids_dist <= centroids_dist_t;
            centroids_close = centroids_close - eye(size(centroids_close));
    
            % Remove centroids closer than threshold
            [centroids_close_r,centroids_close_c] = find(centroids_close);
            c_idx = [centroids_close_r centroids_close_c];
            if ~isempty(c_idx)
                c_idx = sort(c_idx,2);
                c_idx = unique(c_idx,"rows");
                centroids(c_idx(:,2),:) = [];
                length_in_metres(c_idx(:,2),:) = [];
            end

        % Extract centroids
        if ~isempty(centroids)
            % Plot centroids
            plot(centroids(:,1),centroids(:,2),'r*')
            hold off

            % Convert centroids to latitude and longitude
            clat = zeros(size(centroids,1),1);
            clon = zeros(size(centroids,1),1);
            for c = 1 : size(centroids,1)
                clat(c) = latq(centroids(c,2),centroids(c,1)); % (y,x)
                clon(c) = lonq(centroids(c,2),centroids(c,1));
            end
            objects = [clat clon length_in_metres];

            % Export results
            str = strcat("objects","_morph",string(morph),"_v2.csv");
            out = fullfile(base_path,str);
            writematrix(objects,out)
        end

    end
    end_sub_loop_time = toc(start_sub_loop);

end
end_loop_time = toc(start_loop);

%% Supporting local functions
function [beta_nought,beta_nought_dB,sigma_nought,sigma_nought_dB,...
    IncAngle] = f_calibration(A,CalibrationConstant,IncAngleCoeffs)
%F_CALIBRATION Radiometrically calibrate SAR data
%   This function calculates Beta Nought and Sigma Nought from NovaSAR-1
%   data. An example is given at the end.
%
%   INPUTS:
%       A (matrix): [double]/uint16, amplitude DN values read from
%           'image_XX.tif' file in image product subdirectory.
%       CalibrationConstant (scalar): double, radiometric calibration
%           constant (found in 'metadata.xml' under 'Image Attributes').
%       IncAngleCoeffs (vector): double, polynomial coefficients for
%           incidence angle at pixel position (found in 'metadata.xml'
%           under 'Image Generation Parameters').
%
%   OUTPUTS:
%       beta_nought:
%       beta_nought_dB:
%       sigma_nought:
%       sigma_nought_dB:
%       IncAngle:

% Calculate Beta Nought (also called "radar brightness")
%A = single(A); % Convert A to single precision
beta_nought = A.^2 / CalibrationConstant;
beta_nought_dB = 10 * log10(beta_nought);
% beta_nought_dB(~isfinite(beta_nought_dB)) = 0;

% Calculate Incidence Angle at a given pixel in the image range line
IncAngle = zeros(1,size(A,2));

% If pixel 0 starts at near range:
% k = 1 : numel(IncAngleCoeffs);
% for p = 0 : (size(A,2)-1) % Poly: A_0 + A_1 x^1 + ... + A_n x^n
%     IncAngle(size(A,2)-p) = sum(IncAngleCoeffs(k) .* p.^(k-1));
% end

% If pixel 0 starts at far range:
k = 1 : numel(IncAngleCoeffs);
for p = 1 : size(A,2)
    IncAngle(p) = sum(IncAngleCoeffs(k) .* (p-1).^(k-1));
end

% Calculate Sigma Nought
sigma_nought = beta_nought .* sind(IncAngle);
sigma_nought_dB = beta_nought_dB + 10 * log10(sind(IncAngle));

end

% ----------
% Example:

% % 1) Read the image file.
% P = 'C:\Users\mkers\Downloads\NovaSAR_8602_191206\NovaSAR_01_8602_scd_191206_010236_HH\NovaSAR_01_8602_scd_30_191206_010236_HH_1';
% F = 'image_HH.tif';
% FileLoc = fullfile(P, F);
% A = imread(FileLoc);

% % 2) (Optional) Take a subset of the image for faster processing (i.e. split into 3).
% [n, m] = size(A);
% id = fix(m/3);
% A_1 = A(:, 1:id);
% A_2 = A(:, id+1:2*id);
% A_3 = A(:, 2*id+1:m);

% % 3) Call the function calibration_nv.m (< 40 s runtime).
% CalibrationConstant = 5.02E+05;
% IncAngleCoeffs = [34.946 0.00022595 -5.0746E-10 1.6041E-16 4.1128E-21 -1.1582E-26];
% tic
% [beta_nought,beta_nought_dB,sigma_nought,sigma_nought_dB] =...
%     calibration_nv(A_1,CalibrationConstant,IncAngleCoeffs);
% toc

% % 4) Display image(s).
% figure, imshow(beta_nought)
% figure, imshow(beta_nought_dB, [])
% figure, imshow(sigma_nought)
% figure, imshow(sigma_nought_dB, [])

% ----------
% References:
% 0342129_NovaSAR Products PDF
% SNAP Help: Calibration
% https://mdacorporation.com/geospatial/international/satellites/RADARSAT-2/docs/default-source/product-spec-sheets/geospatial-services/r1_prod_spec.pdf?sfvrsn=6
% https://www.intelligence-airbusds.com/en/9315-radiometric-calibration-of-terrasar-x-data
% https://www.intelligence-airbusds.com/files/pmedia/public/r465_9_tsxx-airbusds-tn-0049-radiometric_calculations_d1.pdf
% https://www.iceye.com/hubfs/Downloadables/ICEYE-Level-1-Product-Specs-2019.pdf
% https://earth.esa.int/web/sentinel/radiometric-calibration-of-level-1-products

%% References
% https://www.mathworks.com/help/phased/detection-and-system-analysis.html?s_tid=CRUX_lftnav
% https://www.mathworks.com/help/phased/examples/constant-false-alarm-rate-cfar-detection.html#responsive_offcanvas
% https://www.mathworks.com/help/phased/ref/phased.cfardetector2d-system-object.html
% https://www.mathworks.com/help/phased/ref/phased.cfardetector2d.step.html#bvevmw6-1-cutidx
% https://www.google.com/search?client=firefox-b-d&q=phased.CFARDetector2D+large+image+matlab
% https://www.mathworks.com/help/images/large-image-files.html?s_tid=CRUX_lftnav
% https://uk.mathworks.com/matlabcentral/answers/166629-is-there-any-way-to-list-all-folders-only-in-the-level-directly-below-a-selected-directory
% https://uk.mathworks.com/matlabcentral/answers/49414-check-if-a-file-exists
