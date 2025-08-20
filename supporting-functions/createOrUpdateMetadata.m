function [success, message] = createOrUpdateMetadata(metadata_file, version_str, processing_params, varargin)
%CREATEORUPDATEMETADATA Create or update metadata file with parameter change detection
%   [SUCCESS, MESSAGE] = CREATEORUPDATEMETADATA(METADATA_FILE, VERSION_STR, PROCESSING_PARAMS)
%   creates or updates a metadata JSON file, automatically detecting parameter
%   changes and backing up previous versions.
%
%   [SUCCESS, MESSAGE] = CREATEORUPDATEMETADATA(METADATA_FILE, VERSION_STR, PROCESSING_PARAMS, VARARGIN)
%   allows additional name-value pairs for dataset information:
%       'ScriptName' - Name of the calling script (default: 'unknown_script.m')
%       'DetectionPath' - Path to detection input data (default: '')
%       'AISSource' - Source of AIS data (default: 'Unknown')
%       'AISFilename' - AIS data filename (default: '')
%       'LandMask' - Land mask filename (default: '')
%       'InfrastructureDataset' - Infrastructure dataset filename (default: '')
%
%   Inputs:
%       metadata_file - Full path to metadata JSON file
%       version_str - Version string (e.g., "v1.0.0")
%       processing_params - Struct containing processing parameters
%       varargin - Optional name-value pairs (see above)
%
%   Outputs:
%       success - Logical flag indicating successful operation
%       message - String message describing the operation result
%
%   Example:
%       [success, msg] = createOrUpdateMetadata('metadata.json', 'v1.0.0', params, ...
%           'ScriptName', 's_data_association_nv.m', ...
%           'AISSource', 'Spire', ...
%           'AISFilename', '202311.mat');
%
%   The function will:
%   - Compare current parameters with existing metadata
%   - Create timestamped backups when parameters change
%   - Log changes for reproducibility
%   - Handle corrupted or missing metadata gracefully

% Parse optional inputs
p = inputParser;
addParameter(p, 'ScriptName', 'unknown_script.m', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(p, 'DetectionPath', '', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(p, 'AISSource', 'Unknown', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(p, 'AISFilename', '', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(p, 'LandMask', '', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(p, 'InfrastructureDataset', '', @(x) ischar(x) || (isstring(x) && isscalar(x)));
parse(p, varargin{:});

% Initialize outputs
success = false;
message = '';

try
    % Create current metadata structure
    current_metadata = struct();

    % Basic information
    current_metadata.version = version_str;
    current_metadata.script_name = p.Results.ScriptName;
    current_metadata.last_updated = string(datetime('now'));

    % Environment information
    current_metadata.environment = struct();
    current_metadata.environment.matlab_version = version;
    current_metadata.environment.computer_architecture = computer('arch');
    current_metadata.environment.operating_system = computer;
    current_metadata.environment.username = getenv('USERNAME');

    % Processing parameters
    current_metadata.processing_parameters = processing_params;

    % Dataset information
    current_metadata.data = struct();
    current_metadata.data.detection_input_path = p.Results.DetectionPath;
    current_metadata.data.ais_source = p.Results.AISSource;
    current_metadata.data.ais_month = p.Results.AISFilename;
    current_metadata.data.land_mask = p.Results.LandMask;
    current_metadata.data.infrastructure_dataset = p.Results.InfrastructureDataset;

    should_create_new = true;

    % Check if metadata file exists
    if isfile(metadata_file)
        try
            % Load existing metadata
            existing_json = fileread(metadata_file);
            existing_metadata = jsondecode(existing_json);

            % Compare processing parameters
            if isfield(existing_metadata, 'processing_parameters')
                params_changed = ~isequal(existing_metadata.processing_parameters, processing_params);
                version_changed = ~isfield(existing_metadata, 'version') || ...
                    ~strcmp(existing_metadata.version, version_str);

                if params_changed || version_changed
                    % Parameters or version changed - backup old file
                    backup_file = strrep(metadata_file, '.json', ...
                        sprintf('_backup_%s.json', datestr(now, 'yyyymmdd_HHMMSS')));
                    copyfile(metadata_file, backup_file);
                    message = sprintf('Parameters changed. Backed up old metadata to: %s\n', backup_file);

                    % Add change log entry
                    current_metadata.change_log = struct();
                    current_metadata.change_log.previous_backup = backup_file;
                    current_metadata.change_log.changes_detected = string(datetime('now'));

                    if params_changed
                        current_metadata.change_log.parameter_changes = true;
                        message = [message 'Processing parameters have changed.\n'];
                    end
                    if version_changed
                        current_metadata.change_log.version_changes = true;
                        message = [message sprintf('Version changed from %s to %s.\n', ...
                            string(existing_metadata.version), version_str)];
                    end
                else
                    % No changes detected
                    should_create_new = false;
                    message = 'No changes detected in parameters or version. Using existing metadata.';
                end
            else
                % Old metadata format - update it
                message = 'Updating metadata file to include processing parameters.';
            end

        catch ME
            message = sprintf('Warning: Could not read existing metadata file. Creating new one.\nError: %s', ME.message);
        end
    else
        % File doesn't exist - this is the first run
        current_metadata.creation_date = string(datetime('now'));
        message = 'Creating new metadata file.';
    end

    % Write metadata file if needed
    if should_create_new
        fid = fopen(metadata_file, 'w');
        if fid == -1
            error('Could not open metadata file for writing: %s', metadata_file);
        end

        encoded_json = jsonencode(current_metadata, 'PrettyPrint', true);
        fprintf(fid, '%s', encoded_json);
        fclose(fid);

        message = [message sprintf('\nUpdated metadata file: %s', metadata_file)];
    end

    success = true;

catch ME
    success = false;
    message = sprintf('Failed to create/update metadata: %s', ME.message);

    % Close file handle if it exists
    if exist('fid', 'var') && fid ~= -1
        fclose(fid);
    end
end

end