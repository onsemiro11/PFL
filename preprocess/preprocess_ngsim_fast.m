%% Vectorized reimplementation of PiP/preprocess/preprocess_ngsim.m
%  Same semantics & output format, but O(N) instead of O(N^2).
%  Env vars:
%    NGSIM_RAW      folder holding the 6 raw txt files (default ./raw_ngsim/)
%    NGSIM_OUT      output root                        (default ./dataset/ngsim/)
%    NGSIM_MAX_ROWS keep only first N rows per dataset (smoke test; default all)
%    NGSIM_SAVE_RAW '1' to also dump the pre-checker mat files (default '0')
clear; clc;

RAW_DIR  = getenv('NGSIM_RAW');   if isempty(RAW_DIR),  RAW_DIR  = './raw_ngsim/';    end
OUT_ROOT = getenv('NGSIM_OUT');   if isempty(OUT_ROOT), OUT_ROOT = './dataset/ngsim/';end
MAX_ROWS = str2double(getenv('NGSIM_MAX_ROWS'));
SAVE_RAW = strcmp(getenv('NGSIM_SAVE_RAW'), '1');

lane_filter = false;
grid_length = 25; grid_width = 5; cell_length = 8; cell_width = 7;

if lane_filter
    raw_folder     = fullfile(OUT_ROOT, sprintf('%dx%d_raw',      grid_length, grid_width));
    post_folder    = fullfile(OUT_ROOT, sprintf('%dx%d',          grid_length, grid_width));
    fix_tar_folder = fullfile(OUT_ROOT, sprintf('%dx%d_8Veh',     grid_length, grid_width));
else
    raw_folder     = fullfile(OUT_ROOT, sprintf('%dx%d_nofL_raw', grid_length, grid_width));
    post_folder    = fullfile(OUT_ROOT, sprintf('%dx%d_nofL',     grid_length, grid_width));
    fix_tar_folder = fullfile(OUT_ROOT, sprintf('%dx%d_8Veh_nofL',grid_length, grid_width));
end
for f = {raw_folder, post_folder, fix_tar_folder}, if ~exist(f{1},'dir'), mkdir(f{1}); end, end

grid_cells          = grid_length * grid_width;
grid_cent_location  = ceil(grid_cells*0.5);
dataset_to_use      = 6;

files = { 'us101-0750am-0805am.txt', 'us101-0805am-0820am.txt', 'us101-0820am-0835am.txt', ...
          'i80-0400-0415.txt',       'i80-0500-0515.txt',       'i80-0515-0530.txt' };

%% 1. Load ------------------------------------------------------------------
disp('Loading data...')
traj = cell(1, dataset_to_use);
for k = 1:dataset_to_use
    fp = fullfile(RAW_DIR, files{k});
    D  = readmatrix(fp, 'FileType', 'text');
    if ~isnan(MAX_ROWS) && size(D,1) > MAX_ROWS, D = D(1:MAX_ROWS, :); end
    fprintf('  dataset %d: %s  (%d rows)\n', k, files{k}, size(D,1));
    % Columns of the raw NGSIM txt (1-based):
    %  1 VehId 2 FrameId 3 TotalFrames 4 GlobalTime 5 LocalX 6 LocalY 7 GlobalX
    %  8 GlobalY 9 Length 10 Width 11 Class 12 Vel 13 Acc 14 LaneId ...
    % After prepending dataset id every index shifts by one (PiP convention).
    D = single([k*ones(size(D,1),1,'like',D), D]);
    D = D(:, [1,2,3,6,7,15,10,11,12,13,14]);   % ds veh frame x y lane len wid class vel acc
    if lane_filter
        fprintf('    #data: %d ==>> ', size(D,1));
        D = D(D(:,6) < 7, :);
        fprintf('%d after filtering lane>6\n', size(D,1));
    elseif k <= 3
        D(D(:,6) >= 6, 6) = 6;                 % US101: merge lanes >=6 into 6
    end
    % leave room for 2 maneuver labels + grid
    traj{k} = [D(:,1:6), zeros(size(D,1),2,'single'), D(:,7:11), zeros(size(D,1),grid_cells,'single')];
end
clear D

% Use the vehicle's center as its location, keep Y positive
for k = 1:dataset_to_use
    traj{k}(:,5) = traj{k}(:,5) - 0.5*traj{k}(:,9);
    off = min(traj{k}(:,5));
    if off < 0, traj{k}(:,5) = traj{k}(:,5) - off; end
end

%% 2. Parse maneuver labels and neighbour grid ------------------------------
disp('Parsing fields...')
halfL = 0.5*grid_length*cell_length;   % half grid length  (feet)
halfW = 0.5*grid_width *cell_width;    % half grid width   (feet)

for ii = 1:dataset_to_use
    tic
    T = traj{ii};
    N = size(T,1);

    % ---- rows sorted by (vehicle, frame) -> contiguous per-vehicle blocks
    [~, ordV] = sortrows([double(T(:,2)), double(T(:,3))]);
    vSorted   = double(T(ordV,2));
    gStart    = [1; find(diff(vSorted) ~= 0) + 1];
    gEnd      = [gStart(2:end) - 1; numel(vSorted)];

    % Arithmetic stays in single precision throughout, as in the original, so
    % that rounding at comparison and ceil() boundaries matches exactly.
    lane = T(ordV,6);
    yloc = T(ordV,5);
    latMan = ones(N,1); lonMan = ones(N,1);
    for g = 1:numel(gStart)
        s = gStart(g); e = gEnd(g); n = e - s + 1;
        ind = (1:n)';
        la  = lane(s:e);  yy = yloc(s:e);
        % ---- lateral maneuver (+-40 frames)
        ub = min(n, ind+40); lb = max(1, ind-40);
        % The original is an if/elseif, so "turn right" wins when both hold.
        right = la(ub) > la(ind) | la(ind) > la(lb);
        left  = la(ub) < la(ind) | la(ind) < la(lb);
        m  = ones(n,1);
        m(left & ~right) = 2;   % turn left
        m(right) = 3;           % turn right
        latMan(s:e) = m;
        % ---- longitudinal maneuver (+50 / -30 frames)
        ub = min(n, ind+50); lb = max(1, ind-30);
        m  = ones(n,1);
        ok = (ub ~= ind) & (lb ~= ind);
        vHist = (yy(ind) - yy(lb)) ./ (ind - lb);
        vFut  = (yy(ub)  - yy(ind)) ./ (ub - ind);
        m(ok & (vFut./vHist < 0.8)) = 2;              % brake
        lonMan(s:e) = m;
    end
    T(ordV,7) = single(latMan);
    T(ordV,8) = single(lonMan);

    % ---- neighbour grid, frame by frame
    [~, ordF] = sortrows([double(T(:,3)), double(T(:,2))]);
    fSorted   = double(T(ordF,3));
    fStart    = [1; find(diff(fSorted) ~= 0) + 1];
    fEnd      = [fStart(2:end) - 1; numel(fSorted)];
    X = T(:,4); Y = T(:,5); ID = T(:,2);
    for g = 1:numel(fStart)
        rows = ordF(fStart(g):fEnd(g));
        M = numel(rows);
        if M < 2, continue; end
        x = X(rows); y = Y(rows);
        dX = x.' - x;                    % dX(i,j) = x_j - x_i
        dY = y.' - y;
        inRange = (abs(dY) < halfL) & (abs(dX) < halfW);
        % Offsetting the grid origin first (rather than adding half a grid to
        % the difference) keeps the single-precision rounding identical to the
        % original, which matters right at the ceil() boundaries.
        gridMinX = x - halfW;
        gridMinY = y - halfL;
        gx  = ceil((x.' - gridMinX) / cell_width);
        gy  = ceil((y.' - gridMinY) / cell_length);
        loc = double(gy + (gx - 1)*grid_length);
        [i_, j_] = find(inRange & loc ~= grid_cent_location);
        if isempty(i_), continue; end
        locv = loc(sub2ind([M M], i_, j_));
        T(sub2ind(size(T), rows(i_), 13 + locv)) = ID(rows(j_));
    end
    traj{ii} = T;
    fprintf('  dataset %d parsed in %.1f s\n', ii, toc);
end
clear T X Y ID dX dY inRange gx gy loc

%% 3. Merge, build tracks, split -------------------------------------------
disp('Splitting into train, validation and test sets...')
trajAll = vertcat(traj{:});
clear traj

maxVeh = max(trajAll(:,2));
tracks     = cell(dataset_to_use, maxVeh);
firstFrame = zeros(dataset_to_use, maxVeh);     % 0 == vehicle absent
consecutive = true(dataset_to_use, maxVeh);     % frames increase by exactly 1
lastFrame  = zeros(dataset_to_use, maxVeh);
frame31    = inf(dataset_to_use, maxVeh);       % inf == never enough history

[~, ord] = sortrows([double(trajAll(:,1)), double(trajAll(:,2)), double(trajAll(:,3))]);
trajAll  = trajAll(ord, :);
key      = double(trajAll(:,1))*(double(maxVeh)+1) + double(trajAll(:,2));
gStart   = [1; find(diff(key) ~= 0) + 1];
gEnd     = [gStart(2:end) - 1; numel(key)];
for g = 1:numel(gStart)
    s = gStart(g); e = gEnd(g);
    ds = trajAll(s,1); veh = trajAll(s,2);
    blk = trajAll(s:e, 3:end).';                % (11+grid_cells) x frames
    tracks{ds, veh} = blk;
    firstFrame(ds, veh) = blk(1,1);
    lastFrame(ds, veh)  = blk(1,end);
    if size(blk,2) >= 31, frame31(ds, veh) = blk(1,31); end
    % consecutive frames allow an O(1) column lookup downstream
    consecutive(ds, veh) = ~any(diff(blk(1,:)) ~= 1);
end
nGap = nnz(~consecutive & firstFrame > 0);
fprintf('  %d tracks built, %d with frame gaps (slow-path lookup)\n', ...
        nnz(firstFrame > 0), nGap);

trajTr = []; trajVal = []; trajTs = [];
for k = 1:dataset_to_use
    sel = trajAll(:,1) == k;
    ul1 = round(0.7 * max(trajAll(sel,2)));
    ul2 = round(0.8 * max(trajAll(sel,2)));
    trajTr  = [trajTr ; trajAll(sel & trajAll(:,2) <= ul1, :)];
    trajVal = [trajVal; trajAll(sel & trajAll(:,2) >  ul1 & trajAll(:,2) <= ul2, :)];
    trajTs  = [trajTs ; trajAll(sel & trajAll(:,2) >  ul2, :)];
end

%% 4. Filter edge cases (need 3s history and 5s future) --------------------
disp('Filtering edge cases...')
keepRows = @(A) A( ...
    frame31(sub2ind(size(frame31), double(A(:,1)), double(A(:,2)))) <= double(A(:,3)) & ...
    lastFrame(sub2ind(size(lastFrame), double(A(:,1)), double(A(:,2)))) >= double(A(:,3)) + 50, :);
trajTr = keepRows(trajTr); trajVal = keepRows(trajVal); trajTs = keepRows(trajTs);
fprintf('  after edge filtering: train %d / val %d / test %d\n', ...
        size(trajTr,1), size(trajVal,1), size(trajTs,1));

%% 5. Save -----------------------------------------------------------------
if SAVE_RAW
    disp('Saving raw mat files...')
    save(fullfile(raw_folder,'gridTrainAround.mat'), 'trajTr', 'tracks','-v7.3');
    save(fullfile(raw_folder,'gridValAround.mat'),   'trajVal','tracks','-v7.3');
    save(fullfile(raw_folder,'gridTestAround.mat'),  'trajTs', 'tracks','-v7.3');
end

disp('### Train data:')
traj = nbhCheckerFast(trajTr, tracks, firstFrame, frame31, consecutive);
save(fullfile(post_folder,'gridTrainAround.mat'),'traj','tracks','-v7.3');
traj = targSpecFast(traj, tracks, firstFrame, consecutive);
save(fullfile(fix_tar_folder,'gridTrainAround.mat'),'traj','tracks','-v7.3');

disp('### Validation data:')
traj = nbhCheckerFast(trajVal, tracks, firstFrame, frame31, consecutive);
save(fullfile(post_folder,'gridValAround.mat'),'traj','tracks','-v7.3');
traj = targSpecFast(traj, tracks, firstFrame, consecutive);
save(fullfile(fix_tar_folder,'gridValAround.mat'),'traj','tracks','-v7.3');

disp('### Test data:')
traj = nbhCheckerFast(trajTs, tracks, firstFrame, frame31, consecutive);
save(fullfile(post_folder,'gridTestAround.mat'),'traj','tracks','-v7.3');
traj = targSpecFast(traj, tracks, firstFrame, consecutive);
save(fullfile(fix_tar_folder,'gridTestAround.mat'),'traj','tracks','-v7.3');

disp('All done.')
