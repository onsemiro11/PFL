%% highD counterpart of preprocess_ngsim_fast.m
%  The parsing, grid construction, splitting and edge filtering are identical to
%  the NGSIM pipeline -- highD is made to look like NGSIM by highd_to_ngsim.py,
%  which resamples to 10 Hz and writes one file per travel direction, so the
%  30/50 frame history/future windows and the +-40 maneuver windows carry over
%  unchanged. What differs here is only the file list (2 files per recording
%  instead of 6 periods) and the absence of the US101 lane merge.
%  Env vars:
%    HIGHD_RAW   folder holding the NN-fwd.csv / NN-bck.csv (default ./raw_highd/)
%    HIGHD_OUT   output root                                (default ./dataset/highd/)
%    HIGHD_RECS  use only the first N recordings            (default all found)
%    HIGHD_SAVE_RAW '1' to also dump the pre-checker mat files (default '0')
clear; clc;

RAW_DIR  = getenv('HIGHD_RAW');   if isempty(RAW_DIR),  RAW_DIR  = './raw_highd/';    end
OUT_ROOT = getenv('HIGHD_OUT');   if isempty(OUT_ROOT), OUT_ROOT = './dataset/highd/';end
MAX_REC  = str2double(getenv('HIGHD_RECS'));
SAVE_RAW = strcmp(getenv('HIGHD_SAVE_RAW'), '1');

grid_length = 25; grid_width = 5; cell_length = 8; cell_width = 7;
raw_folder     = fullfile(OUT_ROOT, sprintf('%dx%d_nofL_raw', grid_length, grid_width));
post_folder    = fullfile(OUT_ROOT, sprintf('%dx%d_nofL',     grid_length, grid_width));
fix_tar_folder = fullfile(OUT_ROOT, sprintf('%dx%d_8Veh_nofL',grid_length, grid_width));
for f = {raw_folder, post_folder, fix_tar_folder}, if ~exist(f{1},'dir'), mkdir(f{1}); end, end

grid_cells         = grid_length * grid_width;
grid_cent_location = ceil(grid_cells*0.5);

%% 1. Load ------------------------------------------------------------------
disp('Loading data...')
listing = dir(fullfile(RAW_DIR, '*-fwd.csv'));
recIds  = sort(cellfun(@(n) str2double(n(1:2)), {listing.name}));
if ~isnan(MAX_REC), recIds = recIds(1:min(MAX_REC, numel(recIds))); end
files = {};
for r = recIds
    files{end+1} = sprintf('%02d-fwd.csv', r); %#ok<SAGROW>
    files{end+1} = sprintf('%02d-bck.csv', r); %#ok<SAGROW>
end
dataset_to_use = numel(files);
fprintf('  %d recordings -> %d datasets\n', numel(recIds), dataset_to_use);

traj = cell(1, dataset_to_use);
for k = 1:dataset_to_use
    fp = fullfile(RAW_DIR, files{k});
    D  = readmatrix(fp, 'FileType', 'text', 'Delimiter', ',');
    fprintf('  dataset %d: %s  (%d rows)\n', k, files{k}, size(D,1));
    % Columns match the classic per-period NGSIM txt, so the same selection applies:
    %  1 VehId 2 FrameId 3 TotalFrames 4 GlobalTime 5 LocalX 6 LocalY 7 GlobalX
    %  8 GlobalY 9 Length 10 Width 11 Class 12 Vel 13 Acc 14 LaneId
    D = single([k*ones(size(D,1),1,'like',D), D]);
    D = D(:, [1,2,3,6,7,15,10,11,12,13,14]);   % ds veh frame x y lane len wid class vel acc
    traj{k} = [D(:,1:6), zeros(size(D,1),2,'single'), D(:,7:11), zeros(size(D,1),grid_cells,'single')];
end
clear D listing

% Use the vehicle's center as its location, keep Y positive
for k = 1:dataset_to_use
    traj{k}(:,5) = traj{k}(:,5) - 0.5*traj{k}(:,9);
    off = min(traj{k}(:,5));
    if off < 0, traj{k}(:,5) = traj{k}(:,5) - off; end
end

%% 2. Parse maneuver labels and neighbour grid ------------------------------
disp('Parsing fields...')
halfL = 0.5*grid_length*cell_length;
halfW = 0.5*grid_width *cell_width;

for ii = 1:dataset_to_use
    tic
    T = traj{ii};
    N = size(T,1);

    [~, ordV] = sortrows([double(T(:,2)), double(T(:,3))]);
    vSorted   = double(T(ordV,2));
    gStart    = [1; find(diff(vSorted) ~= 0) + 1];
    gEnd      = [gStart(2:end) - 1; numel(vSorted)];

    lane = T(ordV,6);
    yloc = T(ordV,5);
    latMan = ones(N,1); lonMan = ones(N,1);
    for g = 1:numel(gStart)
        s = gStart(g); e = gEnd(g); n = e - s + 1;
        ind = (1:n)';
        la  = lane(s:e);  yy = yloc(s:e);
        ub = min(n, ind+40); lb = max(1, ind-40);
        right = la(ub) > la(ind) | la(ind) > la(lb);
        left  = la(ub) < la(ind) | la(ind) < la(lb);
        m  = ones(n,1);
        m(left & ~right) = 2;
        m(right) = 3;
        latMan(s:e) = m;
        ub = min(n, ind+50); lb = max(1, ind-30);
        m  = ones(n,1);
        ok = (ub ~= ind) & (lb ~= ind);
        vHist = (yy(ind) - yy(lb)) ./ (ind - lb);
        vFut  = (yy(ub)  - yy(ind)) ./ (ub - ind);
        m(ok & (vFut./vHist < 0.8)) = 2;
        lonMan(s:e) = m;
    end
    T(ordV,7) = single(latMan);
    T(ordV,8) = single(lonMan);

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
        dX = x.' - x;
        dY = y.' - y;
        inRange = (abs(dY) < halfL) & (abs(dX) < halfW);
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
firstFrame = zeros(dataset_to_use, maxVeh);
consecutive = true(dataset_to_use, maxVeh);
lastFrame  = zeros(dataset_to_use, maxVeh);
frame31    = inf(dataset_to_use, maxVeh);

[~, ord] = sortrows([double(trajAll(:,1)), double(trajAll(:,2)), double(trajAll(:,3))]);
trajAll  = trajAll(ord, :);
clear ord
key      = double(trajAll(:,1))*(double(maxVeh)+1) + double(trajAll(:,2));
gStart   = [1; find(diff(key) ~= 0) + 1];
gEnd     = [gStart(2:end) - 1; numel(key)];
for g = 1:numel(gStart)
    s = gStart(g); e = gEnd(g);
    ds = trajAll(s,1); veh = trajAll(s,2);
    blk = trajAll(s:e, 3:end).';
    tracks{ds, veh} = blk;
    firstFrame(ds, veh) = blk(1,1);
    lastFrame(ds, veh)  = blk(1,end);
    if size(blk,2) >= 31, frame31(ds, veh) = blk(1,31); end
    consecutive(ds, veh) = ~any(diff(blk(1,:)) ~= 1);
end
clear key blk
nGap = nnz(~consecutive & firstFrame > 0);
fprintf('  %d tracks built, %d with frame gaps (slow-path lookup)\n', ...
        nnz(firstFrame > 0), nGap);

% Split on vehicle id, as PiP does: 70% train / 10% val / 20% test. highD ids are
% dense per recording, so the cut points are taken on the sorted unique ids.
trajTr = []; trajVal = []; trajTs = [];
for k = 1:dataset_to_use
    sel = trajAll(:,1) == k;
    uv  = sort(unique(trajAll(sel,2)));
    if isempty(uv), continue; end
    ul1 = uv(max(1, round(0.7*numel(uv))));
    ul2 = uv(max(1, round(0.8*numel(uv))));
    trajTr  = [trajTr ; trajAll(sel & trajAll(:,2) <= ul1, :)];
    trajVal = [trajVal; trajAll(sel & trajAll(:,2) >  ul1 & trajAll(:,2) <= ul2, :)];
    trajTs  = [trajTs ; trajAll(sel & trajAll(:,2) >  ul2, :)];
end
clear trajAll sel uv

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
clear trajTr

disp('### Validation data:')
traj = nbhCheckerFast(trajVal, tracks, firstFrame, frame31, consecutive);
save(fullfile(post_folder,'gridValAround.mat'),'traj','tracks','-v7.3');
traj = targSpecFast(traj, tracks, firstFrame, consecutive);
save(fullfile(fix_tar_folder,'gridValAround.mat'),'traj','tracks','-v7.3');
clear trajVal

disp('### Test data:')
traj = nbhCheckerFast(trajTs, tracks, firstFrame, frame31, consecutive);
save(fullfile(post_folder,'gridTestAround.mat'),'traj','tracks','-v7.3');
traj = targSpecFast(traj, tracks, firstFrame, consecutive);
save(fullfile(fix_tar_folder,'gridTestAround.mat'),'traj','tracks','-v7.3');

disp('All done.')
