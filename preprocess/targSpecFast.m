function final_traj = targSpecFast(traj, tracks, firstFrame, consecutive)
%TARGSPECFAST  O(1)-lookup version of PiP's targSpecFunc.
%  Reduces the up-to-125 grid occupants of each row to the <=8 vehicles used
%  as prediction targets: preceding / following in the ego lane plus up to
%  three vehicles on each adjacent lane.
gridRows    = 25;
gridCols    = 5;
nbrStartIdx = 14;
nbrEndIdx   = 13 + gridRows*gridCols;
dataNum     = size(traj, 1);
final_traj  = traj;

fprintf('targSpecFast processing: ');
for i = 1:dataNum
    targsGrid = traj(i, nbrStartIdx:nbrEndIdx);
    targsVeh  = nonzeros(targsGrid);
    targsNum  = numel(targsVeh);
    if targsNum < 2, continue; end

    dsId    = double(traj(i,1));
    frameId = double(traj(i,3));
    centX   = traj(i,4);
    centY   = traj(i,5);
    laneId  = traj(i,6);

    % Position and lane of every target at this frame
    targsInfo = zeros(4, targsNum);
    targsInfo(1,:) = targsVeh;
    valid = true(1, targsNum);
    for j = 1:targsNum
        v   = double(targsVeh(j));
        trk = tracks{dsId, v};
        if isempty(trk), valid(j) = false; continue; end
        if consecutive(dsId, v)
            col = frameId - firstFrame(dsId, v) + 1;
            if col < 1 || col > size(trk,2), col = []; end
        else
            col = find(trk(1,:) == frameId, 1);
        end
        if isempty(col)
            valid(j) = false; continue
        end
        targsInfo(2:4, j) = trk(2:4, col);
    end
    targsInfo = targsInfo(:, valid);
    if isempty(targsInfo), continue; end

    targsInfo(2,:) = targsInfo(2,:) - double(centX);
    targsInfo(3,:) = targsInfo(3,:) - double(centY);
    precedingInfo = targsInfo(:, targsInfo(4,:)==laneId & targsInfo(3,:)>0);
    followingInfo = targsInfo(:, targsInfo(4,:)==laneId & targsInfo(3,:)<0);
    leftLaneInfo  = targsInfo(:, targsInfo(4,:)< laneId);
    rightLaneInfo = targsInfo(:, targsInfo(4,:)> laneId);

    qualVehs = zeros(1,8);
    if size(precedingInfo,2) == 1
        qualVehs(1) = precedingInfo(1);
    elseif size(precedingInfo,2) > 1
        [~, idx] = min(precedingInfo(3,:));  qualVehs(1) = precedingInfo(1, idx);
    end
    if size(followingInfo,2) == 1
        qualVehs(2) = followingInfo(1);
    elseif size(followingInfo,2) > 1
        [~, idx] = max(followingInfo(3,:));  qualVehs(2) = followingInfo(1, idx);
    end
    qualVehs(3:5) = pickLane(leftLaneInfo);
    qualVehs(6:8) = pickLane(rightLaneInfo);

    % Zero out every grid occupant that is not a selected target
    row = final_traj(i, nbrStartIdx:nbrEndIdx);
    drop = row > 0 & ~ismember(row, qualVehs(qualVehs>0));
    row(drop) = 0;
    final_traj(i, nbrStartIdx:nbrEndIdx) = row;

    if mod(i, 200000) == 0, fprintf('%.2f ... ', i/dataNum); end
end

fprintf('\nPast #dataNum: %d ===>>> Now #dataNum: %d\n', size(traj,1), size(final_traj,1));
fprintf('Past #targets: %d ===>>> Now #targets: %d\n', ...
        nnz(traj(:, nbrStartIdx:nbrEndIdx)), nnz(final_traj(:, nbrStartIdx:nbrEndIdx)));
end

function q = pickLane(laneInfo)
% Nearest vehicle on the adjacent lane plus the one ahead of and behind it.
q = zeros(1,3);
n = size(laneInfo, 2);
if n == 1
    q(1) = laneInfo(1);
elseif n == 2
    q(1:2) = laneInfo(1,:);
elseif n > 2
    [~, idx] = min(laneInfo(2,:).^2 + laneInfo(3,:).^2);
    q(1)  = laneInfo(1, idx);
    centY = laneInfo(3, idx);
    front = laneInfo(:, laneInfo(3,:) > centY);
    if ~isempty(front), [~, id2] = min(front(3,:)); q(2) = front(1, id2); end
    back  = laneInfo(:, laneInfo(3,:) < centY);
    if ~isempty(back),  [~, id3] = max(back(3,:));  q(3) = back(1, id3);  end
end
end
