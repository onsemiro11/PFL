function final_traj = nbhCheckerFast(traj, tracks, firstFrame, frame31, consecutive)
%NBHCHECKERFAST  O(1)-lookup version of PiP's nbhCheckerFunc.
%  Keeps a row only if
%    1. it has at least one neighbour (= prediction target),
%    2. every target exists in tracks with >= 3 s of history,
%    3. the ego vehicle appears in every target's own grid at that frame,
%    4. every sub-neighbour of every target exists in tracks.
rowNum      = size(traj, 1);
columnNum   = size(traj, 2);
vehIdxLimit = size(tracks, 2);
existsVeh   = firstFrame > 0;
keep        = false(rowNum, 1);

fprintf('nbhCheckerFast processing: ');
for k = 1:rowNum
    ds  = double(traj(k,1));
    ego = double(traj(k,2));
    t   = double(traj(k,3));

    neighbours = nonzeros(traj(k, 14:columnNum));
    if isempty(neighbours) || any(neighbours > vehIdxLimit), continue; end

    keepThisRow = true;
    for i = 1:numel(neighbours)
        nbID = double(neighbours(i));
        if ~existsVeh(ds, nbID) || t < frame31(ds, nbID)
            keepThisRow = false; break
        end
        trk = tracks{ds, nbID};
        col = trackCol(trk, t, firstFrame(ds, nbID), consecutive(ds, nbID));
        if isempty(col)
            keepThisRow = false; break
        end
        sub_neighbours = nonzeros(trk(12:(columnNum-2), col));
        if isempty(sub_neighbours) || any(sub_neighbours > vehIdxLimit) ...
                || all(sub_neighbours ~= ego) ...
                || ~all(existsVeh(sub2ind(size(existsVeh), ...
                        ds*ones(numel(sub_neighbours),1), double(sub_neighbours))))
            keepThisRow = false; break
        end
    end
    keep(k) = keepThisRow;

    if mod(k, 200000) == 0, fprintf('%.2f ... ', k/rowNum); end
end

final_traj = traj(keep, :);
fprintf('\nPast #data: %d ===>>> Now #data: %d \n\n', rowNum, size(final_traj,1));
end

function col = trackCol(trk, t, first, isConsecutive)
%TRACKCOL  Column of frame t inside a track; empty when the frame is missing.
if isConsecutive
    col = t - first + 1;
    if col < 1 || col > size(trk,2), col = []; end
else
    col = find(trk(1,:) == t, 1);
end
end
