import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import quintic_spline, fitting_traj_by_qs

## NGSIM / HighD datasets are publicly available datasets
## First using preprocess code to make them into mat file with the following format:

'''
% Data: #row = data number, #column = 138 (13+grid_num)
%{
0: Dataset Id
1: Vehicle Id
|2 : Frame Id
|3 : Local X
|4 : Local Y
|5 : Lane Id
|6 : Lateral maneuver
|7 : Longitudinal maneuver
|8 : Length
|9 : Width
|10: Class label
|11: Velocity
|12: Accerlation
|13-137: Neighbor Car Ids at grid location
%}
'''

'''
% Tracks: cells: {Dataset_Id * Vehicle_Id}, each cell: #row = 136 (11+grid_num), #column=totalFramNum
%{
|0 : Frame Id
|1 : Local X
|2 : Local Y
|3 : Lane Id
|4 : Lateral maneuver
|5 : Longitudinal maneuver
|6 : Length
|7 : Width
|8 : Class label
|9 : Velocity
|10: Accerlation
|11-135: Neighbor Car Ids at grid location
'''

### Class for the highway trajectory datasets (NISIM, HighD, etc.)
class highwayTrajDataset(Dataset):

    def __init__(self, path, t_h=30, t_f=50, d_s=2,
                 enc_size=64, targ_enc_size=112, grid_size=(25, 5), fit_plan_traj=False, fit_plan_further_ds=1):
        if not os.path.exists(path):
            raise RuntimeError("{} not exists!!".format(path))
        if path.endswith('.mat'):
            f = h5py.File(path, 'r')
            f_tracks = f['tracks']
            track_cols, track_rows = f_tracks.shape
            self.Data = np.transpose(f['traj'])
            self.Tracks = []
            for i in range(track_rows):
                self.Tracks.append([  np.transpose(f[f_tracks[j][i]][:]) for j in range(track_cols)  ])
        else:
            raise RuntimeError("Path should be end with '.mat' for file or '/' for folder")

        # If torch version >= 1.2.0
        if int(torch.__version__[0])>=1 and int(torch.__version__[2])>=2:
            self.mask_num_type = torch.bool
        else:
            self.mask_num_type = torch.uint8

        self.t_h = t_h  # length of track history.
        self.t_f = t_f  # length of predicted trajectory.
        self.d_s = d_s  # downsampling rate of all trajectories to be processed.
        self.enc_size = enc_size
        self.targ_enc_size = targ_enc_size
        self.hist_len = self.t_h // self.d_s + 1  # data length of the history trajectory
        self.fut_len = self.t_f // self.d_s       # data length of the future  trajectory
        self.plan_len = self.t_f // self.d_s      # data length of the planning  trajectory

        self.fit_plan_traj = fit_plan_traj              # Fitting the future planned trajectory in testing/evaluation.
        self.further_ds_plan = fit_plan_further_ds      # Further downsampling to restrict the planning info

        self.cell_length = 8
        self.cell_width = 7
        self.grid_size = grid_size                # size of social context grid
        self.grid_cells = grid_size[0] * grid_size[1]
        self.grid_length = self.cell_length * grid_size[0]
        self.grid_width = self.cell_width * grid_size[1]

        self._buildTrackIndex()

    def _buildTrackIndex(self):
        # Locating a frame inside a track is the innermost operation of the whole
        # pipeline, so cache what is needed to do it without scanning the track.
        # Most tracks have consecutive frames and allow plain arithmetic; the rest
        # fall back to a binary search.
        n_ds = len(self.Tracks)
        n_veh = max(len(t) for t in self.Tracks)
        self._trackFirst = np.zeros((n_ds, n_veh), dtype=np.int64)
        self._trackConsec = np.zeros((n_ds, n_veh), dtype=bool)
        for i in range(n_ds):
            for j, trk in enumerate(self.Tracks[i]):
                # Absent vehicles are stored by MATLAB as an empty 1-D placeholder.
                if trk.ndim < 2 or trk.shape[1] == 0:
                    continue
                frames = trk[0]
                self._trackFirst[i, j] = int(frames[0])
                self._trackConsec[i, j] = frames.size < 2 or bool(np.all(np.diff(frames) == 1))

    def __len__(self):
        return len(self.Data)

    ## Functions of retrieving information according to the item's index
    def itsDsId(self, idx):
        return self.Data[idx, 0].astype(int)

    def itsPlanVehId(self, idx):
        return self.Data[idx, 1].astype(int)

    def itsTime(self, idx):
        return self.Data[idx, 2]

    def itsLocation(self, idx):
        return self.Data[idx, 3:5]

    def itsPlanVehBehavior(self, idx):
        return int(self.Data[idx, 6] + (self.Data[idx, 7] - 1) * 3)

    def itsPlanVehSize(self, idx):
        return self.Data[idx, 8:10]

    def itsPlanVehDynamic(self, idx):
        planVel, planAcc = self.getDynamic(self.itsDsId(idx), self.itsPlanVehId(idx), self.itsTime(idx))
        return planVel, planAcc

    def itsCentGrid(self, idx):
        return self.Data[idx, 13:].astype(int)

    def itsTargVehsId(self, idx):
        centGrid = self.itsCentGrid(idx)
        targVehsId = centGrid[np.nonzero(centGrid)]
        return targVehsId

    def itsNbrVehsId(self, idx):
        dsId = self.itsDsId(idx)
        planVehId = self.itsPlanVehId(idx)
        targVehsId = self.itsTargVehsId(idx)
        t = self.itsTime(idx)
        nbrVehsId = np.array([], dtype=np.int64)
        for target in targVehsId:
            subGrid = self.getGrid(dsId, target, t)
            subIds = subGrid[np.nonzero(subGrid)]
            for i in subIds:
                if i==planVehId or any(i==targVehsId) or any(i==nbrVehsId):
                    continue
                else:
                    nbrVehsId = np.append(nbrVehsId, i)
        return nbrVehsId

    def itsTargsCentLoc(self, idx):
        dsId = self.itsDsId(idx)
        t = self.itsTime(idx)
        centGrid = self.itsCentGrid(idx)
        targsCenterLoc = np.empty((0,2), dtype=np.float32)
        for target in centGrid:
            if target:
                targsCenterLoc = np.vstack([targsCenterLoc, self.getLocation(dsId, target, t)])
        return torch.from_numpy(targsCenterLoc)

    def itsAllAroundSizes(self, idx):
        dsId = self.itsDsId(idx)
        centGrid = self.itsCentGrid(idx)
        t = self.itsTime(idx)
        planVehSize = []
        targVehSizes = []
        nbsVehSizes = []
        planVehSize.append(self.getSize(dsId, self.itsPlanVehId(idx)))
        for i, target in enumerate(centGrid):
            if target:
                targVehSizes.append(self.getSize(dsId, target))
                targVehGrid = self.getGrid(dsId, target, t)
                for targetNb in targVehGrid:
                    if targetNb:
                        nbsVehSizes.append(self.getSize(dsId, targetNb))
        return np.asarray(planVehSize), np.asarray(targVehSizes), np.asarray(nbsVehSizes)

    ## Functions for retrieving trajectory data with absolute coordinate, mainly used for visualization
    def itsAllGroundTruthTrajs(self, idx):
        return [self.absPlanTraj(idx), self.absTargsTraj(idx), self.absNbrsTraj(idx)]

    def absPlanTraj(self, idx):
        dsId = self.itsDsId(idx)
        planVeh = self.itsPlanVehId(idx)
        t = self.itsTime(idx)
        colIndex = self.getTracksCol(dsId, planVeh, t)
        vehTrack = self.Tracks[dsId - 1][planVeh - 1].transpose()
        planHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]
        planFut = vehTrack[(colIndex + self.d_s): (colIndex + self.t_f + 1): self.d_s, 1:3]
        return [planHis, planFut]

    def absTargsTraj(self, idx):
        dsId = self.itsDsId(idx)
        targVehs = self.itsTargVehsId(idx)
        t = self.itsTime(idx)
        targHisList, targFutList = [], []
        for targVeh in targVehs:
            colIndex = self.getTracksCol(dsId, targVeh, t)
            vehTrack = self.Tracks[dsId - 1][targVeh - 1].transpose()
            targHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]
            targFut = vehTrack[(colIndex + self.d_s): (colIndex + self.t_f + 1): self.d_s, 1:3]
            targHisList.append(targHis)
            targFutList.append(targFut)
        return [targHisList, targFutList]

    def absNbrsTraj(self, idx):
        dsId = self.itsDsId(idx)
        nbrVehs = self.itsNbrVehsId(idx)
        t = self.itsTime(idx)
        nbrHisList, nbrFutList = [], []
        for nbrVeh in nbrVehs:
            colIndex = self.getTracksCol(dsId, nbrVeh, t)
            vehTrack = self.Tracks[dsId - 1][nbrVeh - 1].transpose()
            targHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]
            nbrHisList.append(targHis)
        return [nbrHisList, nbrFutList]

    def batchTargetVehsInfo(self, idxs):
        count = 0
        dsIds = np.zeros(len(idxs)*self.grid_cells, dtype=int)
        vehIds = np.zeros(len(idxs)*self.grid_cells, dtype=int)
        for idx in idxs:
            dsId = self.itsDsId(idx)
            targets = self.itsCentGrid(idx)
            targetsIndex = np.nonzero(targets)
            for index in targetsIndex[0]:
                dsIds[count] = dsId
                vehIds[count] = targets[index]
                count += 1
        return [dsIds[:count], vehIds[:count]]

    ## Avoid searching the correspond column for too many times.
    def getTracksCol(self, dsId, vehId, t):
        i, j = dsId - 1, vehId - 1
        if self._trackConsec[i, j]:
            return int(t) - int(self._trackFirst[i, j])
        return int(np.searchsorted(self.Tracks[i][j][0, :], t))

    ## Get the vehicle's location from tracks
    def getLocation(self, dsId, vehId, t):
        colIndex = self.getTracksCol(dsId, vehId, t)
        location = self.getLocationByCol(dsId, vehId, colIndex)
        return location
    def getLocationByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][1:3, colIndex].transpose()

    ## Get the vehicle's maneuver given dataset id, vehicle id and time point t.
    def getManeuver(self, dsId, vehId, t):
        colIndex = self.getTracksCol(dsId, vehId, t)
        lat_lon_maneuvers = self.getManeuverByCol(dsId, vehId, colIndex)
        return lat_lon_maneuvers
    def getManeuverByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][4:6, colIndex].astype(int)

    ## Get the vehicle's nearby neighbours
    def getGrid(self, dsId, vehId, t):
        colIndex = self.getTracksCol(dsId, vehId, t)
        grid = self.getGridByCol(dsId, vehId, colIndex)
        return grid
    def getGridByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][11:, colIndex].astype(int)

    ## Get the vehicle's dynamic (velocity & acceleration) given dataset id, vehicle id and time point t.
    def getDynamic(self, dsId, vehId, t):
        colIndex = self.getTracksCol(dsId, vehId, t)
        vel_acc = self.getDynamicByCol(dsId, vehId, colIndex)
        return vel_acc
    def getDynamicByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][9:11, colIndex]

    def getVelc(self, dsId, vehId, t):
        colIndex = self.getTracksCol(dsId, vehId, t)
        vel_acc = self.getVelcByCol(dsId, vehId, colIndex)
        return vel_acc
    def getVelcByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][9:10, colIndex]


    ## Get the vehicle's size (length & width) given dataset id and vehicle id
    def getSize(self, dsId, vehId):
        length_width = self.Tracks[dsId - 1][vehId - 1][6:8, 0]
        return length_width

    ## Helper function to get track history
    def getHistory(self, dsId, vehId, refVehId, t, wholePeriod=False):
        if vehId == 0:
            # if return empty, it denotes there's no vehicle in that grid.
            return np.empty([0, 2])
        else:
            vehColIndex = self.getTracksCol(dsId, vehId, t)
            refColIndex = self.getTracksCol(dsId, refVehId, t)
            vehTrack = self.Tracks[dsId - 1][vehId - 1][1:3].transpose()
            refTrack = self.Tracks[dsId - 1][refVehId - 1][1:3].transpose()
            # Use the sequence of trajectory or just the last instance as the refPos
            if wholePeriod:
                refStpt = np.maximum(0, refColIndex - self.t_h)
                refEnpt = refColIndex + 1
                refPos = refTrack[refStpt:refEnpt:self.d_s, :]
            else:
                refPos = np.tile(refTrack[refColIndex, :], (self.hist_len, 1))
            stpt = np.maximum(0, vehColIndex - self.t_h)
            enpt = vehColIndex + 1
            vehPos = vehTrack[stpt:enpt:self.d_s, :]
            if len(vehPos) < self.hist_len:
                histPart = vehPos - refPos[-len(vehPos)::]
                paddingPart = np.tile(histPart[0, :], (self.hist_len - len(vehPos), 1))
                hist = np.concatenate((paddingPart, histPart), axis=0)
                return hist
            else:
                hist = vehPos - refPos
                return hist

    def getdifVel(self, dsId, vehId, refVehId, t, wholePeriod=False):
        if vehId == 0:
            # if return empty, it denotes there's no vehicle in that grid.
            return np.empty([0, 1])
        else:
            vehColIndex = self.getTracksCol(dsId, vehId, t)
            refColIndex = self.getTracksCol(dsId, refVehId, t)
            vehTrackvel = self.Tracks[dsId - 1][vehId - 1][9:10].transpose()
            refTrackvel = self.Tracks[dsId - 1][refVehId - 1][9:10].transpose()
            # Use the sequence of trajectory or just the last instance as the refPos
            if wholePeriod:
                refStpt = np.maximum(0, refColIndex - self.t_h)
                refEnpt = refColIndex + 1
                refPos = refTrackvel[refStpt:refEnpt:self.d_s, :]
            else:
                refPos = np.tile(refTrackvel[refColIndex, :], (self.hist_len, 1))
            stpt = np.maximum(0, vehColIndex - self.t_h)
            enpt = vehColIndex + 1
            vehPos = vehTrackvel[stpt:enpt:self.d_s, :]
            if len(vehPos) < self.hist_len:
                histPart = vehPos-refPos[-len(vehPos)::]
                paddingPart = np.tile(histPart[0, :], (self.hist_len - len(vehPos), 1))
                hist = np.concatenate((paddingPart, histPart), axis=0)
                return hist
            else:
                hist = vehPos-refPos
                return hist


    def getVel(self, dsId, vehId, refVehId, t, wholePeriod=False):
        if vehId == 0:
            # if return empty, it denotes there's no vehicle in that grid.
            return np.empty([0, 1])
        else:
            vehColIndex = self.getTracksCol(dsId, vehId, t)
            refColIndex = self.getTracksCol(dsId, refVehId, t)
            vehTrackvel = self.Tracks[dsId - 1][vehId - 1][9:10].transpose()
            refTrackvel = self.Tracks[dsId - 1][refVehId - 1][9:10].transpose()
            # Use the sequence of trajectory or just the last instance as the refPos
            if wholePeriod:
                refStpt = np.maximum(0, refColIndex - self.t_h)
                refEnpt = refColIndex + 1
                refPos = refTrackvel[refStpt:refEnpt:self.d_s, :]
            else:
                refPos = np.tile(refTrackvel[refColIndex, :], (self.hist_len, 1))
            stpt = np.maximum(0, vehColIndex - self.t_h)
            enpt = vehColIndex + 1
            vehPos = vehTrackvel[stpt:enpt:self.d_s, :]
            if len(vehPos) < self.hist_len:
                histPart = vehPos
                paddingPart = np.tile(histPart[0, :], (self.hist_len - len(vehPos), 1))
                hist = np.concatenate((paddingPart, histPart), axis=0)
                return hist
            else:
                hist = vehPos
                return hist

    def getAcc(self, dsId, vehId, refVehId, t, wholePeriod=False):
        if vehId == 0:
            # if return empty, it denotes there's no vehicle in that grid.
            return np.empty([0, 1])
        else:
            vehColIndex = self.getTracksCol(dsId, vehId, t)
            refColIndex = self.getTracksCol(dsId, refVehId, t)
            vehTrackvel = self.Tracks[dsId - 1][vehId - 1][10:11].transpose()
            refTrackvel = self.Tracks[dsId - 1][refVehId - 1][10:11].transpose()
            # Use the sequence of trajectory or just the last instance as the refPos
            if wholePeriod:
                refStpt = np.maximum(0, refColIndex - self.t_h)
                refEnpt = refColIndex + 1
                refPos = refTrackvel[refStpt:refEnpt:self.d_s, :]
            else:
                refPos = np.tile(refTrackvel[refColIndex, :], (self.hist_len, 1))
            stpt = np.maximum(0, vehColIndex - self.t_h)
            enpt = vehColIndex + 1
            vehPos = vehTrackvel[stpt:enpt:self.d_s, :]
            if len(vehPos) < self.hist_len:
                histPart = vehPos
                paddingPart = np.tile(histPart[0, :], (self.hist_len - len(vehPos), 1))
                hist = np.concatenate((paddingPart, histPart), axis=0)
                return hist
            else:
                hist = vehPos
                return hist


    ## Helper function to get track future
    def getFuture(self, dsId, vehId, t):
        colIndex = self.getTracksCol(dsId, vehId, t)
        futTraj = self.getFutureByCol(dsId, vehId, colIndex)
        return futTraj
    def getFutureByCol(self, dsId, vehId, colIndex):
        vehTrack = self.Tracks[dsId - 1][vehId - 1].transpose()
        refPos = self.Tracks[dsId - 1][vehId - 1][1:3, colIndex].transpose()
        stpt = colIndex + self.d_s
        enpt = np.minimum(len(vehTrack), colIndex + self.t_f + 1)
        futTraj = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return futTraj

    def getPlanFuture(self, dsId, planId, refVehId, t):
        # Traj of the reference veh
        refColIndex = self.getTracksCol(dsId, refVehId, t)
        refPos = self.Tracks[dsId - 1][refVehId - 1][1:3, refColIndex].transpose()
        # Traj of the planned veh
        planColIndex = self.getTracksCol(dsId, planId, t)
        stpt = planColIndex
        enpt = planColIndex + self.t_f + 1
        planGroundTrue = self.Tracks[dsId - 1][planId - 1][1:3, stpt:enpt:self.d_s].transpose()
        planFut = planGroundTrue.copy()
        # Fitting the downsampled waypoints as the planned trajectory in testing and evaluation.
        if self.fit_plan_traj:
            wayPoint        = np.arange(0, self.t_f + self.d_s, self.d_s)
            wayPoint_to_fit = np.arange(0, self.t_f + 1, self.d_s * self.further_ds_plan)
            planFut_to_fit = planFut[::self.further_ds_plan, ]
            laterParam = fitting_traj_by_qs(wayPoint_to_fit, planFut_to_fit[:, 0])
            longiParam = fitting_traj_by_qs(wayPoint_to_fit, planFut_to_fit[:, 1])
            planFut[:, 0] = quintic_spline(wayPoint, *laterParam)
            planFut[:, 1] = quintic_spline(wayPoint, *longiParam)
        revPlanFut = np.flip(planFut[1:,] - refPos, axis=0).copy()
        return revPlanFut

    def __getitem__(self, idx):
        dsId = self.itsDsId(idx)
        centVehId = self.itsPlanVehId(idx)
        t = self.itsTime(idx)
        centGrid = self.itsCentGrid(idx)
        planGridLocs = []
        targsHists = []
        targsFuts = []
        targsLonEnc = []
        targsLatEnc = []
        nbsHists = []
        planFuts = []
        nbsVel = []
        nbsAcc = []
        nbsdifVel = []
        nbsGridLocs = []
        vel_hist=[]
        acc_hist=[]
        targsVehs = np.zeros(self.grid_cells)
        for id, target in enumerate(centGrid):
            if target:
                targetColumn = self.getTracksCol(dsId, target, t)
                # Get the grid of each neighbour vehicle.
                grid = self.getGridByCol(dsId, target, targetColumn)
                # Targets history and future
                targsVehs[id] = target
                targsHists.append(self.getHistory(dsId, target, target, t))
                targsFuts.append(self.getFutureByCol(dsId, target, targetColumn))
                vel_hist.append(self.getVel(dsId, target, target, t))
                acc_hist.append(self.getAcc(dsId, target, target, t))
                # Targets maneuvers
                latMan, lonMan = self.getManeuverByCol(dsId, target, targetColumn)
                lat_enc = np.zeros([3])
                lon_enc = np.zeros([2])
                lat_enc[latMan - 1] = 1
                lon_enc[lonMan - 1] = 1
                targsLatEnc.append(lat_enc)
                targsLonEnc.append(lon_enc)
                # Neighbours history. Only the occupied grid cells are visited; an
                # empty cell would just produce a zero-length array anyway. The cell
                # index is carried along because the collate function needs it to
                # place the encoding back on the grid.
                occupied = np.nonzero(grid)[0]
                nbsGridLocs.append(occupied)
                nbsHists.append([self.getHistory(dsId, grid[i], target, t, wholePeriod=True) for i in occupied])
                nbsVel.append([self.getVel(dsId, grid[i], target, t, wholePeriod=True) for i in occupied])
                nbsAcc.append([self.getAcc(dsId, grid[i], target, t, wholePeriod=True) for i in occupied])
                nbsdifVel.append([self.getdifVel(dsId, grid[i], target, t, wholePeriod=True) for i in occupied])
                # PlanVeh future
                planGridLocs.append(np.where(grid == centVehId)[0][0])
                planFuts.append(self.getPlanFuture(dsId, centVehId, target, t))
        return planFuts, nbsHists, \
               targsHists, targsFuts, targsLonEnc, targsLatEnc, \
               centGrid, planGridLocs, idx , nbsVel, nbsAcc ,nbsdifVel, vel_hist, acc_hist, nbsGridLocs


    ## Collate function for dataloader
    def collate_fn(self, samples):
        targs_batch_size = 0
        nbs_batch_size = 0
        for _, nbsHists, targsHists, _, _, _, _, _, _, _, _, _,_ ,_ , _ in samples:
            targs_batch_size += len(targsHists)
            nbs_batch_size += sum(len(sub_nbsHist) for sub_nbsHist in nbsHists)
        # The batch is assembled in numpy and handed to torch only at the end:
        # filling tens of thousands of tensor slices one by one dominated the
        # runtime, and the equivalent numpy stores are an order of magnitude cheaper.
        mask_np_type = np.bool_ if self.mask_num_type == torch.bool else np.uint8
        nbsHist_batch   = np.zeros((self.hist_len,  nbs_batch_size,   2), dtype=np.float32)
        nbsVel_batch = np.zeros((self.hist_len, nbs_batch_size, 1), dtype=np.float32)
        nbsAcc_batch = np.zeros((self.hist_len, nbs_batch_size, 1), dtype=np.float32)
        nbsColl_batch = np.zeros((self.hist_len, nbs_batch_size, 1), dtype=np.float32)
        vel_hist_batch = np.zeros((self.hist_len,  targs_batch_size, 1), dtype=np.float32)
        acc_hist_batch = np.zeros((self.hist_len,  targs_batch_size, 1), dtype=np.float32)
        coll_hist_batch = np.zeros((self.hist_len,  targs_batch_size, 1), dtype=np.float32)
        targsHist_batch = np.zeros((self.hist_len,  targs_batch_size, 2), dtype=np.float32)
        targsFut_batch  = np.zeros((self.fut_len,   targs_batch_size, 2), dtype=np.float32)
        lat_enc_batch   = np.zeros((targs_batch_size, 3), dtype=np.float32)
        lon_enc_batch   = np.zeros((targs_batch_size, 2), dtype=np.float32)
        planFut_batch   = np.zeros((self.plan_len,   targs_batch_size, 2), dtype=np.float32)
        idxs = []
        pos = [0, 0]
        # Fill 1 on those grid locations with neighbour
        nbsMask_batch      = np.zeros((targs_batch_size, self.grid_size[1], self.grid_size[0], self.enc_size), dtype=mask_np_type)
        planMask_batch     = np.zeros((targs_batch_size, self.grid_size[1], self.grid_size[0], self.enc_size), dtype=mask_np_type)
        targsEncMask_batch = np.zeros((targs_batch_size, self.grid_size[1], self.grid_size[0], self.enc_size), dtype=mask_np_type)
        targsFutMask_batch = np.zeros((self.fut_len,     targs_batch_size,  2), dtype=np.float32)
        targetCount = 0
        nbCount = 0
        for i, (planFuts, nbsHists, targsHists, targsFuts, targsLonEnc, targsLatEnc, centGrid, planGridLocs, idx, nbsVel, nbsAcc, nbsdifVel,vel_hist, acc_hist, nbsGridLocs) in enumerate(samples):
            idxs.append(idx)
            for j in range(len(targsFuts)):
                targsHist_batch[0:len(targsHists[j]), targetCount, :] = targsHists[j]
                vel_hist_batch[0:len(vel_hist[j]), targetCount, :] = vel_hist[j]
                acc_hist_batch[0:len(acc_hist[j]), targetCount, :] = acc_hist[j]
                targsFut_batch[0:len(targsFuts[j]), targetCount, :] = targsFuts[j]
                targsFutMask_batch[0:len(targsFuts[j]), targetCount, :] = 1
                lat_enc_batch[targetCount, :] = targsLatEnc[j]
                lon_enc_batch[targetCount, :] = targsLonEnc[j]
                planFut_batch[0:len(planFuts[j]), targetCount, :] = planFuts[j]
                # Set up neighbor, neighbor sequence length, and mask batches:
                for local, index in enumerate(nbsGridLocs[j]):
                    nbHist = nbsHists[j][local]
                    difVel = nbsdifVel[j][local][:, 0]
                    nbLen = len(nbHist)
                    nbsHist_batch[0:nbLen, nbCount, :] = nbHist
                    nbsVel_batch[0:nbLen, nbCount, :] = nbsVel[j][local]
                    nbsAcc_batch[0:nbLen, nbCount, :] = nbsAcc[j][local]
                    pos[0] = index % self.grid_size[0]
                    pos[1] = index // self.grid_size[0]
                    nbsMask_batch[targetCount, pos[1], pos[0], :] = 1
                    # Time to collision, kept only when the neighbour is actually
                    # converging: those behind the target (pos[0] <= 13) must be
                    # faster than it, those ahead of it slower.
                    converging = difVel[-1] < 0 if pos[0] <= 13 else difVel[-1] > 0
                    if converging:
                        # The squares stay in the array dtype but the root and the
                        # division are taken in float64, matching what numpy does
                        # when these are evaluated element by element.
                        dist = (nbHist[:, 0] ** 2 + nbHist[:, 1] ** 2).astype(np.float64) ** 0.5
                        divisor = difVel.astype(np.float64)
                        ttc = np.divide(dist, divisor, out=np.zeros_like(dist), where=divisor != 0)
                        nbsColl_batch[0:nbLen, nbCount, 0] = ttc
                    nbCount += 1
                    targsEncMask_batch[targetCount, 3, 12, :] = 1
                    if index == planGridLocs[j]:
                        planMask_batch[targetCount, pos[1], pos[0], :] = 1
                targetCount += 1

        return torch.from_numpy(nbsHist_batch), torch.from_numpy(nbsMask_batch), \
               torch.from_numpy(planFut_batch), torch.from_numpy(planMask_batch), \
               torch.from_numpy(targsHist_batch), torch.from_numpy(targsEncMask_batch), \
               torch.from_numpy(targsFut_batch), torch.from_numpy(targsFutMask_batch), \
               torch.from_numpy(lat_enc_batch), torch.from_numpy(lon_enc_batch), idxs, \
               torch.from_numpy(nbsVel_batch), torch.from_numpy(nbsAcc_batch), torch.from_numpy(nbsColl_batch), \
               torch.from_numpy(vel_hist_batch), torch.from_numpy(acc_hist_batch), torch.from_numpy(coll_hist_batch)
    # _______________________________________________________________________________________________________________________________________
