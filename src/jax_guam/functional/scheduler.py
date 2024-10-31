# % INPUTS:
# %   Trim Files (delineated in m-script Concatenate_Trim_Files.m)
# % OUTPUTS:
# %   Trim/Controller output .mat file for use in Generalized Urban Air Mobility
# %   (GUAM) or Generalized Vehicle Sim (GVS)
# % OTHER UTILIZED FUNCTIONS:
# %   Concatenate_Trim_Files.m: % Concatenates trim files
# %   ctrl_lon.m  % Designs unified longitudinal controller (see Ref.)
# %   ctrl_lat.m  % Designs unified lateral controller (see Ref.)


from attrs import define
from jax_guam.utils.paths import data_dir
from jax_guam.data.read_data import read_data
import pdb
import h5py
import numpy as np
from jax_guam.data.read_data import read_data


# def ctrl_lon():


# @define
# class CtrlSchdulerCfg:
#     ref_traj_on: bool = True
#     feedback_current: bool = True
#     position_error: bool = False



from control.matlab import *



def Rx(self, x):
    return np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])



def Ry(self, x):
    return np.array([[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0, np.cos(x)]])



def Rz(self, x):
    return np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])



def hat(self, x):
    return np.array(
        [
            [0, -float(x[2]), float(x[1])],
            [float(x[2]), 0, -float(x[0])],
            [float(-x[1]), float(x[0]), 0],
        ]
    )



def get_lin_dynamics_heading(
    tiltwing,
    xeq,
    ueq,
    NS,
    NP,
):


    # define some unit vectors, rotations, and skew symmetric matrix functions
    rho = 0.0023769
    e1 = [1, 0, 0]
    e2 = [0, 1, 0]
    e3 = [0, 0, 1]



def get_long_dynamics_heading(xeq, ueq, NS, NP, FreeVar_pnt, Trans_pnt):


    A, B, C, D, XU0 = get_lin_dynamics_heading(
        aircraft,
        xeq,
        ueq,
        NS,
        NP,
    )


    A_full = A
    B_full = B
    C_full = C
    D_full = D


    x_idx = [6, 8, 10, 4]
    u_idx = [4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0]
    y_idx = x_idx


    Alon = A[x_idx, x_idx]
    Blon = B[x_idx, u_idx]
    Clon = C[y_idx, x_idx]
    Dlon = D[y_idx, u_idx]


    return Alon, Blon, Clon, Dlon



def ctrl_lon(xu_eq, q, r, wc, FreeVar_pnt, Trans_pnt, i, j):
    """
    args:
        aircraft: aero/propulsive model
        xu_eq: Vector of trim states and controls
        rho: scalar value of air density (slugs/ft^3)
        grav: scalar value of gravity acceleration (ft/sec^2)
        q:  Desired state cost weight
        r: Desired control acceleration cost weights
        wc: Desired control allocation weights
        FreeVar_pnt: Arrays of variables (including effectors) avail for control use
        Trans_pnt: Column vector of transition start velocity and transition end velocity


    returns:
        out: Structure of linerized unified controller gains etc
        ctrl_error: Scalar flag indicating error in controller design at current trim point
    """


    ctrl_error = 0


    Q = np.diag(q)
    R = np.diag(r)
    Wc = np.diag(wc)


    xeq = xu_eq[:8]
    ueq = [xu_eq[8:]]


    NS = 4
    NP = 9


    mat = read_data(data_dir() / "trim_table_Poly_ConcatVer4p0.mat")
    # pdb.set_trace()
    Alon = mat["Ap_lon_interp"][:, :, i, j]
    Blon = mat["Bp_lon_interp"][:, :, i, j]
    Clon = mat["Cp_lon_interp"][:, :, i, j]
    Dlon = mat["Dp_lon_interp"][:, :, i, j]


    Alon_lin, Blon_lin, Clon_lin, Dlon_lin = get_long_dynamics_heading(
        xeq, ueq, NS, NP, FreeVar_pnt, Trans_pnt
    )


    # import pdb


    print(Alon, Alon_lin)
    print(Blon, Blon_lin)
    print(Clon, Clon_lin)
    print(Dlon, Dlon_lin)


    pdb.set_trace()
    if xeq[0] > Trans_pnt[1]:  # Zero out the lifting rotors after the trans regime ends
        Blon[:, :8] = np.zeros(4, 8)


    # Size definitions
    Nx = 4
    Ni = 3
    Nr = 3
    Nu = 11
    Nv = 1
    Nmu = 3
    Nxi = 3


    # Performance design with general acceleration inputs
    Av = Alon[:3, :3]
    Bv = np.eye(Nxi)
    Cv = np.eye(Nxi)
    Dv = np.zeros((Nxi, Nxi))
    # pdb.set_trace()
    At = np.vstack(
        (np.hstack((np.zeros((Ni, Ni)), Cv)), np.hstack((np.zeros((Nxi, Ni)), Av)))
    )
    Bt = np.vstack((Dv, Bv))
    # pdb.set_trace()
    # LQR optimal feedback gains
    Kc, P, CLP = lqr(At, Bt, Q, R)
    Ki0 = Kc[:, :Ni]
    Kx0 = Kc[:, Ni : Ni + Nxi]


    # control Allocation
    # pdb.set_trace()
    Bu = np.concatenate((Blon[:3, :], Alon[:3, 3].reshape(3, 1)), 1)


    Wc_inv = np.linalg.inv(Wc)
    M = Wc_inv @ Bu.T @ np.linalg.inv(Bu @ Wc_inv @ Bu.T)


    A = Alon
    B = Blon
    C = Clon[:3, :]
    D = Dlon[:3, :]


    Kx = Kx0 @ [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    Ki = Ki0


    pdb.set_trace()
    Mu = M[:Nu, :]
    Mv = M[Nu + Nv, :]


    Cv = Clon[4, :]
    Kv = np.array([0, 0, 1]).T


    F = np.eye(3)
    G = np.zeros(3)


    Acl = np.vstack(
        (
            np.hstack((Kv @ Mv @ Ki, Kv @ Mv @ Kx + Kv @ Cv + C)),
            np.hstack((-B @ Mu @ Ki, A - B @ Mu @ Kx)),
        )
    )


    if np.sum(np.linal.eig(Acl) > 0):
        print("trim point has unstable poles\n")
        ctrl_error = 1
    return out, ctrl_error



class CtrlSchduler:
    def __init__(self):


        # Process the data
        self.XEQ = np.array(
            h5py.File(data_dir() / "Trim_poly_XEQ_ConcatV4p0.mat", "r").get("XEQ")
        )
        self.FreeVar_Table = np.array(
            h5py.File(data_dir() / "Trim_poly_XEQ_ConcatV4p0.mat", "r").get(
                "FreeVar_Table"
            )
        )
        self.R_concat = np.array(
            h5py.File(data_dir() / "Trim_poly_XEQ_ConcatV4p0.mat", "r").get("R_concat")
        )
        self.Trans_Table = np.array(
            h5py.File(data_dir() / "Trim_poly_XEQ_ConcatV4p0.mat", "r").get(
                "Trans_Table"
            )
        )
        self.UH_concat = np.array(
            h5py.File(data_dir() / "Trim_poly_XEQ_ConcatV4p0.mat", "r").get("UH_concat")
        )
        self.WH_concat = np.array(
            h5py.File(data_dir() / "Trim_poly_XEQ_ConcatV4p0.mat", "r").get("WH_concat")
        )


        #  1: use polynomial aero/propulsive (A/P) database
        self.poly = 1


        # Parse the trim table
        self.XEQ_TABLE = np.swapaxes(self.XEQ, 0, 2)
        # Determine size of gain scheduling arrays
        #    UH     WH      R
        self.N_trim = 28  # np.shape(self.XEQ)[1]
        self.M_trim = 3  # np.shape(self.XEQ)[0]
        self.L_trim = 1


        # Initial output table
        self.XEQ0 = np.zeros((21, self.N_trim, self.M_trim, self.L_trim))


        # Specify necessary constants (e.g., gravity and air density)
        self.rho = 0.0023769  # slugs/ft^3
        self.grav = 32.17405  # ft/sec^2
        self.ft2kts = 1 / (
            1852.0 / 0.3048 / 3600
        )  # Obtained from setUnits... Old = 0.592484; % Conversion from ft/sec to kts


        # Longtitudinal Control
        # State Cost
        self.Qlon0 = np.array(
            [
                0.01,
                0.01,
                1000,
                0,
                0,
                0,
            ]
        ).T


        # Control acceleration cost
        self.Rlon0 = np.array([1, 1, 1]).T


        # Control allocation weighting
        self.Wlon0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1000, 10000000, 0.1]).T


        self.Qlon = np.tile(
            self.Qlon0[:, None, None], [1, self.N_trim, self.M_trim]
        )  # 6 x 28 x 3
        self.Rlon = np.tile(
            self.Rlon0[:, None, None], [1, self.N_trim, self.M_trim]
        )  # 3 x 28 x 3
        self.Wlon = np.tile(
            self.Wlon0[:, None, None], [1, self.N_trim, self.M_trim]
        )  # 12 x 28 x 3


        # Lateral Control
        # State Cost
        # vi   pi   ri   v p r
        self.Qlat0 = np.array(
            [
                0.01,
                1000,
                1000,
                0,
                0,
                0,
            ]
        ).T


        # Control acceleration cost
        self.Rlat0 = np.array([1, 1, 1]).T


        # Control allocation weighting
        self.Wlat0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1000, 1000, 1]).T


        self.Qlat = np.tile(
            self.Qlat0[:, None, None], [1, self.N_trim, self.M_trim]
        )  # 6 x 28 x 3
        self.Rlat = np.tile(
            self.Rlat0[:, None, None], [1, self.N_trim, self.M_trim]
        )  # 3 x 28 x 3
        self.Wlat = np.tile(
            self.Wlat0[:, None, None], [1, self.N_trim, self.M_trim]
        )  # 12 x 28 x 3


        # Specify system sizes
        self.Nx_lon = 4
        self.Ni_lon = 3
        self.Nu_lon = 11
        self.Nr_lon = 3
        self.Nv_lon = 1


        self.Nx_lat = 4
        self.Ni_lat = 3
        self.Nu_lat = 10
        self.Nr_lat = 2
        self.Nv_lat = 1


    def design(self):
        LON = []
        LAT = []
        UH = []
        WH = []
        R = []


        for k in range(self.L_trim):  # 1
            for j in range(self.M_trim):  # 3
                for i in range(self.N_trim):  # 28
                    # pdb.set_trace()
                    trim_pnt = self.XEQ_TABLE[:, i, j]
                    FreeVar_pnt = self.FreeVar_Table[:, :, j]
                    Trans_pnt = self.Trans_Table[j, :]
                    lon, lon_err = ctrl_lon(
                        trim_pnt,
                        self.Qlon[:, i, j],
                        self.Rlon[:, i, j],
                        self.Wlon[:, i, j],
                        FreeVar_pnt,
                        Trans_pnt,
                        i,
                        j,
                    )


                    # Concatenate longitudinal and lateral controller structures
                    LON.append(lon)



if __name__ == "__main__":
    fun = CtrlSchduler()
    fun.design()