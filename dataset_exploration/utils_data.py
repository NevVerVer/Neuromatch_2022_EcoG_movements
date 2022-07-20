import os
import mne
import nilearn
from scipy import signal, stats

# for nwb files and AJILE dataset
from brunton_lab_to_nwb.brunton_widgets import BruntonDashboard
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import dandi
import warnings


def load_nwb(subject, session, use_cloud_file=True):
    """
    Get .nwb file from AJILE12 dataset
    
    Parameters:
        subject: int
            Subject ID (1 - 12)
        session: int
            Session ID (>=3)
        use_cloud_file: bool
            Default=True; Whether to use local or cloud nwb.
    Returns:
        nwb file in PyNWB readable format
    
    """
    sbj = subject
    
    fname = "sub-{0:>02d}_ses-{1:.0f}_behavior+ecephys.nwb".format(sbj, session)
    local_fpath = "sub-{0:>02d}_ses-{1:.0f}_behavior+ecephys.nwb".format(sbj, session)
    cloud_fpath = "sub-{0:>02d}/sub-{0:>02d}_ses-{1:.0f}_behavior+ecephys.nwb".format(sbj, session) 
    
    warnings.filterwarnings("ignore")

    if use_cloud_file:
        print("Retrieving cloud file...")
        with DandiAPIClient() as client:

            print("Starting a Dandi streaming client...")

            asset = client.get_dandiset("000055", "draft").get_asset_by_path(cloud_fpath)
            s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)

        io = NWBHDF5IO(s3_path, mode='r', load_namespaces=True, driver='ros3')
    else:
        print("Retrieving local file..")
        local_file_path = os.path.join(os.getcwd(), local_fpath)
        io = NWBHDF5IO(local_file_path, mode='r', load_namespaces=False)

    print("Reading a file...")

    nwb = io.read()

    print(f"Returning nwb for {fname}")
    
    return nwb
