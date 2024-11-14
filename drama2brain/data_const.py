SUBJ_NAME = ["DM01",
             "DM03",
             "DM06",
             "DM07",
             "DM09",
             "DM11"]
VIDEO_NAMES = [
                "BigBangTheory1",
                "BreakingBad",
                "Crown",
                "DreamGirls",
                "GIS1",
                "GIS2",
                "Glee",
                "Heroes",
                "Mentalist",
                "SUITS"
                ]
VIDEO_NUMS = {
            "BigBangTheory1":2,
            "BreakingBad":5,
            "Crown":5,
            "DreamGirls":9,
            "GIS1":2,
            "GIS2":2,
            "Glee":4,
            "Heroes":5,
            "Mentalist":4,
            "SUITS":3
            }
VIDEO_FRAMES = {
                "BigBangTheory1":[747, 647],
                "BreakingBad":[679, 709, 715, 722, 734],
                "Crown":[638, 625, 711, 844, 769],
                "DreamGirls":[764, 821, 817, 592, 1118, 691, 883, 684, 1271],
                "GIS1":[804, 726],
                "GIS2":[673, 852],
                "Glee":[722, 812, 601, 808],
                "Heroes":[617, 639, 512, 774, 732],
                "Mentalist":[737, 607, 660, 760],
                "SUITS":[754, 968, 680],
                }
TRAINVALID_FRAMES = {
                "BigBangTheory1":["TRAIN", "VALID"],
                "BreakingBad":["TRAIN", "TRAIN", "TRAIN", "TRAIN", "VALID"],
                "Crown":["TRAIN", "TRAIN", "TRAIN", "TRAIN", "VALID"],
                "DreamGirls":["TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "VALID", "VALID"],
                "GIS1":["TRAIN", "TRAIN"],
                "GIS2":["TRAIN", "VALID"],
                "Glee":["TRAIN", "TRAIN", "TRAIN", "VALID"],
                "Heroes":["TRAIN", "TRAIN", "TRAIN", "TRAIN", "VALID"],
                "Mentalist":["TRAIN", "TRAIN", "TRAIN", "VALID"],
                "SUITS":["TRAIN", "TRAIN", "VALID"],
}
TRAINVALID_FRAMES_DATASPLIT_A = {
                "BigBangTheory1":["", ""],
                "BreakingBad":["TRAIN", "TRAIN", "", "", "VALID"],
                "Crown":["TRAIN", "TRAIN", "", "", "VALID"],
                "DreamGirls":["TRAIN", "TRAIN", "TRAIN", "TRAIN", "", "", "", "VALID", "VALID"],
                "GIS1":["TRAIN", "TRAIN"],
                "GIS2":["", "VALID"],
                "Glee":["TRAIN", "", "", "VALID"],
                "Heroes":["TRAIN", "TRAIN", "", "", "VALID"],
                "Mentalist":["TRAIN", "TRAIN", "", "VALID"],
                "SUITS":["TRAIN", "", "VALID"],
}

TRAINVALID_FRAMES_DATASPLIT_B = {
                "BigBangTheory1":["", ""],
                "BreakingBad":["", "", "TRAIN", "TRAIN", "VALID"],
                "Crown":["", "", "TRAIN", "TRAIN", "VALID"],
                "DreamGirls":["", "", "", "", "TRAIN", "TRAIN", "TRAIN", "VALID", "VALID"],
                "GIS1":["", ""],
                "GIS2":["TRAIN", "VALID"],
                "Glee":["", "TRAIN", "TRAIN", "VALID"],
                "Heroes":["", "", "TRAIN", "TRAIN", "VALID"],
                "Mentalist":["", "", "TRAIN", "VALID"],
                "SUITS":["", "TRAIN", "VALID"],
}
VIDEO_CVIDX = {
            "BigBangTheory1":1,
            "BreakingBad":0,
            "Crown":0,
            "DreamGirls":1,
            "GIS1":2,
            "GIS2":2,
            "Glee":2,
            "Heroes":3,
            "Mentalist":3,
            "SUITS":2
            }

"""
Please set the path according to your environment.
"""
FMRI_RAW_DIR = ""
FMRI_PRE_DIR = ""
FMRI_BBREGISTER_DIR = ""
FMRI_EXP2019_DIR = ""
STIM_DIR = "./data/stim_features/"
AUDIO_DIR = ""
FRAME_DIR = ""
CAPTION_DIR = ""
RESULT_DIR = f"./data/encoding/"
