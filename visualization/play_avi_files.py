# quick script to play videos saved out of load_from_DLC.py by functions in check_tracking.py

import numpy as np
import cv2

# cap = cv2.VideoCapture('/Users/dylanmartins/data/Niell/PreyCapture/Cohort3Outputs/J463c(blue)_110719/analysis_test_01/_mouse_J463c_trial_1_110719_09.avi')
cap = cv2.VideoCapture('/Users/dylanmartins/data/Niell/PreyCapture/WorldCamCohortOutputs/CuratedDataset_ObjectArena_J463b_112619_1_2/analysis_test_00.avi')

while (1):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()