# Week 1: Data Foundation

**Date**: February 3--9, 2025
**Focus**: Data acquisition, manifest building, annotation cleaning, camera correction, and clip extraction.

---

## 1. Data Description

### 1.1 Animals and Housing

Six groups (cohorts) of calves were observed over a five-month data collection period spanning July to November 2024. Each group was recorded on two observation days: Day 1 (the day of grouping) and Day 4 (three days post-grouping), except for Group 4, which was recorded on Day 1 only. A total of 26 individual calves were identified across the six groups as either initiators or receivers of cross-sucking events.

### 1.2 Video Recording

Each pen was monitored by a network video recorder (NVR, identifier N884A6) equipped with four surveillance cameras. Videos were recorded continuously at a resolution of 3840 x 2160 pixels (4K Ultra HD) and a frame rate of 15 frames/sec. Recordings were segmented into approximately 30-minute files, yielding 208 unique video segments across all groups and observation days. Each camera channel was identified by a channel number embedded in the filename (e.g., `ch08` for Camera 8).

Of the four cameras per group, one was positioned to capture an outdoor or wide-angle view and was not used for behavioral annotation:

| Groups   | Outdoor Camera |
|----------|---------------|
| 1, 2     | Camera 10     |
| 3, 4, 5, 6 | Camera 8   |

The remaining three indoor cameras per group provided close-range views of the pen interior. The primary annotation camera for each group was identified through manual verification of extracted clips:

| Group | Primary Camera | Secondary Camera(s) |
|-------|---------------|---------------------|
| 1     | Camera 8      | Camera 9            |
| 2     | Camera 16     | Camera 12           |
| 3     | Camera 12     | Camera 14           |
| 4     | Camera 3      | Camera 5, Camera 10 |
| 5     | Camera 1      | Camera 9            |
| 6     | Camera 12     | Camera 14           |

### 1.3 Observation Schedule

| Group | Day 1 Date     | Day 4 Date     | Events (Day 1) | Events (Day 4) | Total |
|-------|----------------|----------------|-----------------|-----------------|-------|
| 1     | July 6, 2024   | July 9, 2024   | 316             | 355             | 671   |
| 2     | July 14, 2024  | July 17, 2024  | 171             | 65              | 236   |
| 3     | August 2, 2024 | August 5, 2024 | 157             | 174             | 331   |
| 4     | Sept. 13, 2024 | --             | 68              | --              | 68    |
| 5     | Oct. 16, 2024  | Oct. 19, 2024  | 68              | 5               | 73    |
| 6     | Nov. 7, 2024   | Nov. 10, 2024  | 243             | 275             | 518   |
| **Total** |            |                | **1,023**       | **874**         | **1,897** |

---

## 2. Data Annotation

### 2.1 Behavioral Events

Cross-sucking events were annotated from the video recordings by a trained observer. Each event was coded with a start time, end time, and the following attributes:

- **Behavior**: The body part targeted by the sucking calf --- ear, tail, teat, or other.
- **Initiator ID / Receiver ID**: Individual calf identifiers for the animal performing and receiving the cross-sucking.
- **Ended by**: Whether the event was terminated by the initiator, the receiver, or by external interruption (e.g., a third calf intervening).
- **Pen location**: The spatial zone within the pen where the event occurred --- front, middle, or back.

### 2.2 Behavior Distribution

A total of 1,897 cross-sucking events were annotated across all six groups. The distribution of behaviors was heavily skewed toward ear-sucking:

| Behavior   | Count | Proportion |
|------------|-------|------------|
| Ear        | 1,651 | 87.0%      |
| Tail       | 225   | 11.9%      |
| Teat       | 12    | 0.6%       |
| Other      | 9     | 0.5%       |

For the binary classification task, only ear-sucking and tail-sucking events were retained (n = 1,876), yielding a 7.3:1 class imbalance.

### 2.3 Event Duration

Event durations ranged from 0 to 272 seconds (mean = 6.9 s, median = 4.0 s, IQR = 1--8 s). A substantial proportion of events were very brief: 196 events (10.3%) had a recorded duration of 0 seconds, and 515 events (27.1%) lasted 1 second or less. Nine events exceeded 60 seconds in duration. The short-duration events likely reflect momentary contacts that were still coded as discrete events by the annotator.

### 2.4 Event Termination

The majority of events were ended by the initiator (n = 1,082; 57.0%), followed by the receiver (n = 787; 41.5%). A small number of events were terminated by external interruption (n = 27; 1.4%), such as a third calf intervening or the animals being disturbed.

### 2.5 Pen Location

Events were distributed across three spatial zones within the pen: middle (n = 1,019; 53.7%), front (n = 591; 31.2%), and back (n = 287; 15.1%).

---

## 3. Data Processing Pipeline

### 3.1 Manifest Building and Event Linking

A video manifest was constructed by scanning all 12 group/day folders on the storage drive, cataloguing 208 video files across 24 camera channels. Each annotated event was linked to its corresponding video file by matching the event's start time to the video segment's time window (encoded in the filename). All 1,897 events were successfully linked (100% linking rate).

### 3.2 Camera Correction

An initial quality audit of the first 100 extracted clips revealed that approximately 46% did not contain the annotated behavioral event. The root cause was traced to the event-linking algorithm, which iterated over cameras in arbitrary (JSON key) order and selected the first camera with a matching time window. Since all four cameras in a group covered identical time spans, the algorithm frequently selected an outdoor or otherwise unsuitable camera.

A camera correction step was implemented in which each event was re-linked to the verified primary annotation camera for its group (see Table in Section 1.2), with outdoor cameras excluded. This correction changed the camera assignment for 1,691 of 1,897 events (89.1%).

### 3.3 Clip Extraction

Event clips were extracted at full source resolution (3840 x 2160) using FFmpeg with H.264 encoding (CRF 20, medium preset). Each clip included a temporal padding of 3 seconds before and after the annotated event boundaries (minimum clip duration: 6 seconds), with asymmetric compensation when padding extended beyond video boundaries. A total of 1,897 clips were extracted to `data/processed/clips_v4/`.

The full-resolution extraction was chosen to support downstream annotation tasks including bounding box annotation, interaction keypoint labeling, and motion reconstruction, where spatial detail is critical.

### 3.4 Train/Validation/Test Split

A bout-grouped intra-video split was designed to prevent temporal leakage while maintaining the intra-video property (both train and validation contain events from the same videos):

1. **Binary filtering**: Retained ear and tail events only (n = 1,876).
2. **OOD holdout**: Groups 4 and 5 were held out as an out-of-distribution (OOD) test set (n = 141), ensuring that Cameras 3 and 1 are never seen during training.
3. **Bout grouping**: Within each video, consecutive events separated by fewer than 30 seconds were grouped into temporal bouts (n = 871 bouts in the development set). All events within a bout were assigned to the same split.
4. **Intra-video split**: Bouts within each video were split approximately 85/15 into train and validation, stratified by the presence of tail events where possible.

| Split     | Events | Tail Events | Tail % | Groups     | Cameras        |
|-----------|--------|-------------|--------|------------|----------------|
| Train     | 1,374  | 175         | 12.7%  | 1, 2, 3, 6 | 8, 16, 12, 12 |
| Val       | 361    | 38          | 10.5%  | 1, 2, 3, 6 | 8, 16, 12, 12 |
| Test (OOD)| 141    | 12          | 8.5%   | 4, 5       | 3, 1           |

Verification confirmed: zero bout leakage across splits, zero temporal leakage (no cross-split event pairs within 30 seconds in the same video), and 91% of videos appearing in both train and validation sets.

---

## 4. Issues and Ongoing Work

1. **Remaining camera mismatches**: A post-extraction audit identified a small number of clips where the assigned camera does not clearly show the annotated event (e.g., event 1247 in Group 4; events 1566--1577 in Group 6). A full audit is in progress; affected events will be re-extracted from alternative cameras or flagged for exclusion.
2. **Zero-duration events**: 196 events with 0-second duration were retained and extracted with the standard 6-second minimum clip window. These may require manual review to confirm event presence.
3. **Smoke test**: A pipeline smoke test (dataloader validation and short training run) is pending.

---

## 5. Next Steps

- Complete full clip audit across all 1,897 events.
- Re-extract or flag clips with camera/visibility issues.
- Run pipeline smoke test with `configs/train_binary_v4.yaml`.
- Begin baseline training (R3D-18, focal loss, balanced sampling).
