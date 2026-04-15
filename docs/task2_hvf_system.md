# Task 1: Glaucoma Severity Classification

This task involves categorizing the functional loss of a patient's vision based on a single static perimetry snapshot.

**TODO**: remove the link below once the model card is complete.

See example of a [Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf)


## Getting the labels

The standard way to define severity from visual field data is using the Hodapp-Parrish-Anderson (HPA) criteria, which is based on the Mean Deviation (MD) value. ([paper](https://www.researchgate.net/publication/270223566_Disease_severity_in_newly_diagnosed_glaucoma_patients_with_visual_field_loss_Trends_from_more_than_a_decade_of_data))

The UWHVF dataset contains a **Mean Total Deviation** field (`MTD`)

| Severity | MD Range | Meaning |
| :--- | :--- | :--- |
| Mild | > - 6 dB | Early-stage vision loss |
| Moderate | - 6 dB to -12 dB | Significant field defects |
| Severe | < -12 dB| Advanced glaucoma/vision loss |

## Model Information

### Description

A basic [CNN model](../src/glaucoma_vf/models/hvf_cnn_classifier.py).

### Model dependencies

There are no model dependencies, the model was trained from scratch.

### Inputs

8x9 grid of 54 HVF points, the empty cells were filled with a value of `100` to fill up the grid.

Humphrey Visual Field: 54 points (52 test points plus two blind spot locations)

### Outputs

One of
  - **Mild**: (MD > -6 dB)
  - **Moderate**: (-6 dB ≥ MD ≥ -12 dB)
  - **Severe**: (MD < -12 dB)
