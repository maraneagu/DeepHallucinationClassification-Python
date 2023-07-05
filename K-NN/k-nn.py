import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# --- DATA ---

# LOADING THE DATA FOR OUR MODEL
trainingDataFrame = pd.read_csv('/kaggle/input/unibuc-ml2/train.csv')
validationDataFrame = pd.read_csv('/kaggle/input/unibuc-ml2/val.csv')
testingDataFrame = pd.read_csv('/kaggle/input/unibuc-ml2/test.csv')

# DEFINING THE IMAGES DIRECTORIES FOR OUR THREE STAGES
trainingImagesDirectory = '/kaggle/input/unibuc-ml2/train_images'
validationImagesDirectory = '/kaggle/input/unibuc-ml2/val_images'
testingImagesDirectory = '/kaggle/input/unibuc-ml2/test_images'

# DEFINING THE GENERAL IMAGE SIZE
imageSize = (64, 64)


# --- DATA PROCESSING ---

# FUNCTION TO PROCESS THE DATA THAT WE GET FROM THE DATAFRAME AND THE DIRECTORIES
def processing(dataFrame, imagesDirectory, isTestingImagesDirectory=False):
    images = []
    labels = []

    # ITERATING THROUGH EACH LINE OF THE DATAFRAME, EXTRACTING THE "IMAGE" AND THE "CLASS", IF IT EXISTS
    for index, row in dataFrame.iterrows():

        # CREATING THE FULL PATH TO THE IMAGE FILE
        imagePath = imagesDirectory + '/' + row['Image']

        # CONVERTS THE IMAGE TO THE RGB COLOR MODE, ENSURING THAT THE IMAGE HAS THREE COLOR CHANNELS (RED, GREEN, BLUE), REQUIRED BY THE CNN MODEL
        image = Image.open(imagePath).convert('RGB')

        # RESIZING THE IMAGE TO THE REQUIRED SIZE, THUS ENSURING THAT ALL THE IMAGES HAVE THE SAME DIMENSIONS
        image = image.resize(imageSize)

        # CREATING A NUMERICAL REPRESENTATION OF THE IMAGE THAT CAN BE PROCESSED BY THE CNN MODEL
        image = np.array(image)

        # ADDING THE NEW IMAGE TO OUR ARRAY OF ALREADY PROCESSED IMAGES
        images.append(image)

        # IF WE ARE PROCESSING THE TRAINING IMAGES OR THE VALIDATION IMAGES, WE HAVE THE "CLASS" COLUMN AND WE ARE ADDING IT TO THE EXISTING CLASSES
        if not isTestingImagesDirectory:
            label = row['Class']

            # ADDING THE NEW LABEL TO OUR ARRAY OF ALREADY PROCESSED LABELS
            labels.append(label)

    # WE ARE CONVERTING THE IMAGES AND THE LABELS PROCESSED TO AN ARRAY
    images = np.array(images)
    if not isTestingImagesDirectory:
        labels = np.array(labels)
        return images, labels
    else:
        return images


# EXTRACTING THE DATA FOR OUR MODEL FROM THE DATAFRAME
trainingImages, trainingLabels = processing(trainingDataFrame, trainingImagesDirectory)
validationImages, validationLabels = processing(validationDataFrame, validationImagesDirectory)
testingImages = processing(testingDataFrame, testingImagesDirectory, isTestingImagesDirectory=True)

# NORMALIZING THE IMAGES FROM OUR DATA TO THE RANGE OF [0,1], ENSURING A MORE EFICIENT TRAINING, PREVENTING A SPECIFIC FEATURE FROM DOMINATING
trainingImages = trainingImages / 255.0
validationImages = validationImages / 255.0
testingImages = testingImages / 255.0

# CONVERT LABELS TO THE CATEGORICAL FORMAT, WE TRANSFORM THE CLASSES INTO A BINARY MATRIX REPRSENTATION, EACH CLASS REPRESENTED BY A BINARY VECTOR WITH A SINGLE ELEMENT SET TO 1 AND THE REST 0
labelEncoder = LabelEncoder()

# WE ARE USING "fit_transform" BECAUSE IT INVOLVES BOTH LEARNING THE MAPPING OF CLASSES TO INTEGERS AND THEN APPLYING THE TRANSFORMATION TO THE LABELS
trainingLabels = labelEncoder.fit_transform(trainingLabels)

# WE ARE USING "transform" BECAUSE IT APPLIES THE LEARNED MAPPING FROM THE trainingLabels AND THUS IT NEEDS ONLY TO APPLY THE TRANSFORMATION TO THE LABELS
validationLabels = labelEncoder.transform(validationLabels)


# --- THE KNN MODEL ---

trainingImages = trainingImages.reshape(len(trainingImages), -1)
validationImages = validationImages.reshape(len(validationImages), -1)
testingImages = testingImages.reshape(len(testingImages), -1)

# CREATE AND TRAIN THE KNN MODEL
model = KNeighborsClassifier(n_neighbors=5)
model.fit(trainingImages, trainingLabels)

# THE TRAINING ACCURACY FOR OUR MODEL
trainingAccuracy = model.score(trainingImages, trainingLabels)
print("Training Accuracy:", trainingAccuracy)

# THE VALIDATION ACCURACY FOR OUR MODEL
validationAccuracy = model.score(validationImages, validationLabels)
print("Validation Accuracy:", validationAccuracy)

# PREDICT LABELS FOR TESTING IMAGES
predictedLabels = model.predict(testingImages)


# --- THE SUBMISSION FILE ---

# CREATING THE SUBMISSION FILE
submissionDataFrame = pd.DataFrame({'Image': testingDataFrame['Image'], 'Class': predictedLabels})
submissionDataFrame.to_csv('/kaggle/working/submission.csv', index=False)

# --- THE REPORT CLASSIFICATION ---

# CONVERTING THE PREDICTED AND THE VALIDATION CLASSES TO A NUMERIC FORMAT, AND RESIZING IT AFTER THE VALIDATION
predictions = model.predict(validationImages)
predictedLabels = labelEncoder.inverse_transform(predictions)
validationLabels = labelEncoder.transform(validationLabels)

# CREATING A DATA FRAME TO STORE THE METRICS FOR EACH CLASS
reportDataFrame = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall'])

# LOOPING THROUGH ALL THE CLASSES
for i, label in enumerate(labelEncoder.classes_):
    # CALCULATING THE ACCURACY FOR THE CLASSES
    accuracy = accuracy_score(validationLabels == i, predictedLabels == i)

    # CALCULATING THE PRECISION FOR THE CLASS
    precision = precision_score(validationLabels == i, predictedLabels == i)

    # CALCULATING THE RECALL FOR THE CLASS
    recall = recall_score(validationLabels == i, predictedLabels == i)
    reportDataFrame.loc[i] = [accuracy, precision, recall]

# CREATING THE CLASSIFICATION REPORT FILE
reportDataFrame.to_csv('/kaggle/working/classification_report.csv', index=True, sep='\t', index_label='Class')

# PLOTTING THE ACCURACY CHART
fig_accuracy, ax_accuracy = plt.subplots(figsize=(12, 6))
sns.barplot(x=reportDataFrame.index, y='Accuracy', data=reportDataFrame, color='darkgrey', label='Accuracy')
ax_accuracy.set_xlabel('Class')
ax_accuracy.set_ylabel('Accuracy')
ax_accuracy.set_title('Accuracy')
plt.xticks(rotation=90)
ax_accuracy.legend(loc='lower right')
plt.tight_layout()
fig_accuracy.savefig('/kaggle/working/accuracy_chart.png', dpi=300)

# PLOTTING THE PRECISION CHART
fig_precision, ax_precision = plt.subplots(figsize=(12, 6))
sns.barplot(x=reportDataFrame.index, y='Precision', data=reportDataFrame, color='gray', label='Precision')
ax_precision.set_xlabel('Class')
ax_precision.set_ylabel('Precision')
ax_precision.set_title('Precision')
plt.xticks(rotation=90)
ax_precision.legend(loc='lower right')
plt.tight_layout()
fig_precision.savefig('/kaggle/working/precision_chart.png', dpi=300)

# PLOTTING THE RECALL CHART
fig_recall, ax_recall = plt.subplots(figsize=(12, 6))
sns.barplot(x=reportDataFrame.index, y='Recall', data=reportDataFrame, color='lightgrey', label='Recall')
ax_recall.set_xlabel('Class')
ax_recall.set_ylabel('Recall')
ax_recall.set_title('Recall')
plt.xticks(rotation=90)
ax_recall.legend(loc='lower right')
plt.tight_layout()
fig_recall.savefig('/kaggle/working/recall_chart.png', dpi=300)

# --- THE CONFUSION MATRIX ---

# TESTING THE MODEL
validationImages = validationImages.reshape(len(validationImages), -1)
predictions = model.predict(validationImages)
predictedLabels = labelEncoder.inverse_transform(predictions)

# CALCULATING THE CONFUSION MATRIX FOR THE VALIDATION SET
confusionMatrix = confusion_matrix(validationLabels, predictedLabels)

# CONVERTING THE CONFUSION MATRIX TO A DATAFRAME
confusionMatrixDataFrame = pd.DataFrame(confusionMatrix)

# CREATING THE CONFUSION MATRIX FILE
confusionMatrixDataFrame.to_csv('/kaggle/working/confusion_matrix.csv', index=False)